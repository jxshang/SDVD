import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import PAD, get_previous_user_mask
from model.transformer import Transformer
from model.dynamic_graph_encoder import DynamicVHGAE
from model.static_graph_encoder import StaticGraphEnocoder


class GatedFusion(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size, input_size), nn.Tanh(), nn.Linear(input_size, 1)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1.unsqueeze(0), x2.unsqueeze(0)], dim=0)
        score = F.softmax(self.transform(x), dim=0)
        out = torch.sum(score * x, dim=0)
        return out


class SDVD(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.user_size = args.user_size
        self.cas_size = args.cas_size
        self.drop_prob = args.drop_prob
        self.n_heads = args.n_heads
        self.device = args.device
        self.graph_drop_prob = args.graph_drop_prob
        self.n_interval = args.n_interval

        self.user_embedding = nn.Embedding(
            self.user_size, self.hidden_dim, padding_idx=PAD
        )
        self.static_graph_encoder = StaticGraphEnocoder(
            num_nodes=self.user_size,
            hidden_dim=self.hidden_dim,
            drop_prob=self.drop_prob,
            graph_drop_prob=self.graph_drop_prob,
        )
        self.dynamic_graph_encoder = DynamicVHGAE(
            hidden_dim=self.hidden_dim,
            n_interval=self.n_interval,
            drop_prob=self.drop_prob,
            graph_drop_prob=self.graph_drop_prob,
        )
        self.dropout = nn.Dropout(self.drop_prob)
        self.fusion1 = GatedFusion(input_size=self.hidden_dim)
        self.fusion2 = GatedFusion(input_size=self.hidden_dim)
        self.attention1 = Transformer(
            d_model=self.hidden_dim, n_head=self.n_heads, drop_prob=self.drop_prob
        )
        self.attention2 = Transformer(
            d_model=self.hidden_dim, n_head=self.n_heads, drop_prob=self.drop_prob
        )
        self.linear1 = nn.Linear(self.hidden_dim, self.user_size)
        self.CE_loss_func = nn.CrossEntropyLoss(reduction="mean", ignore_index=PAD)

    def cal_CE_loss(self, pred, target):
        pred = pred.contiguous().view(-1, pred.size(-1))
        target = target.contiguous().view(-1)
        loss = self.CE_loss_func(pred, target)
        return loss

    def forward(
        self,
        cas,
        timestamp,
        cas_index,
        static_graphs,
        dynamic_graph_list,
    ):
        cas = cas.to(self.device)
        timestamp = timestamp.to(self.device)
        cas_index = cas_index.to(self.device)
        static_graphs = [g.to(self.device) for g in static_graphs]
        for key in sorted(dynamic_graph_list[0].keys()):
            dynamic_graph_list[0][key] = dynamic_graph_list[0][key].to(self.device)
        dynamic_graph_list[1] = dynamic_graph_list[1].to(self.device)

        seed_seq = cas[:, :-1]
        gold_seq = cas[:, 1:]
        seed_timestamp = timestamp[:, :-1]
        mask = seed_seq == PAD
        batch_size, max_len = seed_seq.size()
        cas_index = cas_index.unsqueeze(1).expand(-1, max_len)

        # Static Graph Learning
        static_emb_mat = self.static_graph_encoder(static_graphs)
        static_emb = F.embedding(seed_seq, static_emb_mat)

        # Dynamic Learning
        init_emb_mat = static_emb_mat
        emb_list = self.dynamic_graph_encoder(init_emb_mat, dynamic_graph_list)

        # Memory Look-Up
        zero_vec = torch.zeros_like(seed_seq)
        dynamic_user_emb = torch.zeros(batch_size, max_len, self.hidden_dim).to(
            self.device
        )
        dynamic_cas_emb = torch.zeros(batch_size, max_len, self.hidden_dim).to(
            self.device
        )
        pre_time = 0
        for i, time in enumerate(sorted(emb_list.keys())):
            sub_user_index = torch.where(
                seed_timestamp <= time, seed_seq, zero_vec
            ) - torch.where(seed_timestamp <= pre_time, seed_seq, zero_vec)
            sub_cas_index = torch.where(sub_user_index != PAD, cas_index, zero_vec)
            sub_mask = sub_user_index == PAD

            if i == 0:
                sub_user_emb = F.embedding(sub_user_index, init_emb_mat)
                sub_cas_emb = F.embedding(sub_cas_index, init_emb_mat)
            else:
                sub_user_emb = F.embedding(
                    sub_user_index, emb_list[pre_time][0].to(self.device)
                )
                sub_cas_emb = F.embedding(
                    sub_cas_index, emb_list[pre_time][1].to(self.device)
                )

            sub_user_emb[sub_mask] = 0
            sub_cas_emb[sub_mask] = 0
            dynamic_user_emb = dynamic_user_emb + sub_user_emb
            dynamic_cas_emb = dynamic_cas_emb + sub_cas_emb
            pre_time = time

        final_emb = self.fusion1(dynamic_cas_emb, dynamic_user_emb)
        final_emb1 = self.attention1(static_emb, final_emb, final_emb, mask)
        final_emb2 = self.attention2(final_emb, static_emb, static_emb, mask)
        final_emb = self.fusion2(final_emb1, final_emb2)

        pred = self.linear1(final_emb)
        mask = get_previous_user_mask(seed_seq, self.user_size)
        pred = pred + mask

        if self.training:
            cls_loss = self.cal_CE_loss(pred, gold_seq)
            loss = cls_loss + self.dynamic_graph_encoder.loss
            return loss, pred
        else:
            return pred
