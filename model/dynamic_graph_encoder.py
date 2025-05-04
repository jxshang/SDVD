import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import dropout_edge


class VariationalHypergraphAutoencoder(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.base_hgnn = HypergraphConv(hidden_dim, hidden_dim)
        self.mean_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.std_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinear = nn.ReLU()

    def encode(self, x, edge_index):
        x = self.base_hgnn(x, edge_index)
        x = self.nonlinear(x)
        self.mean = self.mean_mlp(x)
        self.logstd = self.std_mlp(x)

        # sample
        noise = torch.randn_like(x)
        sampled_z = noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def get_loss(self):
        loss = (
            0.5
            * (-1 + torch.exp(self.logstd) ** 2 + self.mean**2 - 2 * self.logstd)
            .sum(1)
            .mean()
        )
        return loss

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        if self.training:
            self.loss = self.get_loss()
        return z


class NormLayer(nn.Module):

    def __init__(self, hidden_dim, drop_prob):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, _x):
        x = self.dropout(x)
        return self.norm(x + _x)


class DynamicVHGAE(nn.Module):

    def __init__(self, hidden_dim, n_interval, drop_prob, graph_drop_prob):
        super().__init__()
        self.user_vae = VariationalHypergraphAutoencoder(hidden_dim)
        self.cas_vae = VariationalHypergraphAutoencoder(hidden_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.graph_drop_prob = graph_drop_prob
        self.n_interval = n_interval
        self.user_norm_layers = nn.ModuleList(
            [NormLayer(hidden_dim, drop_prob) for _ in range(self.n_interval)]
        )
        self.cas_norm_layers = nn.ModuleList(
            [NormLayer(hidden_dim, drop_prob) for _ in range(self.n_interval)]
        )
        # self.linear1 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, dynamic_graph_list):
        graph_list = dynamic_graph_list[0]
        root_users = dynamic_graph_list[1]
        embedding_list = {}
        init_user_emb = x
        init_cas_emb = F.embedding(root_users, x)
        self.loss = 0
        keys = sorted(graph_list.keys())
        for i in range(self.n_interval):
            user_graph = graph_list[keys[i]]
            cas_graph = torch.flip(user_graph, [0])
            if self.training:
                user_graph = dropout_edge(user_graph, p=self.graph_drop_prob)[0]
                cas_graph = dropout_edge(cas_graph, p=self.graph_drop_prob)[0]

            user_emb = self.user_vae(init_user_emb, user_graph)
            cas_emb = self.cas_vae(init_cas_emb, cas_graph)

            init_user_emb = self.user_norm_layers[i](user_emb, init_user_emb)
            init_cas_emb = self.cas_norm_layers[i](cas_emb, init_cas_emb)
            embedding_list[keys[i]] = [init_user_emb.cpu(), init_cas_emb.cpu()]

            if self.training:
                target_graph = user_graph
                adj = (
                    torch.sparse_coo_tensor(
                        target_graph,
                        target_graph.new_ones(target_graph.size(1)),
                        [init_user_emb.size(0), init_cas_emb.size(0)],
                    )
                    .to_dense()
                    .float()
                )
                adj_rec = torch.sigmoid(user_emb @ cas_emb.t())
                rec_loss = F.binary_cross_entropy(adj_rec.view(-1), adj.view(-1))
                self.loss = (
                    self.loss + self.user_vae.loss + self.cas_vae.loss + rec_loss
                )

        self.loss = self.loss / self.n_interval

        return embedding_list
