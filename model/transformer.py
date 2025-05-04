import torch
import torch.nn.functional as F
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

    def forward(self, Q, K, V, pad_mask):
        # 1. dot product with weight matrices
        Q, K, V = self.w_q(Q), self.w_k(K), self.w_v(V)
        # 2. split tensor by number of heads
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        batch_size, head, length, d_tensor = K.size()
        k_t = K.transpose(2, 3)
        score = (Q @ k_t) / math.sqrt(d_tensor)

        pad_mask = (
            pad_mask.unsqueeze(-2)
            .expand(-1, head, -1)
            .unsqueeze(-2)
            .expand(-1, -1, length, -1)
        )

        mask = torch.triu(torch.ones(score.size()), diagonal=1).bool().to(score.device)
        mask += pad_mask
        score = score.masked_fill(mask, -10000)

        score = F.softmax(score, dim=-1)
        out = score @ V

        out = self.concat(out)
        out = self.w_concat(out)

        return out


class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, hidden, drop_prob):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForwardNetwork(
            d_model=d_model, hidden=d_model, drop_prob=drop_prob
        )
        self.dropout = nn.Dropout(p=drop_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, pad_mask):
        x = self.attention(Q, K, V, pad_mask)
        x = self.dropout(x)
        x = self.norm1(x + V)
        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(x + _x)
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, drop_prob):
        super().__init__()
        self.encoder1 = EncoderLayer(d_model, n_head, drop_prob)

    def forward(self, Q, K, V, pad_mask):
        x = self.encoder1(Q, K, V, pad_mask)
        return x
