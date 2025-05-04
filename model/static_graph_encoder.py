import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv
from torch_geometric.utils import dropout_edge
from utils import PAD


class GNNLayer(nn.Module):

    def __init__(self, hidden_dim, drop_prob):
        super().__init__()
        self.gnn1 = GCNConv(hidden_dim, hidden_dim * 2)
        self.gnn2 = GCNConv(hidden_dim * 2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.nonlinear = nn.ReLU()

    def forward(self, x, edge_index):
        out = self.gnn1(x, edge_index)
        out = self.nonlinear(out)
        out = self.dropout(out)
        out = self.norm1(out)

        out = self.gnn2(out, edge_index)
        out = self.nonlinear(out)
        out = self.dropout(out)
        out = self.norm2(out)
        return out


class HGNNLayer(nn.Module):
    def __init__(self, hidden_dim, drop_prob):
        super().__init__()
        self.gnn1 = HypergraphConv(hidden_dim, hidden_dim * 2)
        self.gnn2 = HypergraphConv(hidden_dim * 2, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.nonlinear = nn.ReLU()

    def forward(self, x, edge_index):
        out = self.gnn1(x, edge_index)
        out = self.nonlinear(out)
        out = self.dropout(out)
        out = self.norm1(out)

        out = self.gnn2(out, edge_index)
        out = self.nonlinear(out)
        out = self.dropout(out)
        out = self.norm2(out)
        return out


class StaticGraphEnocoder(nn.Module):

    def __init__(self, num_nodes, hidden_dim, drop_prob, graph_drop_prob):
        super().__init__()
        self.node_embedding1 = nn.Embedding(num_nodes, hidden_dim, padding_idx=PAD)

        self.dropout = nn.Dropout(drop_prob)
        self.graph_drop_prob = graph_drop_prob

        self.norm = nn.LayerNorm(normalized_shape=hidden_dim)

        self.gnn_layer1 = GNNLayer(hidden_dim, drop_prob)
        self.gnn_layer2 = GNNLayer(hidden_dim, drop_prob)
        self.gnn_layer3 = HGNNLayer(hidden_dim, drop_prob)

    def user_att_fusion(self, x1, x2, x3):
        Q = self.node_embedding1.weight.unsqueeze(-2)
        K = V = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2), x3.unsqueeze(-2)], -2)

        k_t = K.transpose(-2, -1)
        score = (Q @ k_t) / math.sqrt(K.size(-1))
        score = F.softmax(score, -1)

        out = score @ V
        out = out.squeeze(-2)
        out = self.dropout(self.norm(out))
        return out

    def forward(self, static_graphs):
        social_graph = static_graphs[0]
        diffusion_graph = static_graphs[1]
        hypergraph = static_graphs[2]
        if self.training:
            social_graph = dropout_edge(social_graph, p=self.graph_drop_prob)[0]
            diffusion_graph = dropout_edge(diffusion_graph, p=self.graph_drop_prob)[0]
            hypergraph = dropout_edge(hypergraph, p=self.graph_drop_prob)[0]

        soc_emb_mat = self.gnn_layer1(self.node_embedding1.weight, social_graph)
        dif_emb_mat = self.gnn_layer2(self.node_embedding1.weight, diffusion_graph)
        hyper_emb_mat = self.gnn_layer3(self.node_embedding1.weight, hypergraph)

        out = self.user_att_fusion(soc_emb_mat, dif_emb_mat, hyper_emb_mat)
        return out
