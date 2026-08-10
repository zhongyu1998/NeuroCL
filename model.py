import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList
from torch_geometric.nn import MessagePassing, SGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class NeuroCL(MessagePassing):
    def __init__(self, hidden_dim, K, dropout, device, add_self_loops=False, layer_comb=True):
        super(NeuroCL, self).__init__(aggr='add')

        self.K = K
        self.dropout = dropout
        self.device = device
        self.add_self_loops = add_self_loops
        self.layer_comb = layer_comb

        self.init_embs = Embedding(3, hidden_dim)

        self.lins_pos = ModuleList()
        self.lins_neg = ModuleList()
        self.lins_ant = ModuleList()

        for _ in range(K):
            self.lins_pos.append(Linear(hidden_dim, hidden_dim))
            self.lins_neg.append(Linear(hidden_dim, hidden_dim))
            self.lins_ant.append(Linear(hidden_dim, hidden_dim))

        self.lin_pol = Linear(hidden_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        self.init_embs.reset_parameters()

        for lin_pos, lin_neg, lin_ant in zip(self.lins_pos, self.lins_neg, self.lins_ant):
            lin_pos.reset_parameters()
            lin_neg.reset_parameters()
            lin_ant.reset_parameters()

        self.lin_pol.reset_parameters()

    def forward(self, x, edge_index, edge_attr, ant_edge_index=None):
        h = self.init_embs((x+1))
        h_list = [h.clone()]

        edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        pos_edge_index, pos_edge_weight = edge_index[:, edge_attr == 1], edge_weight[edge_attr == 1]
        neg_edge_index, neg_edge_weight = edge_index[:, edge_attr == -1], edge_weight[edge_attr == -1]

        if pos_edge_index.size(1):
            pos_updated_nodes = torch.unique(pos_edge_index[1])
        if neg_edge_index.size(1):
            neg_updated_nodes = torch.unique(neg_edge_index[1])
        if ant_edge_index is not None:
            in_degree = torch.bincount(ant_edge_index[1]).float()
            ant_edge_weight = 1.0 / in_degree[ant_edge_index[1]]
            ant_updated_nodes = torch.unique(ant_edge_index[1])

        for k in range(self.K):
            if pos_edge_index.size(1):
                h_pro = self.propagate(pos_edge_index, x=h, edge_weight=pos_edge_weight)
                h[pos_updated_nodes] = self.lins_pos[k](h_pro[pos_updated_nodes])
            if neg_edge_index.size(1):
                h_pro = self.propagate(neg_edge_index, x=h, edge_weight=neg_edge_weight)
                h[neg_updated_nodes] = self.lins_neg[k](h_pro[neg_updated_nodes])
            if ant_edge_index is not None:
                h_pro = self.propagate(ant_edge_index, x=h, edge_weight=ant_edge_weight)
                h[ant_updated_nodes] = self.lins_ant[k](h_pro[ant_updated_nodes])
            h_list.append(F.dropout(h.clone(), self.dropout, training=self.training))

        if self.layer_comb:
            rep = torch.stack(h_list, dim=0).mean(dim=0)
        else:
            rep = h

        return rep

    def polarity_prediction(self, rep):
        return F.sigmoid(self.lin_pol(rep))

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
