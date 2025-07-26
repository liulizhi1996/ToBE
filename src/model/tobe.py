import torch
import torch.nn as nn

from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.dense import Linear

from src.model.gps_conv import GPSConv


class ToBE(nn.Module):
    def __init__(
        self,
        num_user: int,
        num_item: int,
        feat_dim: int,
        pe_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        lambda_reg: float = 1e-4
    ):
        super().__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.feat_dim = feat_dim
        self.pe_dim = pe_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lambda_reg = lambda_reg

        self.user_embedding = nn.Embedding(num_user, feat_dim)
        self.item_embedding = nn.Embedding(num_item, feat_dim)

        self.pe_norm_user, self.pe_norm_item = BatchNorm(pe_dim), BatchNorm(pe_dim)
        self.pe_lin_user, self.pe_lin_item = Linear(pe_dim, pe_dim), Linear(pe_dim, pe_dim)

        self.convs_u2i, self.convs_i2u = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            conv = GPSConv(feat_dim, pe_dim, dropout=dropout, flow='source_to_target')
            self.convs_u2i.append(conv)
            conv = GPSConv(feat_dim, pe_dim, dropout=dropout, flow='target_to_source')
            self.convs_i2u.append(conv)

        self.lin_user = nn.Sequential(
            Linear(feat_dim, feat_dim * 2),
            nn.LeakyReLU(),
            Linear(feat_dim * 2, feat_dim)
        )
        self.lin_item = nn.Sequential(
            Linear(feat_dim, feat_dim * 2),
            nn.LeakyReLU(),
            Linear(feat_dim * 2, feat_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_embedding(self, edge_index, pe_user, pe_item):
        x_user = self.user_embedding.weight
        x_item = self.item_embedding.weight

        # Pre-processing positional encodings
        pe_user = self.pe_lin_user(self.pe_norm_user(pe_user))
        pe_item = self.pe_lin_item(self.pe_norm_item(pe_item))

        # GraphGPS to learn embeddings
        for i in range(self.num_layers):
            h_user = self.convs_i2u[i](x_user, x_item, pe_user, pe_item, edge_index)
            h_item = self.convs_u2i[i](x_user, x_item, pe_user, pe_item, edge_index)
            x_user, x_item = h_user, h_item

        x_user = self.lin_user(x_user)
        x_item = self.lin_item(x_item)

        return x_user, x_item

    def forward(self, edge_index, pe_user, pe_item, edge_label_index):
        r"""Computes rankings for pairs of nodes.
        Args:
            edge_index (torch.Tensor): Edge tensor specifying
                the connectivity of the graph.
            pe_user (torch.Tensor): Positional encodings of user's nodes.
            pe_item (torch.Tensor): Positional encodings of item's nodes.
            edge_label_index (torch.Tensor): Edge tensor specifying
                the node pairs for which to compute rankings or probabilities.
        """
        emb_user, emb_item = self.get_embedding(edge_index, pe_user, pe_item)

        emb_user = emb_user[edge_label_index[0]]
        emb_item = emb_item[edge_label_index[1]]

        return (emb_user * emb_item).sum(dim=-1)
