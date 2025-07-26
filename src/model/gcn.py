from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class GCN(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

    def forward(self, x_src, x_dst, edge_index):
        # normalize bipartite graph
        deg_src = degree(edge_index[0], x_src.size(0), dtype=x_src.dtype).pow_(-0.5)
        deg_dst = degree(edge_index[1], x_dst.size(0), dtype=x_dst.dtype).pow_(-0.5)
        deg_src.masked_fill_(deg_src == float('inf'), 0)
        deg_dst.masked_fill_(deg_dst == float('inf'), 0)
        edge_weight = deg_src[edge_index[0]] * deg_dst[edge_index[1]]

        out = self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weight)
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
