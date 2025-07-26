from typing import Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import degree


def random_walk_pe(edge_index: Tensor, num_nodes: Tuple[int, int], walk_length: int, svd_rank: int):
    # Compute the node degrees
    row, col = edge_index[0], edge_index[1]
    deg_u = degree(row, num_nodes[0])

    # Normalize incidence matrix
    weights = torch.ones(edge_index.size(1))
    deg_u_inv_sqrt = deg_u.pow_(-1)
    deg_u_inv_sqrt.masked_fill_(deg_u_inv_sqrt == float('inf'), 0)
    weights = deg_u_inv_sqrt[row] * weights

    # Convert to sparse matrix
    adj = sp.coo_matrix((weights.numpy(), (row.numpy(), col.numpy())), num_nodes)

    # SVD decomposition
    U, s, Vh = randomized_svd(adj, n_components=svd_rank, random_state=0)

    # Compute random walk PEs
    U, s_square, Vh = U ** 2, s ** 2, Vh.T ** 2
    pe_u_list, pe_v_list = [U @ s_square], [Vh @ s_square]
    for _ in range(walk_length - 1):
        s_square *= s ** 2
        pe_u_list.append(U @ s_square)
        pe_v_list.append(Vh @ s_square)
    pe_u = np.stack(pe_u_list, axis=-1)
    pe_v = np.stack(pe_v_list, axis=-1)
    return torch.from_numpy(pe_u), torch.from_numpy(pe_v)


def laplacian_pe(edge_index: Tensor, num_nodes: Tuple[int, int], k: int, svd_rank: int):
    # Compute the node degrees
    row, col = edge_index[0], edge_index[1]
    deg_u = degree(row, num_nodes[0])
    deg_v = degree(col, num_nodes[1])

    # Normalize incidence matrix
    weights = torch.ones(edge_index.size(1))
    deg_u_inv_sqrt = deg_u.pow_(-0.5)
    deg_u_inv_sqrt.masked_fill_(deg_u_inv_sqrt == float('inf'), 0)
    deg_v_inv_sqrt = deg_v.pow_(-0.5)
    deg_v_inv_sqrt.masked_fill_(deg_v_inv_sqrt == float('inf'), 0)
    weights = deg_u_inv_sqrt[row] * weights * deg_v_inv_sqrt[col]

    # Convert to sparse matrix
    adj = sp.coo_matrix((weights.numpy(), (row.numpy(), col.numpy())), num_nodes)

    # SVD decomposition
    U, _, Vh = randomized_svd(adj, n_components=svd_rank, random_state=0)

    # Because the lowest eigenvalue is always zero, we remove it.
    pe_u = torch.from_numpy(U[:, 1:k + 1])
    pe_v = torch.from_numpy(Vh[1:k + 1, :].T)
    pe = torch.cat([pe_u, pe_v], dim=0)
    pe = F.normalize(pe, dim=0)

    # randomly flip signs
    sign = -1 + 2 * torch.randint(0, 2, (k, ))
    pe *= sign

    num_u, num_v = U.shape[0], Vh.shape[1]
    pe_u, pe_v = torch.split(pe, [num_u, num_v])
    return pe_u, pe_v


def positional_encoding(edge_index: Tensor, num_nodes: Tuple[int, int], args):
    # Random walk positional encoding
    rwpe_u, rwpe_v = random_walk_pe(edge_index, num_nodes, args.walk_length, args.svd_rank)

    # Laplacian eigenvector positional encoding
    lappe_u, lappe_v = laplacian_pe(edge_index, num_nodes, args.lap_pe_k, args.svd_rank)

    # Combine PEs
    pe_u = torch.cat([rwpe_u, lappe_u], dim=1)
    pe_v = torch.cat([rwpe_v, lappe_v], dim=1)
    return pe_u, pe_v
