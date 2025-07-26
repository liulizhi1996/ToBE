import pickle as pkl
import os.path as osp
from argparse import ArgumentParser

import torch
from torch_geometric import seed_everything
from torch_geometric.datasets import AmazonBook
from torch_geometric.utils import degree

from src.datasets import GEBEDataset, Gowalla, AlibabaIFashion, BHPPDataset, Tmall
from src.transform.posenc import positional_encoding

ROOT_PATH = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data')


def load_dataset(name):
    path = osp.join(ROOT_PATH, name)
    if name == 'Yelp':
        dataset = GEBEDataset(root=path, name=name)
        data = dataset[0]
        num_user, num_item = data['user']['num_nodes'], data['item']['num_nodes']
        edge_index = data['user', 'interact', 'item']['edge_index']
    elif name == 'AmazonBook':
        dataset = AmazonBook(root=path)
        data = dataset[0]
        num_user, num_item = data['user']['num_nodes'], data['book']['num_nodes']
        edge_index = torch.cat([data['user', 'rates', 'book']['edge_index'],
                                data['user', 'rates', 'book']['edge_label_index']], dim=1)
    elif name == 'Gowalla':
        dataset = Gowalla(root=path)
        data = dataset[0]
        num_user, num_item = data['user']['num_nodes'], data['item']['num_nodes']
        edge_index = torch.cat([data['user', 'rates', 'item']['edge_index'],
                                data['user', 'rates', 'item']['edge_label_index']], dim=1)
    elif name == 'AlibabaIFashion':
        dataset = AlibabaIFashion(root=path)
        data = dataset[0]
        num_user, num_item = data['user']['num_nodes'], data['item']['num_nodes']
        edge_index = data['user', 'interact', 'item']['edge_index']
    elif name == 'Movielens1M':
        dataset = BHPPDataset(root=path, name=name)
        data = dataset[0]
        num_user, num_item = data['user']['num_nodes'], data['item']['num_nodes']
        edge_index = data['user', 'interact', 'item']['edge_index']
    elif name == 'Tmall':
        dataset = Tmall(root=path)
        data = dataset[0]
        num_user, num_item = data['user']['num_nodes'], data['item']['num_nodes']
        edge_index = data['user', 'interact', 'item']['edge_index']
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return num_user, num_item, edge_index


def k_core_filtering(edges, num_user, num_item, k=10):
    # Ensure that all users and items have at least k interactions
    degree_u = degree(edges[0], num_user)
    degree_v = degree(edges[1], num_item)
    mask_u = degree_u >= k
    mask_v = degree_v >= k
    mask = mask_u[edges[0]] & mask_v[edges[1]]
    edges = edges[:, mask]

    # Remap user and item IDs
    unique_user_ids = torch.unique(edges[0])
    unique_item_ids = torch.unique(edges[1])
    new_user_id_map = torch.full((num_user,), -1)
    new_item_id_map = torch.full((num_item,), -1)
    new_user_id_map[unique_user_ids] = torch.arange(len(unique_user_ids))
    new_item_id_map[unique_item_ids] = torch.arange(len(unique_item_ids))
    remapped_user_ids = new_user_id_map[edges[0]]
    remapped_item_ids = new_item_id_map[edges[1]]
    edges = torch.stack([remapped_user_ids, remapped_item_ids], dim=0)
    num_user = len(unique_user_ids)
    num_item = len(unique_item_ids)

    return num_user, num_item, edges


def _setdiff1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ensure x and y are 1-dimensional
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError("Both x and y must be 1-dimensional tensors.")
    # Find unique elements in x
    unique_x = torch.unique(x)
    # Create a mask of elements in unique_x that are not in y
    mask = torch.isin(unique_x, y, invert=True)
    # Return the elements in unique_x that are not in y
    result = unique_x[mask]
    return result


def split_data(edges, split_ratio=0.8):
    # Split train & test edges
    num_edges = edges.size(1)
    perm = torch.randperm(num_edges)
    train_size = int(num_edges * split_ratio)
    train_edges = edges[:, perm[:train_size]]
    test_edges = edges[:, perm[train_size:]]

    # Filter out cold start users & items from test edges
    cold_start_users = _setdiff1d(test_edges[0], train_edges[0])
    if cold_start_users.size(0) > 0:
        mask = ~torch.isin(test_edges[0], cold_start_users)
        test_edges = test_edges[:, mask]
    cold_start_items = _setdiff1d(test_edges[1], train_edges[1])
    if cold_start_items.size(0) > 0:
        mask = ~torch.isin(test_edges[1], cold_start_items)
        test_edges = test_edges[:, mask]

    return train_edges, test_edges


def save_data(dataset_name, num_user, num_item, train_edges, test_edges, pos_enc, split_ratio=0.8):
    data = {
        'num_user': num_user,
        'num_item': num_item,
        'train_edges': train_edges,
        'test_edges': test_edges,
        'positional_encoding': {
            'user': pos_enc[0],
            'item': pos_enc[1]
        }
    }
    file_name = osp.join(ROOT_PATH, dataset_name, 'processed', f'processed_{split_ratio}.pkl')
    with open(file_name, 'wb') as f:
        pkl.dump(data, f)


def main():
    parser = ArgumentParser("Preprocess dataset for bipartite graph embedding learning")
    # Dataset settings
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio of train edges')
    parser.add_argument('--seed', type=int, default=23, help='Random seed')
    parser.add_argument('--do_filtering', action='store_true', help='Whether to adopt the k-core setting')
    parser.add_argument('--k_core', type=int, default=10, help='Minimum interactions of nodes')
    # Positional encoding settings
    parser.add_argument('--svd_rank', type=int, default=100, help='Rank of randomized SVD')
    parser.add_argument('--walk_length', type=int, default=16, help='Number of random walk steps')
    parser.add_argument('--lap_pe_k', type=int, default=16, help='Number of eigenvectors')
    args = parser.parse_args()

    # Set seed for reproducibility
    seed_everything(args.seed)
    # Load dataset
    num_user, num_item, edges = load_dataset(args.dataset)
    # Perform k-core filtering
    if args.do_filtering:
        num_user, num_item, edges = k_core_filtering(edges, num_user, num_item, args.k_core)
    # Split train/test data
    train_edges, test_edges = split_data(edges, args.split_ratio)
    # Precompute positional encoding for each node
    z_u, z_v = positional_encoding(train_edges, (num_user, num_item), args)
    # Save data
    save_data(args.dataset, num_user, num_item, train_edges, test_edges, (z_u, z_v), args.split_ratio)


if __name__ == '__main__':
    main()
