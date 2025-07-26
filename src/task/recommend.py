import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.utils import degree, cumsum

from src.utils.data_utils import load_data
from src.model.tobe import ToBE

# Root path to data
ROOT_PATH = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data')


def parse_args():
    parser = argparse.ArgumentParser(description='ToBE on recommendation task')
    # Dataset settings
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Ratio of train edges')
    parser.add_argument('--seed', type=int, default=23, help='Random seed')
    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--train_batch_size', type=int, default=2048, help='Batch size during training')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='Coefficient of regularization')
    parser.add_argument('--lambda_au', type=float, default=1.0, help='Coefficient of alignment and uniformity loss')
    # Test settings
    parser.add_argument('--test_batch_size', type=int, default=2048, help='Batch size during test')
    parser.add_argument('--topk', default='10,20', type=str, help='top-K recommendation setting')
    # Model settings
    parser.add_argument('--feat_dim', type=int, default=64, help='Node embedding dimension')
    parser.add_argument('--pe_dim', type=int, default=32, help='Position encoding dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    args = parser.parse_args()

    args.topk = list(map(int, args.topk.split(',')))

    return args


def train(model, optimizer, data, batch_size, lambda_reg, lambda_au):
    model.train()
    train_edges = data.train_edge_index
    pe_user, pe_item = data.pe_user, data.pe_item

    total_loss = total_examples = 0
    loader = DataLoader(range(train_edges.size(1)), shuffle=True, batch_size=batch_size)
    for index in loader:
        pos_user, pos_item = train_edges[:, index]
        emb_user, emb_item = model.get_embedding(train_edges, pe_user, pe_item)
        y_pred = torch.einsum('ik,jk->ij', emb_user[pos_user], emb_item)
        y_true = pos_item
        loss = F.cross_entropy(y_pred, y_true)

        if lambda_reg != 0:
            emb = torch.cat([emb_user[pos_user], emb_item[pos_item]], dim=0)
            regularization = lambda_reg * emb.norm(p=2).pow(2)
            regularization = regularization / pos_user.numel()
            loss += regularization

        emb_user = F.normalize(emb_user[pos_user], dim=-1)
        emb_item = F.normalize(emb_item[pos_item], dim=-1)
        align = (emb_user - emb_item).norm(p=2, dim=1).pow(2).mean()
        uniform = (torch.pdist(emb_user, p=2).pow(2).mul(-2).exp().mean().log() +
                   torch.pdist(emb_item, p=2).pow(2).mul(-2).exp().mean().log()) / 2
        loss += lambda_au * (align + uniform)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_user.numel()
        total_examples += pos_user.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(model, data, batch_size, topk):
    model.eval()

    train_edges, test_edges = data.train_edge_index, data.test_edge_index
    pe_user, pe_item = data.pe_user, data.pe_item
    emb_user, emb_item = model.get_embedding(train_edges, pe_user, pe_item)

    total_examples = 0
    metrics = {}
    for k in topk:
        metrics[f'precision@{k}'] = 0
        metrics[f'recall@{k}'] = 0
        metrics[f'ndcg@{k}'] = 0

    for start in range(0, data.num_user, batch_size):
        end = start + batch_size
        logits = torch.einsum('ik,jk->ij', emb_user[start:end], emb_item)

        # Exclude training edges
        mask = ((train_edges[0] >= start) & (train_edges[0] < end))
        logits[train_edges[0, mask] - start, train_edges[1, mask]] = float('-inf')

        # Computing precision, recall and NDCG:
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((test_edges[0] >= start) & (test_edges[0] < end))
        ground_truth[test_edges[0, mask] - start, test_edges[1, mask]] = True
        node_count = degree(test_edges[0, mask] - start, num_nodes=logits.size(0), dtype=torch.int)
        total_examples += int((node_count > 0).sum())
        for k in topk:
            topk_index = logits.topk(k, dim=-1).indices
            isin_mat = ground_truth.gather(1, topk_index)
            metrics[f'precision@{k}'] += float((isin_mat.sum(dim=-1) / k).sum())
            metrics[f'recall@{k}'] += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())

            multiplier = 1.0 / torch.arange(2, k + 2, device=logits.device).log2()
            idcg = cumsum(multiplier)
            dcg = (isin_mat * multiplier.view(1, -1)).sum(dim=-1)
            idcg = idcg[node_count.clamp(max=k)]
            ndcg = dcg / idcg
            ndcg[ndcg.isnan() | ndcg.isinf()] = 0.0
            metrics[f'ndcg@{k}'] += float(ndcg.sum())

    for k in topk:
        metrics[f'precision@{k}'] /= total_examples
        metrics[f'recall@{k}'] /= total_examples
        metrics[f'ndcg@{k}'] /= total_examples

    return metrics


def main():
    args = parse_args()

    # Set seed for reproducibility
    seed_everything(args.seed)
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    path_to_data = osp.join(ROOT_PATH, args.dataset, 'processed', f'processed_{args.split_ratio}.pkl')
    num_user, num_item, train_edges, test_edges, pe_user, pe_item = load_data(path_to_data)
    data = Data(train_edge_index=train_edges, test_edge_index=test_edges,
                num_user=num_user, num_item=num_item, pe_user=pe_user, pe_item=pe_item).to(device)

    # Build model
    model = ToBE(num_user, num_item, args.feat_dim, args.pe_dim,
                 args.num_layers, args.dropout, args.lambda_reg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train & test loop
    for epoch in range(args.epochs):
        loss = train(model, optimizer, data, args.train_batch_size, args.lambda_reg, args.lambda_au)
        metrics = test(model, data, args.test_batch_size, args.topk)

        to_print = f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
        to_print += ', '.join([f'{key}: {value:.4f}' for key, value in metrics.items()])
        print(to_print)


if __name__ == '__main__':
    main()
