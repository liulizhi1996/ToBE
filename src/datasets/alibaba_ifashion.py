import os.path as osp

import torch
from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class AlibabaIFashion(InMemoryDataset):
    url = ('https://raw.githubusercontent.com/QwQ2000/TheWebConf24-LTGNN-PyTorch/'
           'main/data/alibaba-ifashion')

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self):
        return ['train.txt', 'test.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        data = HeteroData()

        # Read dataset
        edges = []
        for raw_file in self.raw_file_names:
            data_path = osp.join(self.raw_dir, raw_file)
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    for dst in line[1:]:
                        edges.append((int(line[0]), int(dst)))

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        data['user', 'interact', 'item']['edge_index'] = edge_index

        # Process number of nodes for each node type
        data['user'].num_nodes = edge_index[0].max().item() + 1
        data['item'].num_nodes = edge_index[1].max().item() + 1

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)
