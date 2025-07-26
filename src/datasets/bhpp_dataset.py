import os
import os.path as osp

import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from google_drive_downloader import GoogleDriveDownloader as gdd


DATASET_INFO = {
    'Movielens1M': {
        'file_id': '12aQuU2c0GeyOD4rP-wZ0CO9NGjq4EIIy',
        'file_name': 'movielens.zip'
    }
}


class BHPPDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self):
        return 'graph.txt.new'

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        file_path = osp.join(self.raw_dir, DATASET_INFO[self.name]['file_name'])
        gdd.download_file_from_google_drive(
            file_id=DATASET_INFO[self.name]['file_id'],
            dest_path=file_path,
            unzip=True
        )
        # remove raw zip file
        if osp.exists(file_path):
            os.remove(file_path)

    def process(self):
        data = HeteroData()

        # Read the edge list
        edges = []
        data_path = osp.join(self.raw_dir, self.raw_file_names)
        with open(data_path, 'r') as f:
            for line in f:
                u, i, _ = line.strip().split('\t')
                edges.append((int(u), int(i)))
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
