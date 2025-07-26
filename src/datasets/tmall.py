import os
import os.path as osp
import pickle
import shutil

import torch
from torch_geometric.data import HeteroData, InMemoryDataset, download_url, extract_zip


class Tmall(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/HKUDS/LightGCL/main/data/tmall.zip'

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self):
        return ['trnMat.pkl', 'tstMat.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'tmall'), self.raw_dir)
        os.unlink(path)

    def process(self):
        data = HeteroData()

        # Read dataset
        edges = [[], []]
        for file in self.raw_file_names:
            data_path = osp.join(self.raw_dir, file)
            with open(data_path, 'rb') as f:
                mat = pickle.load(f)
                row, col = mat.nonzero()
                edges[0].extend(row.tolist())
                edges[1].extend(col.tolist())

        edge_index = torch.tensor(edges, dtype=torch.long)
        data['user', 'interact', 'item']['edge_index'] = edge_index

        # Process number of nodes for each node type
        data['user'].num_nodes = edge_index[0].max().item() + 1
        data['item'].num_nodes = edge_index[1].max().item() + 1

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)
