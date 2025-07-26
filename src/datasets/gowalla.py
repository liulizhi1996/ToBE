import torch
from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class Gowalla(InMemoryDataset):
    url = ('https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/'
           'master/data/gowalla')

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self):
        return ['user_list.txt', 'item_list.txt', 'train.txt', 'test.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        import pandas as pd

        data = HeteroData()

        # Process number of nodes for each node type:
        node_types = ['user', 'item']
        for path, node_type in zip(self.raw_paths, node_types):
            df = pd.read_csv(path, sep=' ', header=0)
            data[node_type].num_nodes = len(df)

        # Process edge information for training and testing:
        attr_names = ['edge_index', 'edge_label_index']
        for path, attr_name in zip(self.raw_paths[2:], attr_names):
            rows, cols = [], []
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                for dst in line[1:]:
                    rows.append(int(line[0]))
                    cols.append(int(dst))
            index = torch.tensor([rows, cols])

            data['user', 'rates', 'item'][attr_name] = index
            if attr_name == 'edge_index':
                data['item', 'rated_by', 'user'][attr_name] = index.flip([0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)
