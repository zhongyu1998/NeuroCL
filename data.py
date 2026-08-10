import gzip
import os
import torch

from torch_geometric.data import Dataset


class SATDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SATDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return sorted(list(os.listdir(self.processed_dir)), key=lambda fn: os.path.getsize(f"{self.processed_dir}/{fn}"))

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.processed_file_names[idx].endswith(".gz"):
            with gzip.open(os.path.join(self.processed_dir, self.processed_file_names[idx]), "rb") as f:
                data = torch.load(f, weights_only=False)
        else:
            data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)

        reverse = data.edge_index.index_select(0, torch.LongTensor([1, 0]))
        data.edge_index = torch.cat([data.edge_index, reverse], dim=1)
        data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)

        data.x = data.x.int()
        data.edge_index = data.edge_index.long()

        if hasattr(data, "ant_edge_index"):
            data.ant_edge_index = data.ant_edge_index.long()

        return data
