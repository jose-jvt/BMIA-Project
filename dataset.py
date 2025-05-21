import torch
from torch_geometric.data import Data, Dataset
from scipy.io import loadmat
import numpy as np
import os

class BrainGraphDataset(Dataset):
    def __init__(self, mat_dir, labels_dict, transform=None):
        """
        Args:
            mat_dir (str): Directory containing .mat files.
            labels_dict (dict): Dictionary mapping subject IDs to BMI values (floats).
            transform (callable, optional): Optional transform to apply to data.
        """
        super().__init__(transform)
        self.mat_dir = mat_dir
        self.file_list = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
        self.labels_dict = labels_dict
        self.transform = transform

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        file_name = self.file_list[idx]
        subject_id = os.path.splitext(file_name)[0]

        # TODO. PROPERLY IMPLEMENT ME
        # Get BMI value
        bmi_value = self.labels_dict.get(subject_id)
        if bmi_value is None:
            raise ValueError(f"No BMI value found for subject ID: {subject_id}")

        # Load connectivity matrix
        mat = loadmat(os.path.join(self.mat_dir, file_name))
        connectivity = mat['connectivity']

        # Build undirected graph from upper triangle (excluding self-loops)
        row, col = np.triu_indices_from(connectivity, k=1)
        weights = connectivity[row, col]
        non_zero = weights != 0

        row, col = row[non_zero], col[non_zero]
        edge_index = np.stack([row, col], axis=0)
        edge_index = np.concatenate([edge_index, edge_index[[1, 0]]], axis=1)  # make undirected
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        edge_attr = torch.tensor(connectivity[edge_index[0], edge_index[1]], dtype=torch.float)

        # TODO. CHECK IF THIS IS NECESSARY
        # Identity matrix as node features (can be upgraded later)
        x = torch.eye(connectivity.shape[0])

        y = torch.tensor([bmi_value], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        if self.transform:
            data = self.transform(data)

        return data
