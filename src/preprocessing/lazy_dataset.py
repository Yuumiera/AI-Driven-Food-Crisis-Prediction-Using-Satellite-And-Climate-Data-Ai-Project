import torch
from torch.utils.data import Dataset
import numpy as np
import os

class LazyNDVIDataset(Dataset):
    def __init__(self, data_dir):
        self.patches = np.load(os.path.join(data_dir, "patches.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode='r')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx].astype(np.float32)
        img = np.expand_dims(img, axis=0)  # Tek kanal
        label = self.labels[idx]
        return torch.tensor(img), torch.tensor(label, dtype=torch.float32)
