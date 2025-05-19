import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LazyNDVIDatasetEffNet(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.patches_path = os.path.join(data_dir, "patches.npy")
        self.labels_path = os.path.join(data_dir, "labels.npy")

        self.labels = np.load(self.labels_path)
        self.n_samples = len(self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with open(self.patches_path, 'rb') as f:
            patch_size = 128
            patch_bytes = patch_size * patch_size * 4  # float32 = 4 bytes
            f.seek(idx * patch_bytes)
            patch = np.frombuffer(f.read(patch_bytes), dtype=np.float32).reshape((patch_size, patch_size))

        # EfficientNet expects 3 channels (RGB gibi), biz NDVI tek kanalı 3 kere kopyalıyoruz
        patch = np.repeat(patch[np.newaxis, :, :], 3, axis=0)  # [3, 128, 128]
        patch = torch.tensor(patch, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return patch, label
