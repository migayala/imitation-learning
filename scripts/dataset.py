"""
PyTorch Dataset for loading HDF5 demonstration data.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class DemoDataset(Dataset):
    def __init__(self, hdf5_path: str, image_size: int = 84):
        self.path = hdf5_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Index all (episode, step) pairs
        self.index = []
        with h5py.File(hdf5_path, "r") as f:
            for ep_key in f.keys():
                n_steps = len(f[ep_key]["actions"])
                for step in range(n_steps):
                    self.index.append((ep_key, step))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_key, step = self.index[idx]
        with h5py.File(self.path, "r") as f:
            obs = f[ep_key]["observations"][step]  # (H, W, C) uint8
            action = f[ep_key]["actions"][step]     # (action_dim,)

        obs_tensor = self.transform(obs)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        return obs_tensor, action_tensor


def make_dataloaders(hdf5_path: str, image_size: int, batch_size: int, train_split: float):
    dataset = DemoDataset(hdf5_path, image_size)
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Dataset: {len(dataset)} steps | train: {n_train} | val: {n_val}")
    return train_loader, val_loader
