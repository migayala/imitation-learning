"""
PyTorch Dataset for loading robomimic HDF5 demonstration data.
"""

import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class DemoDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        camera: str = "agentview_image",
        image_size: int = 84,
        schema: str = "robomimic_image",
    ):
        if schema != "robomimic_image":
            raise ValueError(f"Unsupported dataset schema '{schema}'. Expected: robomimic_image")

        self.path = hdf5_path
        self.camera = camera
        self.schema = schema
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.index = []
        with h5py.File(hdf5_path, "r") as f:
            self._validate_schema(f)
            episode_keys = sorted(f["data"].keys())
            obs_group = f["data"][episode_keys[0]]["obs"]
            self.camera_key = self._resolve_camera_key(obs_group)

            for ep_key in episode_keys:
                n_steps = len(f["data"][ep_key]["actions"])
                for step in range(n_steps):
                    self.index.append((ep_key, step))

        if not self.index:
            raise ValueError(f"No transitions found in dataset: {hdf5_path}")

    def _validate_schema(self, f: h5py.File) -> None:
        if "data" not in f:
            raise KeyError("Invalid HDF5 format: missing top-level group 'data'")
        episode_keys = sorted(f["data"].keys())
        if not episode_keys:
            raise KeyError("Invalid HDF5 format: 'data' contains no episodes")

        first_episode = f["data"][episode_keys[0]]
        missing = [key for key in ["obs", "actions"] if key not in first_episode]
        if missing:
            raise KeyError(
                f"Invalid HDF5 format under data/{episode_keys[0]}: missing keys {missing}"
            )

    def _resolve_camera_key(self, obs_group: h5py.Group) -> str:
        candidates = [self.camera]
        if self.camera.endswith("_image"):
            candidates.append(self.camera[: -len("_image")])
        else:
            candidates.append(f"{self.camera}_image")

        for key in candidates:
            if key in obs_group:
                return key

        available = sorted(obs_group.keys())
        raise KeyError(
            f"Camera key '{self.camera}' not found in dataset. "
            f"Checked {candidates}; available keys: {available}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_key, step = self.index[idx]
        with h5py.File(self.path, "r") as f:
            obs = f["data"][ep_key]["obs"][self.camera_key][step]  # (H, W, C) uint8
            action = f["data"][ep_key]["actions"][step]        # (7,)

        obs_tensor = self.transform(obs)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        return obs_tensor, action_tensor


def make_dataloaders(
    hdf5_path: str,
    camera: str,
    image_size: int,
    batch_size: int,
    train_split: float,
    schema: str = "robomimic_image",
    num_workers: int = 4,
):
    dataset = DemoDataset(hdf5_path, camera, image_size, schema=schema)
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 transitions to create train/val splits")

    n_train = max(1, int(len(dataset) * train_split))
    n_val = len(dataset) - n_train
    if n_val == 0:
        n_train = len(dataset) - 1
        n_val = 1

    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"Dataset: {len(dataset)} steps | train: {n_train} | val: {n_val}")
    return train_loader, val_loader
