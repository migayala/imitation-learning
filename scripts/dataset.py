"""
PyTorch Dataset for loading robomimic HDF5 demonstration data.
"""

import h5py
import numpy as np
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
        frame_stack: int = 1,
    ):
        if schema != "robomimic_image":
            raise ValueError(f"Unsupported dataset schema '{schema}'. Expected: robomimic_image")

        self.path = hdf5_path
        self.camera = camera
        self.schema = schema
        self.frame_stack = frame_stack
        if self.frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {self.frame_stack}")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.action_mean = None
        self.action_std = None

        self.index = []
        with h5py.File(hdf5_path, "r") as f:
            self._validate_schema(f)
            episode_keys = sorted(f["data"].keys())
            obs_group = f["data"][episode_keys[0]]["obs"]
            self.camera_key = self._resolve_camera_key(obs_group)
            self.obs_shape = tuple(obs_group[self.camera_key].shape[1:])

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
            obs_ds = f["data"][ep_key]["obs"][self.camera_key]
            zero_obs = np.zeros(self.obs_shape, dtype=obs_ds.dtype)
            stacked_obs = []
            for offset in range(self.frame_stack - 1, -1, -1):
                frame_step = step - offset
                frame = zero_obs if frame_step < 0 else obs_ds[frame_step]
                stacked_obs.append(self.transform(frame))
            action = f["data"][ep_key]["actions"][step]        # (7,)

        obs_tensor = torch.cat(stacked_obs, dim=0)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        if self.action_mean is not None and self.action_std is not None:
            action_tensor = (action_tensor - self.action_mean) / self.action_std
        return obs_tensor, action_tensor

    def get_raw_action(self, idx: int) -> np.ndarray:
        ep_key, step = self.index[idx]
        with h5py.File(self.path, "r") as f:
            return np.asarray(f["data"][ep_key]["actions"][step], dtype=np.float64)

    def set_action_normalization(self, action_mean: torch.Tensor, action_std: torch.Tensor) -> None:
        self.action_mean = action_mean.to(dtype=torch.float32, device="cpu")
        self.action_std = action_std.to(dtype=torch.float32, device="cpu")


def compute_action_stats(dataset: DemoDataset, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    if not indices:
        raise ValueError("Cannot compute action stats from empty index list")

    first = dataset.get_raw_action(indices[0])
    action_dim = first.shape[0]
    sum_actions = np.zeros(action_dim, dtype=np.float64)
    sum_sq_actions = np.zeros(action_dim, dtype=np.float64)

    for idx in indices:
        action = dataset.get_raw_action(idx)
        sum_actions += action
        sum_sq_actions += action * action

    count = float(len(indices))
    mean = sum_actions / count
    var = np.maximum(sum_sq_actions / count - mean * mean, 1e-12)
    std = np.sqrt(var)

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def make_dataloaders(
    hdf5_path: str,
    camera: str,
    image_size: int,
    batch_size: int,
    train_split: float,
    schema: str = "robomimic_image",
    num_workers: int = 4,
    frame_stack: int = 1,
):
    dataset = DemoDataset(hdf5_path, camera, image_size, schema=schema, frame_stack=frame_stack)
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 transitions to create train/val splits")

    n_train = max(1, int(len(dataset) * train_split))
    n_val = len(dataset) - n_train
    if n_val == 0:
        n_train = len(dataset) - 1
        n_val = 1

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    action_mean, action_std = compute_action_stats(dataset, train_set.indices)
    dataset.set_action_normalization(action_mean, action_std)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"Dataset: {len(dataset)} steps | train: {n_train} | val: {n_val}")
    return train_loader, val_loader, action_mean, action_std
