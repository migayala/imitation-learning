import os
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from dataset import DemoDataset  # noqa: E402
from model import BCPolicy  # noqa: E402


class TestSmoke(unittest.TestCase):
    def test_dataset_and_model_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            hdf5_path = os.path.join(tmp, "sample.hdf5")
            with h5py.File(hdf5_path, "w") as f:
                data = f.create_group("data")
                demo = data.create_group("demo_0")
                obs = demo.create_group("obs")
                images = np.random.randint(0, 255, size=(4, 84, 84, 3), dtype=np.uint8)
                actions = np.random.randn(4, 7).astype(np.float32)
                obs.create_dataset("agentview_image", data=images)
                demo.create_dataset("actions", data=actions)

            dataset = DemoDataset(hdf5_path=hdf5_path, camera="agentview_image", image_size=84)
            self.assertEqual(len(dataset), 4)

            obs_tensor, action_tensor = dataset[0]
            self.assertEqual(tuple(obs_tensor.shape), (3, 84, 84))
            self.assertEqual(tuple(action_tensor.shape), (7,))

            model = BCPolicy(action_dim=7, hidden_dim=64, freeze_encoder=False)
            pred = model(obs_tensor.unsqueeze(0))
            self.assertEqual(tuple(pred.shape), (1, 7))
            self.assertTrue(torch.isfinite(pred).all().item())


if __name__ == "__main__":
    unittest.main()
