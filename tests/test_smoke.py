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

            obs_tensor, state_tensor, action_tensor = dataset[0]
            self.assertEqual(tuple(obs_tensor.shape), (3, 84, 84))
            self.assertEqual(tuple(state_tensor.shape), (0,))  # no state keys
            self.assertEqual(tuple(action_tensor.shape), (7,))

            model = BCPolicy(action_dim=7, hidden_dim=64, freeze_encoder=False)
            pred = model(obs_tensor.unsqueeze(0))
            self.assertEqual(tuple(pred.shape), (1, 7))
            self.assertTrue(torch.isfinite(pred).all().item())

    def test_frame_stack_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            hdf5_path = os.path.join(tmp, "sample_stack.hdf5")
            with h5py.File(hdf5_path, "w") as f:
                data = f.create_group("data")
                demo = data.create_group("demo_0")
                obs = demo.create_group("obs")
                images = np.random.randint(0, 255, size=(3, 84, 84, 3), dtype=np.uint8)
                actions = np.random.randn(3, 7).astype(np.float32)
                obs.create_dataset("agentview_image", data=images)
                demo.create_dataset("actions", data=actions)

            dataset = DemoDataset(
                hdf5_path=hdf5_path,
                camera="agentview_image",
                image_size=84,
                frame_stack=4,
            )
            obs_tensor, _, _ = dataset[0]
            self.assertEqual(tuple(obs_tensor.shape), (12, 84, 84))

            model = BCPolicy(action_dim=7, hidden_dim=64, freeze_encoder=False, in_channels=12)
            pred = model(obs_tensor.unsqueeze(0))
            self.assertEqual(tuple(pred.shape), (1, 7))
            self.assertTrue(torch.isfinite(pred).all().item())


    def test_state_fusion_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            hdf5_path = os.path.join(tmp, "sample_fused.hdf5")
            with h5py.File(hdf5_path, "w") as f:
                data = f.create_group("data")
                demo = data.create_group("demo_0")
                obs = demo.create_group("obs")
                images = np.random.randint(0, 255, size=(4, 84, 84, 3), dtype=np.uint8)
                eef_pos = np.random.randn(4, 3).astype(np.float32)
                eef_quat = np.random.randn(4, 4).astype(np.float32)
                gripper_qpos = np.random.randn(4, 2).astype(np.float32)
                actions = np.random.randn(4, 7).astype(np.float32)
                obs.create_dataset("agentview_image", data=images)
                obs.create_dataset("robot0_eef_pos", data=eef_pos)
                obs.create_dataset("robot0_eef_quat", data=eef_quat)
                obs.create_dataset("robot0_gripper_qpos", data=gripper_qpos)
                demo.create_dataset("actions", data=actions)

            state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
            dataset = DemoDataset(
                hdf5_path=hdf5_path,
                camera="agentview_image",
                image_size=84,
                state_keys=state_keys,
            )
            self.assertEqual(dataset.state_dim, 9)

            obs_tensor, state_tensor, action_tensor = dataset[0]
            self.assertEqual(tuple(obs_tensor.shape), (3, 84, 84))
            self.assertEqual(tuple(state_tensor.shape), (9,))
            self.assertEqual(tuple(action_tensor.shape), (7,))

            model = BCPolicy(action_dim=7, hidden_dim=64, freeze_encoder=False, state_dim=9)
            pred = model(obs_tensor.unsqueeze(0), state_tensor.unsqueeze(0))
            self.assertEqual(tuple(pred.shape), (1, 7))
            self.assertTrue(torch.isfinite(pred).all().item())


if __name__ == "__main__":
    unittest.main()
