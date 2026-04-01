import os
import sys
import tempfile
import unittest
from io import StringIO

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from inspect_hdf5 import inspect_file  # noqa: E402


class TestInspectHDF5(unittest.TestCase):
    def test_inspect_file_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            hdf5_path = os.path.join(tmp, "ok.hdf5")
            with h5py.File(hdf5_path, "w") as f:
                data = f.create_group("data")
                demo = data.create_group("demo_0")
                obs = demo.create_group("obs")
                obs.create_dataset("agentview_image", data=np.zeros((3, 84, 84, 3), dtype=np.uint8))
                demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))

            rc = inspect_file(hdf5_path, max_demos=1, camera="agentview_image")
            self.assertEqual(rc, 0)

    def test_inspect_file_detects_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            hdf5_path = os.path.join(tmp, "bad.hdf5")
            with h5py.File(hdf5_path, "w") as f:
                data = f.create_group("data")
                demo = data.create_group("demo_0")
                obs = demo.create_group("obs")
                obs.create_dataset("agentview_image", data=np.zeros((4, 84, 84, 3), dtype=np.uint8))
                demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))

            old_stdout = sys.stdout
            try:
                sys.stdout = StringIO()
                rc = inspect_file(hdf5_path, max_demos=1, camera="agentview_image")
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            self.assertEqual(rc, 1)
            self.assertIn("length mismatch", output)


if __name__ == "__main__":
    unittest.main()
