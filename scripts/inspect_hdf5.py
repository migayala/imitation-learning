"""
Inspect and validate robomimic-style HDF5 datasets.
"""

import argparse
from typing import List

import h5py


def _format_shape(dataset: h5py.Dataset) -> str:
    return f"shape={tuple(dataset.shape)} dtype={dataset.dtype}"


def inspect_file(path: str, max_demos: int, camera: str) -> int:
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        print(f"Top-level keys: {sorted(f.keys())}")

        if "data" not in f:
            print("ERROR: Missing top-level 'data' group.")
            return 1

        demos = sorted(f["data"].keys())
        print(f"Num demos: {len(demos)}")
        if not demos:
            print("ERROR: No demos found under 'data'.")
            return 1

        error_count = 0
        inspect_keys: List[str] = demos[:max_demos]
        for demo_key in inspect_keys:
            demo = f["data"][demo_key]
            missing = [k for k in ["obs", "actions"] if k not in demo]
            if missing:
                print(f"ERROR: data/{demo_key} missing keys: {missing}")
                error_count += 1
                continue

            obs_group = demo["obs"]
            obs_keys = sorted(obs_group.keys())
            camera_candidates = [camera]
            if camera.endswith("_image"):
                camera_candidates.append(camera[: -len("_image")])
            else:
                camera_candidates.append(f"{camera}_image")
            resolved_camera = next((k for k in camera_candidates if k in obs_group), None)
            if resolved_camera is None:
                print(
                    f"ERROR: data/{demo_key}/obs has no camera in {camera_candidates}. "
                    f"Available: {obs_keys}"
                )
                error_count += 1
                continue

            actions = demo["actions"]
            images = obs_group[resolved_camera]
            print(f"[{demo_key}] camera={resolved_camera} {_format_shape(images)} | {_format_shape(actions)}")
            if len(images) != len(actions):
                print(
                    f"ERROR: data/{demo_key} length mismatch: "
                    f"images={len(images)} actions={len(actions)}"
                )
                error_count += 1

        print(f"Inspection complete. Checked {len(inspect_keys)} demos. Errors: {error_count}")
        return 1 if error_count else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to HDF5 file")
    parser.add_argument("--max-demos", type=int, default=5, help="Number of demos to inspect")
    parser.add_argument("--camera", default="agentview_image", help="Preferred camera key")
    args = parser.parse_args()
    return inspect_file(args.path, args.max_demos, args.camera)


if __name__ == "__main__":
    raise SystemExit(main())
