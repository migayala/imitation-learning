"""
Evaluate a trained Diffusion Policy in the robosuite environment.
"""

import argparse
from collections import deque
import json
import os
import random
import torch
import numpy as np
import robosuite as suite
import yaml
from torchvision import transforms
from tqdm import tqdm

from diffusion_policy import DiffusionPolicy

# Dataset obs key → live env obs key (where they differ)
_DATASET_TO_ENV_KEY = {
    "object": "object-state",
}


def load_action_stats(checkpoint: str):
    path = os.path.join(os.path.dirname(checkpoint), "action_stats.json")
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return (
        np.asarray(stats["action_mean"], dtype=np.float32),
        np.asarray(stats["action_std"], dtype=np.float32),
    )


def load_state_stats(checkpoint: str):
    path = os.path.join(os.path.dirname(checkpoint), "state_stats.json")
    if not os.path.exists(path):
        return None, None, []
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return (
        np.asarray(stats["state_mean"], dtype=np.float32),
        np.asarray(stats["state_std"], dtype=np.float32),
        stats.get("state_keys", []),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _gripper_cube_distance(obs: dict) -> float:
    if "gripper_to_cube_pos" in obs:
        return float(np.linalg.norm(obs["gripper_to_cube_pos"]))
    if "robot0_eef_pos" in obs and "cube_pos" in obs:
        return float(np.linalg.norm(obs["robot0_eef_pos"] - obs["cube_pos"]))
    return float("nan")


def evaluate(cfg: dict, checkpoint: str, num_episodes: int = 50) -> float:
    seed = cfg.get("evaluation", {}).get("seed", cfg["training"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    camera_name = cfg["env"]["camera_names"]
    if isinstance(camera_name, (list, tuple)):
        camera_name = camera_name[0]
    image_key = f"{camera_name}_image"
    frame_stack = int(cfg["data"].get("frame_stack", 1))
    partial_lift_threshold = float(cfg.get("evaluation", {}).get("partial_lift_threshold", 0.03))

    state_mean, state_std, state_keys = load_state_stats(checkpoint)
    if state_keys:
        print(f"Loaded state stats for keys: {state_keys}")

    model = DiffusionPolicy(
        action_dim=cfg["model"]["action_dim"],
        hidden_dim=cfg["model"].get("hidden_dim", 256),
        cond_dim=cfg["model"].get("cond_dim", 256),
        n_blocks=cfg["model"].get("n_blocks", 4),
        T=cfg["model"].get("T", 100),
        ddim_steps=cfg["model"].get("ddim_steps", 10),
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        in_channels=3 * frame_stack,
        state_dim=cfg["model"].get("state_dim", 0),
        pred_horizon=cfg["model"].get("pred_horizon", 1),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    action_mean, action_std = load_action_stats(checkpoint)
    if action_mean is None:
        print("Warning: action_stats.json not found. Outputs will not be unnormalized.")
    else:
        print(f"Loaded action stats from {os.path.dirname(checkpoint)}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    env_cfg = cfg["env"]
    env = suite.make(
        env_name=env_cfg["name"],
        robots=env_cfg["robot"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=env_cfg["camera_names"],
        camera_heights=env_cfg["camera_heights"],
        camera_widths=env_cfg["camera_widths"],
        reward_shaping=False,
        control_freq=env_cfg["control_freq"],
    )
    action_low, action_high = env.action_spec
    exec_horizon = cfg["model"].get("exec_horizon", 1)

    successes = 0
    episode_max_lift = []
    episode_min_gripper_dist = []

    def _get_state_tensor(obs):
        if not state_keys:
            return None
        parts = [
            obs[_DATASET_TO_ENV_KEY.get(k, k)].astype(np.float32).ravel()
            for k in state_keys
        ]
        state_arr = np.concatenate(parts)
        if state_mean is not None:
            state_arr = (state_arr - state_mean) / state_std
        return torch.tensor(state_arr, dtype=torch.float32).unsqueeze(0).to(device)

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        ep_success = False
        initial_cube_z = float(obs["cube_pos"][2]) if "cube_pos" in obs else float("nan")
        max_cube_z = initial_cube_z
        min_gripper_dist = _gripper_cube_distance(obs)

        frame_buffer = deque(maxlen=frame_stack)
        zero_frame = torch.zeros(3, cfg["data"]["image_size"], cfg["data"]["image_size"])
        for _ in range(frame_stack - 1):
            frame_buffer.append(zero_frame)
        frame_buffer.append(transform(obs[image_key]))

        max_steps = env_cfg.get("max_episode_steps", 500)
        step = 0
        while step < max_steps and not ep_success:
            # Plan: get action chunk from current observation
            img_tensor = torch.cat(list(frame_buffer), dim=0).unsqueeze(0).to(device)
            state_tensor = _get_state_tensor(obs)

            with torch.no_grad():
                # action_chunk: (1, pred_horizon, single_action_dim)
                action_chunk = model.get_action(img_tensor, state_tensor)
            action_chunk = action_chunk.squeeze(0).cpu().numpy()  # (pred_horizon, action_dim)

            if action_mean is not None and action_std is not None:
                action_chunk = action_chunk * action_std[None] + action_mean[None]

            # Execute exec_horizon actions before re-planning
            for i in range(exec_horizon):
                if step >= max_steps or ep_success:
                    break
                action = np.clip(action_chunk[i], action_low, action_high)
                obs, reward, done, _ = env.step(action)
                step += 1

                if "cube_pos" in obs:
                    max_cube_z = max(max_cube_z, float(obs["cube_pos"][2]))
                d = _gripper_cube_distance(obs)
                if np.isfinite(d):
                    min_gripper_dist = min(min_gripper_dist, d) if np.isfinite(min_gripper_dist) else d
                frame_buffer.append(transform(obs[image_key]))

                if done or reward > 0:
                    ep_success = True

        if ep_success:
            successes += 1
        episode_max_lift.append(
            max_cube_z - initial_cube_z
            if np.isfinite(max_cube_z) and np.isfinite(initial_cube_z)
            else float("nan")
        )
        episode_min_gripper_dist.append(min_gripper_dist)

    env.close()
    success_rate = successes / num_episodes
    print(f"\nSuccess rate: {successes}/{num_episodes} ({success_rate:.1%})")

    lift_arr = np.asarray(episode_max_lift, dtype=np.float64)
    dist_arr = np.asarray(episode_min_gripper_dist, dtype=np.float64)
    valid_lift = lift_arr[np.isfinite(lift_arr)]
    valid_dist = dist_arr[np.isfinite(dist_arr)]
    if valid_lift.size > 0:
        partial = float(np.mean(valid_lift >= partial_lift_threshold))
        print(
            f"Partial lift >= {partial_lift_threshold:.3f} m: {partial:.1%} | "
            f"max_lift mean: {float(np.mean(valid_lift)):.4f} m | median: {float(np.median(valid_lift)):.4f} m"
        )
    if valid_dist.size > 0:
        print(
            f"Min gripper-cube distance mean: {float(np.mean(valid_dist)):.4f} m | "
            f"median: {float(np.median(valid_dist)):.4f} m"
        )
    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_diffusion.yaml")
    parser.add_argument("--checkpoint", default="models_diffusion/best.pt")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open("configs/collect.yaml") as f:
        col_cfg = yaml.safe_load(f)
    cfg["env"] = col_cfg["env"]

    evaluate(cfg, args.checkpoint, args.episodes)
