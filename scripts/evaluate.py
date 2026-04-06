"""
Evaluate a trained BC policy in the robosuite environment.
Reports success rate over N rollouts.
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

from model import BCPolicy


def load_action_stats(checkpoint: str):
    action_stats_path = os.path.join(os.path.dirname(checkpoint), "action_stats.json")
    if not os.path.exists(action_stats_path):
        return None, None

    with open(action_stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)
    return action_mean, action_std


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


def evaluate(cfg, checkpoint: str, num_episodes: int = 50):
    seed = cfg.get("evaluation", {}).get("seed", cfg["training"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_name = cfg["env"]["camera_names"]
    if isinstance(camera_name, (list, tuple)):
        camera_name = camera_name[0]
    image_key = f"{camera_name}_image"
    frame_stack = int(cfg["data"].get("frame_stack", 1))
    if frame_stack < 1:
        raise ValueError(f"data.frame_stack must be >= 1, got {frame_stack}")
    partial_lift_threshold = float(cfg.get("evaluation", {}).get("partial_lift_threshold", 0.03))

    # Load model
    model = BCPolicy(
        action_dim=cfg["model"]["action_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        freeze_encoder=cfg["model"]["freeze_encoder"],
        in_channels=3 * frame_stack,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    action_mean, action_std = load_action_stats(checkpoint)
    if action_mean is None or action_std is None:
        print("Warning: action_stats.json not found. Using raw model outputs without unnormalization.")
    else:
        print(f"Loaded action stats from: {os.path.join(os.path.dirname(checkpoint), 'action_stats.json')}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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

    successes = 0
    episode_max_lift = []
    episode_min_gripper_dist = []
    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        ep_success = False
        initial_cube_z = float(obs["cube_pos"][2]) if "cube_pos" in obs else float("nan")
        max_cube_z = initial_cube_z
        min_gripper_dist = _gripper_cube_distance(obs)
        frame_buffer = deque(maxlen=frame_stack)
        current_frame = transform(obs[image_key])
        zero_frame = torch.zeros_like(current_frame)
        for _ in range(frame_stack - 1):
            frame_buffer.append(zero_frame)
        frame_buffer.append(current_frame)

        for _ in range(env_cfg["max_episode_steps"] if "max_episode_steps" in env_cfg else 500):
            img_tensor = torch.cat(list(frame_buffer), dim=0).unsqueeze(0).to(device)  # (1, 3K, H, W)

            with torch.no_grad():
                action = model(img_tensor).squeeze(0).cpu().numpy()
            if action_mean is not None and action_std is not None:
                action = action * action_std + action_mean
            action = np.clip(action, action_low, action_high)

            obs, reward, done, info = env.step(action)
            if "cube_pos" in obs:
                max_cube_z = max(max_cube_z, float(obs["cube_pos"][2]))
            gripper_dist = _gripper_cube_distance(obs)
            if np.isfinite(gripper_dist):
                min_gripper_dist = min(min_gripper_dist, gripper_dist) if np.isfinite(min_gripper_dist) else gripper_dist
            frame_buffer.append(transform(obs[image_key]))
            if done or reward > 0:
                ep_success = True
                break

        if ep_success:
            successes += 1
        episode_max_lift.append(max_cube_z - initial_cube_z if np.isfinite(max_cube_z) and np.isfinite(initial_cube_z) else float("nan"))
        episode_min_gripper_dist.append(min_gripper_dist)

    env.close()
    success_rate = successes / num_episodes
    print(f"\nSuccess rate: {successes}/{num_episodes} ({success_rate:.1%})")
    lift_array = np.asarray(episode_max_lift, dtype=np.float64)
    dist_array = np.asarray(episode_min_gripper_dist, dtype=np.float64)
    valid_lift = lift_array[np.isfinite(lift_array)]
    valid_dist = dist_array[np.isfinite(dist_array)]
    if valid_lift.size > 0:
        partial_rate = float(np.mean(valid_lift >= partial_lift_threshold))
        print(
            f"Partial lift >= {partial_lift_threshold:.3f} m: {partial_rate:.1%} | "
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
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Merge collect config for env params
    with open("configs/collect.yaml") as f:
        col_cfg = yaml.safe_load(f)
    cfg["env"] = col_cfg["env"]

    evaluate(cfg, args.checkpoint, args.episodes)
