"""
Debug script: trace gripper action vs cube distance for one episode,
to determine if the model EVER predicts close-gripper when near the cube.
"""

import json
import os
import sys
import numpy as np
import torch
import yaml
from collections import deque
from torchvision import transforms
import robosuite as suite

sys.path.insert(0, os.path.dirname(__file__))
from diffusion_policy import DiffusionPolicy


def load_stats(checkpoint):
    base = os.path.dirname(checkpoint)
    with open(os.path.join(base, "action_stats.json")) as f:
        a = json.load(f)
    with open(os.path.join(base, "state_stats.json")) as f:
        s = json.load(f)
    return (
        np.array(a["action_mean"], dtype=np.float32),
        np.array(a["action_std"], dtype=np.float32),
        np.array(s["state_mean"], dtype=np.float32),
        np.array(s["state_std"], dtype=np.float32),
        s.get("state_keys", []),
    )


_DATASET_TO_ENV_KEY = {"object": "object-state"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_diffusion.yaml")
    parser.add_argument("--checkpoint", default="models_diffusion/best.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open("configs/collect.yaml") as f:
        col_cfg = yaml.safe_load(f)
    cfg["env"] = col_cfg["env"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_mean, action_std, state_mean, state_std, state_keys = load_stats(args.checkpoint)

    model = DiffusionPolicy(
        action_dim=cfg["model"]["action_dim"],
        hidden_dim=cfg["model"].get("hidden_dim", 256),
        cond_dim=cfg["model"].get("cond_dim", 256),
        n_blocks=cfg["model"].get("n_blocks", 4),
        T=cfg["model"].get("T", 100),
        ddim_steps=cfg["model"].get("ddim_steps", 10),
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        in_channels=3,
        state_dim=cfg["model"].get("state_dim", 0),
        pred_horizon=cfg["model"].get("pred_horizon", 1),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    env_cfg = cfg["env"]
    camera_name = env_cfg["camera_names"]
    if isinstance(camera_name, (list, tuple)):
        camera_name = camera_name[0]
    image_key = f"{camera_name}_image"

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

    np.random.seed(42)
    torch.manual_seed(42)
    obs = env.reset()

    frame_buffer = deque(maxlen=1)
    frame_buffer.append(transform(obs[image_key]))

    print(f"{'step':>5} {'dist':>8} {'grip_pred':>10} {'grip_unnorm':>12} {'eef_z':>8} {'cube_z':>8}")
    print("-" * 60)

    max_steps = 200
    step = 0
    while step < max_steps:
        img_tensor = torch.cat(list(frame_buffer), dim=0).unsqueeze(0).to(device)

        # Build state tensor
        parts = [obs[_DATASET_TO_ENV_KEY.get(k, k)].astype(np.float32).ravel() for k in state_keys]
        state_arr = np.concatenate(parts)
        state_arr = (state_arr - state_mean) / state_std
        state_tensor = torch.tensor(state_arr).unsqueeze(0).to(device)

        with torch.no_grad():
            action_chunk = model.get_action(img_tensor, state_tensor)
        action_chunk = action_chunk.squeeze(0).cpu().numpy()  # (pred_horizon, 7)
        action_chunk_unnorm = action_chunk * action_std[None] + action_mean[None]

        # Compute distance
        eef = obs["robot0_eef_pos"]
        cube = obs["cube_pos"]
        dist = float(np.linalg.norm(eef - cube))

        # Log first action of chunk
        grip_norm = action_chunk[0, 6]
        grip_unnorm = action_chunk_unnorm[0, 6]

        if step % 4 == 0 or dist < 0.10:
            print(f"{step:>5} {dist:>8.4f} {grip_norm:>10.3f} {grip_unnorm:>12.3f} {eef[2]:>8.4f} {cube[2]:>8.4f}")

        # Execute exec_horizon steps
        for i in range(exec_horizon):
            if step >= max_steps:
                break
            action = np.clip(action_chunk_unnorm[i], action_low, action_high)
            obs, reward, done, _ = env.step(action)
            step += 1
            frame_buffer.append(transform(obs[image_key]))
            if done or reward > 0:
                print(f"SUCCESS at step {step}!")
                env.close()
                sys.exit(0)

    print(f"\nEpisode ended at step {step} without success")
    env.close()
