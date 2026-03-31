"""
Evaluate a trained BC policy in the robosuite environment.
Reports success rate over N rollouts.
"""

import argparse
import yaml
import torch
import numpy as np
import robosuite as suite
from torchvision import transforms
from tqdm import tqdm

from model import BCPolicy


def evaluate(cfg, checkpoint: str, num_episodes: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_name = cfg["env"]["camera_names"]
    if isinstance(camera_name, (list, tuple)):
        camera_name = camera_name[0]
    image_key = f"{camera_name}_image"

    # Load model
    model = BCPolicy(
        action_dim=cfg["model"]["action_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        freeze_encoder=cfg["model"]["freeze_encoder"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

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
    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        ep_success = False

        for _ in range(env_cfg["max_episode_steps"] if "max_episode_steps" in env_cfg else 500):
            img = obs[image_key]  # (H, W, C) uint8
            img_tensor = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

            with torch.no_grad():
                action = model(img_tensor).squeeze(0).cpu().numpy()
            action = np.clip(action, action_low, action_high)

            obs, reward, done, info = env.step(action)
            if done or reward > 0:
                ep_success = True
                break

        if ep_success:
            successes += 1

    env.close()
    success_rate = successes / num_episodes
    print(f"\nSuccess rate: {successes}/{num_episodes} ({success_rate:.1%})")
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
