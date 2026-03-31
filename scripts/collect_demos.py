"""
Collect random-policy rollouts in robomimic-compatible HDF5 format.
This script is for quick data generation/debugging, not expert demonstration quality.
"""

import argparse
import os
import yaml
import h5py
import numpy as np
import robosuite as suite
from tqdm import tqdm


def collect(cfg):
    env_cfg = cfg["env"]
    col_cfg = cfg["collection"]
    camera_name = env_cfg["camera_names"]
    if isinstance(camera_name, (list, tuple)):
        camera_name = camera_name[0]
    camera_obs_key = f"{camera_name}_image"

    env = suite.make(
        env_name=env_cfg["name"],
        robots=env_cfg["robot"],
        has_renderer=env_cfg["has_renderer"],
        has_offscreen_renderer=env_cfg["has_offscreen_renderer"],
        use_camera_obs=env_cfg["use_camera_obs"],
        camera_names=env_cfg["camera_names"],
        camera_heights=env_cfg["camera_heights"],
        camera_widths=env_cfg["camera_widths"],
        reward_shaping=env_cfg["reward_shaping"],
        control_freq=env_cfg["control_freq"],
    )

    output_path = col_cfg["output_path"]
    num_episodes = col_cfg["num_episodes"]
    max_steps = col_cfg["max_episode_steps"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    observations = []
    actions = []
    rewards = []
    dones = []

    print(
        "Collecting random-policy rollouts. "
        "Use data/image.hdf5 as the canonical expert dataset for training."
    )
    print(f"Episodes: {num_episodes}")

    for ep in tqdm(range(num_episodes)):
        obs = env.reset()
        ep_obs, ep_act, ep_rew, ep_done = [], [], [], []

        for _ in range(max_steps):
            low, high = env.action_spec
            action = np.random.uniform(low, high)

            next_obs, reward, done, _ = env.step(action)

            ep_obs.append(obs[camera_obs_key])
            ep_act.append(action)
            ep_rew.append(reward)
            ep_done.append(done)

            obs = next_obs
            if done:
                break

        observations.append(ep_obs)
        actions.append(ep_act)
        rewards.append(ep_rew)
        dones.append(ep_done)

    env.close()

    # Save to HDF5
    print(f"Saving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        data_group = f.create_group("data")
        for i, (obs, act, rew, don) in enumerate(zip(observations, actions, rewards, dones)):
            demo_group = data_group.create_group(f"demo_{i}")
            obs_group = demo_group.create_group("obs")
            obs_group.create_dataset(
                camera_obs_key,
                data=np.asarray(obs, dtype=np.uint8),
                compression="gzip",
            )
            demo_group.create_dataset(
                "actions",
                data=np.asarray(act, dtype=np.float32),
                compression="gzip",
            )
            demo_group.create_dataset("rewards", data=np.asarray(rew, dtype=np.float32))
            demo_group.create_dataset("dones", data=np.asarray(don, dtype=bool))
            demo_group.attrs["num_samples"] = len(act)

    print(f"Saved {num_episodes} random-rollout episodes to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/collect.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    collect(cfg)
