"""
Collect expert demonstrations for robosuite Lift using a scripted state-machine policy.
All saved demos are successful (100% success rate by design).

State machine (OSC_POSE controller, control_delta=True, output_max=0.05 m/step):
  Phase 0 - HOVER   : move to (cube_x, cube_y, cube_z + HOVER_HEIGHT), gripper open
  Phase 1 - DESCEND : lower to (cube_x, cube_y, cube_z + GRASP_OFFSET), gripper open
  Phase 2 - GRASP   : close gripper for GRASP_STEPS steps
  Phase 3 - LIFT    : move straight up until success, gripper closed

HDF5 output matches the robomimic_image schema expected by dataset.py.

Usage:
    python scripts/collect_scripted.py --episodes 1000 --output data/scripted_expert.hdf5
    python scripts/collect_scripted.py --episodes 1 --verbose   # dry-run / debug
"""

import argparse
import json
import os

import h5py
import numpy as np
import robosuite as suite
from tqdm import tqdm


# --- Policy parameters -------------------------------------------------------

HOVER_HEIGHT = 0.15   # m above cube centre to hover before descending
GRASP_OFFSET = 0.00   # m above cube centre to target for grasp
GRASP_STEPS  = 20     # steps to hold gripper closed before lifting
HOVER_THRESH = 0.025  # m — arrival tolerance for hover phase
GRASP_THRESH = 0.025  # m — slow descent converges to ~0.020 m; threshold slightly above that
# Speed scale during final approach (avoids pushing cube with open fingers at high speed)
_DESCENT_SLOW_DIST  = 0.08  # switch to slow speed below this dist-to-target
_DESCENT_SLOW_SCALE = 0.35
# Per-phase step timeouts: advance even if threshold not met (handles edge cases)
_HOVER_TIMEOUT   = 40
_DESCENT_TIMEOUT = 80

# OSC_POSE output_max for position is 0.05 m/step; clip normalised actions to [-1, 1]
_POS_SCALE = 0.05

# Gripper convention in this robosuite build: -1 = open, +1 = close
# (verified from gripper_qpos in dataset: action=-1 → fingers at 0.04 = open)
GRIP_OPEN  = -1.0
GRIP_CLOSE =  1.0

# Obs key rename: live env key → HDF5 storage key (must match dataset.py expectations)
_OBS_REMAP = {"object-state": "object"}

# Controller config matching the original robomimic Lift dataset


# --- Scripted policy ---------------------------------------------------------

def _scripted_action(eef_pos: np.ndarray, cube_pos: np.ndarray, phase: int, phase_step: int):
    """Return (7D action in [-1,1], advance_phase: bool)."""
    action = np.zeros(7, dtype=np.float64)

    if phase == 0:  # HOVER: fly to position above cube
        target = cube_pos + np.array([0.0, 0.0, HOVER_HEIGHT])
        delta  = target - eef_pos
        action[:3] = np.clip(delta / _POS_SCALE, -1.0, 1.0)
        action[6]  = GRIP_OPEN
        close_enough = bool(np.linalg.norm(delta) < HOVER_THRESH)
        return action, close_enough or phase_step >= _HOVER_TIMEOUT

    elif phase == 1:  # DESCEND: lower to grasp height
        target = cube_pos + np.array([0.0, 0.0, GRASP_OFFSET])
        delta  = target - eef_pos
        action[:3] = np.clip(delta / _POS_SCALE, -1.0, 1.0)
        # Slow down near the target — full-speed approach pushes the cube with open fingers
        if np.linalg.norm(delta) < _DESCENT_SLOW_DIST:
            action[:3] *= _DESCENT_SLOW_SCALE
        action[6]  = GRIP_OPEN
        close_enough = bool(np.linalg.norm(delta) < GRASP_THRESH)
        return action, close_enough or phase_step >= _DESCENT_TIMEOUT

    elif phase == 2:  # GRASP: close gripper, stay in place
        action[6] = GRIP_CLOSE
        return action, phase_step >= GRASP_STEPS

    elif phase == 3:  # LIFT: move straight up, keep gripper closed
        action[2] = 1.0          # max upward velocity
        action[6] = GRIP_CLOSE
        return action, False      # env done signal ends this phase

    return action, False


# --- Episode collection ------------------------------------------------------

def _run_episode(env, camera_obs_key: str, max_steps: int, verbose: bool = False):
    """
    Execute one scripted episode.

    Returns (ep_obs, ep_actions, ep_rewards, ep_dones) on success, or None on failure.
    ep_obs is a dict mapping storage_key → list of np.ndarray (one per step).
    """
    obs = env.reset()
    action_low, action_high = env.action_spec

    # Determine which obs keys to persist (using renamed keys for HDF5)
    save_keys = [(env_k, _OBS_REMAP.get(env_k, env_k)) for env_k in obs]
    ep_obs = {store_k: [] for _, store_k in save_keys}

    ep_act: list[np.ndarray] = []
    ep_rew: list[float]       = []
    ep_don: list[bool]        = []

    phase      = 0
    phase_step = 0

    for step in range(max_steps):
        eef_pos  = obs["robot0_eef_pos"]
        cube_pos = obs["object-state"][:3]   # cube_pos is dims [0:3] of object-state

        action, advance = _scripted_action(eef_pos, cube_pos, phase, phase_step)
        action = np.clip(action, action_low, action_high)

        if verbose:
            dist = np.linalg.norm(eef_pos - cube_pos)
            print(
                f"  step={step:3d} phase={phase} phase_step={phase_step:2d} "
                f"dist={dist:.3f} grip={action[6]:+.1f} dz={action[2]:+.2f}"
            )

        # Record current observation
        for env_k, store_k in save_keys:
            val = obs[env_k]
            ep_obs[store_k].append(val.copy() if isinstance(val, np.ndarray) else np.asarray(val))
        ep_act.append(action.copy())

        obs, reward, done, _ = env.step(action)
        ep_rew.append(float(reward))
        ep_don.append(bool(done))

        if reward > 0:
            if verbose:
                print(f"  SUCCESS at step {step} (reward={reward:.2f})")
            return (
                ep_obs,
                np.array(ep_act,  dtype=np.float64),
                np.array(ep_rew,  dtype=np.float64),
                np.array(ep_don,  dtype=bool),
            )

        if advance:
            if verbose:
                print(f"  → phase {phase} → {phase + 1}")
            phase      = min(phase + 1, 3)
            phase_step = 0
        else:
            phase_step += 1

    return None   # failed: exceeded max_steps without reward


# --- Main --------------------------------------------------------------------

def collect(args):
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        reward_shaping=False,
        control_freq=20,
    )

    camera_obs_key = "agentview_image"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    demos:    list = []
    attempts: int  = 0
    pbar = tqdm(total=args.episodes, desc="Collecting", unit="demo")

    while len(demos) < args.episodes:
        attempts += 1
        if args.verbose:
            print(f"\n--- Episode {len(demos) + 1} (attempt {attempts}) ---")
        result = _run_episode(env, camera_obs_key, args.max_steps, verbose=args.verbose)
        if result is not None:
            demos.append(result)
            pbar.update(1)
        if attempts >= args.episodes * 5:
            print(f"\nWarning: only {len(demos)}/{attempts} episodes succeeded — stopping.")
            break

    pbar.close()
    env.close()

    success_rate = len(demos) / attempts if attempts > 0 else 0.0
    print(f"Success rate: {len(demos)}/{attempts} ({success_rate:.1%})")

    if not demos:
        print("No successful episodes — nothing to save.")
        return

    # --- Save HDF5 -----------------------------------------------------------
    print(f"Saving {len(demos)} demos to {args.output} ...")
    with h5py.File(args.output, "w") as f:
        data_grp = f.create_group("data")
        total_steps = 0

        for i, (ep_obs, ep_act, ep_rew, ep_don) in enumerate(demos):
            n = len(ep_act)
            demo_grp = data_grp.create_group(f"demo_{i}")
            demo_grp.attrs["num_samples"] = n

            obs_grp = demo_grp.create_group("obs")
            for store_k, frames in ep_obs.items():
                arr = np.array(frames)
                obs_grp.create_dataset(store_k, data=arr, compression="gzip")

            demo_grp.create_dataset("actions", data=ep_act, compression="gzip")
            demo_grp.create_dataset("rewards", data=ep_rew)
            demo_grp.create_dataset("dones",   data=ep_don)
            total_steps += n

        data_grp.attrs["total"] = total_steps

    print(f"Done. {len(demos)} episodes, {total_steps} steps → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int, default=1000,
                        help="Number of successful demos to collect")
    parser.add_argument("--output",    default="data/scripted_expert.hdf5")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode before declaring failure")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print step-level trace for debugging")
    args = parser.parse_args()
    collect(args)
