# Diffusion Policy Final Results — 2026-04-07

## Result

**100% success (50/50) on robosuite Lift** using Diffusion Policy trained on 1,000 scripted expert demonstrations.

---

## Summary

The architecture was correct from the start. The bottleneck was data: 200 human demonstrations were insufficient for a 13M-parameter model to reliably learn the approach→grasp→lift phase transition. Generating 1,000 clean scripted demonstrations (5.3× more data, zero noise in labels) resolved the failure entirely.

---

## Dataset Comparison

| Dataset | Demos | Total Steps | Source | Best Val Loss | Success Rate |
|---------|-------|-------------|--------|--------------|--------------|
| robomimic human (lift_ph) | 200 | ~9,666 | Human teleoperation | 0.1662 (ep96) | 1.5% (3/200) |
| scripted expert | 1,000 | 51,198 | State-machine policy | **0.0070 (ep167)** | **100% (50/50)** |

Val loss improvement: **24×** (0.1662 → 0.0070).

---

## Scripted Expert Policy

A 4-phase state machine operating in OSC_POSE delta action space:

| Phase | Behavior | Transition |
|-------|----------|------------|
| HOVER | Move to (cube_x, cube_y, cube_z + 0.15 m), gripper open | dist to target < 0.025 m |
| DESCEND | Lower to cube center, slow to 35% speed within 0.08 m | dist to cube < 0.025 m |
| GRASP | Close gripper, hold 20 steps | step count |
| LIFT | Move straight up, gripper closed | env `done=True` |

Collection: 1,000/1,000 episodes succeeded (100%), ~1.67 demos/sec, 10 minutes wall time.

Key engineering detail: full-speed descent with an open gripper pushes the cube laterally due to finger contact forces. The 35% speed scale (`_DESCENT_SLOW_SCALE = 0.35`) in the final approach eliminates this, allowing convergence to within 0.016 m of the cube center.

---

## Training Configuration

Config: `configs/train_diffusion_scripted.yaml`

| Parameter | Value |
|-----------|-------|
| pred_horizon | 8 |
| exec_horizon | 8 |
| ddim_steps (train) | 10 |
| state_dim | 19 (eef_pos + eef_quat + gripper_qpos + object) |
| dropout | 0.1 |
| epochs | 200 |
| best epoch | 167 |
| train steps/epoch | 720 batches × 64 = ~46k steps |

---

## Why Scripted Demos Succeeded

1. **No label noise**: every demo is a perfect hover→descend→grasp→lift. Human demos include suboptimal corrections, hesitations, and gripper oscillation that create conflicting training signal.
2. **Sharp mode**: the gripper closes decisively at the right moment in every demo. The model learns a clear "close gripper + lift" policy rather than averaging toward "hover near cube."
3. **5× more grasp examples**: with 200 demos and ~15/59 grasp-phase steps, only ~2,500 training steps show closed-gripper behavior. With 1,000 scripted demos, this rises to ~15,000 — enough to learn the mode reliably.

---

## Evaluation Notes

**Partial lift ≥ 0.03 m: 0.0%** — this is expected and not a failure. The metric measures cube lift height at episode end, but successful episodes terminate with `done=True` when the task is complete. The `max_cube_z - initial_cube_z` delta does not accumulate a 3 cm threshold before termination in many episodes. The 100% success rate is the ground-truth metric.

**Min gripper-cube distance**: mean 0.0194 m, median 0.0193 m — consistent approach to the cube in every episode.

---

## Conclusion

Data quality and quantity dominate over architecture choice for this task complexity and model size. The diffusion policy with exec_horizon=8 and ddim_steps=10 (training) is the right architecture; it simply requires sufficient demonstrations of each phase transition to generalize reliably.

For future work: the scripted expert trajectory style may limit visual diversity. Image augmentation (random crop, color jitter) during training would improve robustness to cube position and lighting variation without requiring more data collection.
