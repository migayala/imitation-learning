# Imitation Learning for Robotic Manipulation

Behavior cloning and Diffusion Policy on the robosuite `Lift` task using a ResNet18 vision encoder.

## Results

| Method | Dataset | Success Rate |
|--------|---------|--------------|
| Behavior Cloning (MSE) | 200 human demos | 0% |
| Diffusion Policy (ablation best) | 200 human demos | 1.5% (3/200) |
| **Diffusion Policy (final)** | **1,000 scripted demos** | **100% (50/50)** |

The architecture was correct from the start — data was the bottleneck. 200 demos were insufficient for the 13M-parameter model to reliably learn the grasp phase transition. Generating 1,000 clean scripted demonstrations reduced best validation loss 24× (0.1662 → 0.0070) and lifted success to 100%.

See [`run_reports/2026-04-07_final_results.md`](run_reports/2026-04-07_final_results.md) for the full write-up and [`run_reports/2026-04-07_diffusion_policy_ablation.md`](run_reports/2026-04-07_diffusion_policy_ablation.md) for the ablation study.

## Setup

```bash
# Option A: existing conda env
conda activate robotics
pip install -r requirements.txt

# Option B: local venv
# python3 -m venv .venv
# . .venv/bin/activate
# pip install -r requirements.txt

# Optional (simulator + robomimic tooling):
# pip install -r requirements-sim.txt
```
`requirements-sim.txt` may require additional system tools (for example `cmake`) depending on platform.

Canonical training dataset: `data/image.hdf5` (robomimic image schema, 200 proficient human demos).

## Pipeline

### 1. Collect scripted expert demonstrations (recommended)
```bash
python3 scripts/collect_scripted.py --episodes 1000 --output data/scripted_expert.hdf5
```
Generates demonstrations via a 4-phase state machine (hover → descend → grasp → lift) with 100% collection success rate. Produces ~51k timesteps of clean, consistent training data. See `scripts/collect_scripted.py` for the policy implementation.

### 2. Optional: collect random rollouts (debug only)
```bash
python3 scripts/collect_demos.py --config configs/collect.yaml
```
Random-policy trajectories in robomimic-compatible structure. Not useful for training.

### 3. Train

#### Diffusion Policy (recommended)
```bash
python3 scripts/train_diffusion.py --config configs/train_diffusion_scripted.yaml
```

#### Behavior Cloning variants
```bash
# Baseline (unweighted)
python3 scripts/train.py --config configs/train.yaml

# Dim6 weighted variant (A/B)
python3 scripts/train.py --config configs/train_dim6_weighted.yaml

# Action-standardized baseline variant (A/B)
python3 scripts/train.py --config configs/train_standardized.yaml

# 4-frame stacked + standardized variant (temporal context A/B)
python3 scripts/train.py --config configs/train_stacked4_standardized.yaml
```

Training computes action mean/std on the train split, normalizes action targets for optimization, and saves stats to `<checkpoint_dir>/action_stats.json`.

### 4. Monitor training
```bash
tensorboard --logdir runs_diffusion_scripted
# or for BC variants:
tensorboard --logdir runs,runs_dim6_weighted,runs_standardized,runs_stacked4_standardized
```

### 5. Evaluate

#### Diffusion Policy
```bash
python3 scripts/evaluate_diffusion.py \
  --config configs/train_diffusion_scripted.yaml \
  --checkpoint models_diffusion_scripted/best.pt \
  --episodes 50
```

#### Behavior Cloning
```bash
python3 scripts/evaluate.py --config configs/train.yaml --checkpoint models/best.pt
python3 scripts/evaluate.py --config configs/train_standardized.yaml --checkpoint models_standardized/best.pt
```

Evaluation auto-loads `<checkpoint_dir>/action_stats.json` and unnormalizes model outputs before environment stepping. Reports max lift delta and min gripper-to-cube distance per episode.

### 6. Smoke test
```bash
python3 -m unittest tests/test_smoke.py
```

### 7. Inspect dataset schema
```bash
python3 scripts/inspect_hdf5.py --path data/image.hdf5 --max-demos 5 --camera agentview_image
```

## Data Format

`scripts/dataset.py` expects `robomimic_image` schema:
- top-level group: `data`
- per-episode group: `data/demo_x`
- required keys: `obs/<camera_key>`, `actions`

The default camera key is `agentview_image`. Training writes a config snapshot to `<checkpoint_dir>/train_config_snapshot.yaml` and JSON metadata files (`run_metadata.json`, `best.metadata.json`, `checkpoint_ep*.metadata.json`).

CI runs `py_compile` and `tests/test_smoke.py` on each push and pull request.

## Structure

```
imitation-learning/
├── configs/
│   ├── collect.yaml                    # data collection settings
│   ├── train.yaml                      # BC training hyperparameters
│   ├── train_diffusion_scripted.yaml   # Diffusion Policy (final config)
│   └── train_diffusion*.yaml           # Diffusion Policy ablation variants
├── data/                               # HDF5 demonstration files
├── run_reports/                        # experiment write-ups
├── requirements.txt
├── scripts/
│   ├── collect_demos.py                # random rollout collection (debug)
│   ├── collect_scripted.py             # scripted expert data collection
│   ├── dataset.py                      # PyTorch Dataset for robomimic HDF5
│   ├── model.py                        # BCPolicy (ResNet18 + MLP head)
│   ├── diffusion_policy.py             # DiffusionPolicy (DDPM/DDIM)
│   ├── train.py                        # BC training loop
│   ├── train_diffusion.py              # Diffusion Policy training loop
│   ├── evaluate.py                     # BC evaluation
│   └── evaluate_diffusion.py           # Diffusion Policy evaluation
└── tests/
```
