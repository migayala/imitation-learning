# Imitation Learning for Robotic Manipulation

Behavior cloning on the robosuite `Lift` task using a ResNet18 vision encoder + MLP policy head.

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

Canonical training dataset: `data/image.hdf5` (robomimic image schema).

## Pipeline

### 1. Optional: collect random rollouts (debug only)
```bash
python3 scripts/collect_demos.py --config configs/collect.yaml
```
This script collects random-policy trajectories and writes robomimic-compatible structure. It is not an expert policy collector.

### 2. Train
```bash
# Baseline (unweighted)
python3 scripts/train.py --config configs/train.yaml

# Dim6 weighted variant (A/B)
python3 scripts/train.py --config configs/train_dim6_weighted.yaml
```
`configs/train_dim6_weighted.yaml` sets `training.action_loss_weights: [1, 1, 1, 1, 1, 1, 3]` and writes outputs to `runs_dim6_weighted/` and `models_dim6_weighted/`.

### 3. Monitor training
```bash
tensorboard --logdir runs,runs_dim6_weighted
```
Logged scalars include aggregate loss (`Loss/train`, `Loss/val`) and per-action-dimension MSE (`LossPerDim/*`).

### 4. Evaluate
```bash
# Baseline
python3 scripts/evaluate.py --config configs/train.yaml --checkpoint models/best.pt

# Dim6 weighted variant
python3 scripts/evaluate.py --config configs/train_dim6_weighted.yaml --checkpoint models_dim6_weighted/best.pt
```

### 5. Smoke test
```bash
python3 -m unittest tests/test_smoke.py
```

### 6. Inspect dataset schema
```bash
python3 scripts/inspect_hdf5.py --path data/image.hdf5 --max-demos 5 --camera agentview_image
```

## Data Format

`scripts/dataset.py` expects `robomimic_image` schema:
- top-level group: `data`
- per-episode group: `data/demo_x`
- required keys: `obs/<camera_key>`, `actions`

The default camera key in `configs/train.yaml` is `agentview_image`.
Training writes a config snapshot to `<checkpoint_dir>/train_config_snapshot.yaml`.
Training also writes JSON metadata files in the checkpoint directory (`run_metadata.json`, `best.metadata.json`, and `checkpoint_ep*.metadata.json`).
CI runs `py_compile` and `tests/test_smoke.py` on each push and pull request.

## Structure

```
imitation-learning/
├── configs/
│   ├── collect.yaml    # data collection settings
│   └── train.yaml      # training hyperparameters
├── data/               # HDF5 demonstration files
├── models/             # saved checkpoints
├── runs/               # tensorboard logs
├── requirements.txt    # python dependencies
├── scripts/
│   ├── collect_demos.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── evaluation/
```
