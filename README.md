# Imitation Learning for Robotic Manipulation

Behavior cloning on the robosuite `Lift` task using a ResNet18 vision encoder + MLP policy head.

## Setup

```bash
conda activate robotics
pip install -r requirements.txt
```

Canonical training dataset: `data/image.hdf5` (robomimic image schema).

## Pipeline

### 1. Optional: collect random rollouts (debug only)
```bash
python3 scripts/collect_demos.py --config configs/collect.yaml
```
This script collects random-policy trajectories and writes robomimic-compatible structure. It is not an expert policy collector.

### 2. Train
```bash
python3 scripts/train.py --config configs/train.yaml
```

### 3. Monitor training
```bash
tensorboard --logdir runs
```

### 4. Evaluate
```bash
python3 scripts/evaluate.py --config configs/train.yaml --checkpoint models/best.pt
```

## Data Format

`scripts/dataset.py` expects `robomimic_image` schema:
- top-level group: `data`
- per-episode group: `data/demo_x`
- required keys: `obs/<camera_key>`, `actions`

The default camera key in `configs/train.yaml` is `agentview_image`.

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
