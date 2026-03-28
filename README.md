# Imitation Learning for Robotic Manipulation

Behavior cloning on the robosuite `Lift` task using a ResNet18 vision encoder + MLP policy head.

## Setup

```bash
conda activate robotics
```

## Pipeline

### 1. Collect expert demonstrations
```bash
cd scripts
python collect_demos.py --config ../configs/collect.yaml
```

### 2. Train
```bash
python train.py --config ../configs/train.yaml
```

### 3. Monitor training
```bash
tensorboard --logdir ../runs
```

### 4. Evaluate
```bash
python evaluate.py --checkpoint ../models/best.pt
```

## Structure

```
imitation-learning/
├── configs/
│   ├── collect.yaml    # data collection settings
│   └── train.yaml      # training hyperparameters
├── data/               # HDF5 demonstration files
├── models/             # saved checkpoints
├── runs/               # tensorboard logs
├── scripts/
│   ├── collect_demos.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── evaluation/
```
