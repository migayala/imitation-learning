# Standardized-Action Baseline Run — 2026-04-02

## Setup
- Train command: `./.venv/bin/python scripts/train.py --config configs/train_standardized.yaml`
- Eval command: `./.venv/bin/python scripts/evaluate.py --config configs/train_standardized.yaml --checkpoint models_standardized/best.pt --episodes 50`
- Dataset: `data/image.hdf5`
- Device: `cuda`
- Action standardization: enabled (train-split mean/std from `models_standardized/action_stats.json`)

## Checkpoint And Config Identity
- Best checkpoint: `models_standardized/best.pt`
- Best epoch: `80`
- Best checkpoint SHA-256: `714275aeacfcefad22734e3e31e07a81aa8649a4981a01e88bd39cd7b36d08bc`
- Config SHA-256: `88b8908998c48926cbe6bc2add8c57baaec272c5c688d21a6d0f9aab9be3cfc6`

## Training Metrics (Best Epoch = 80)
- `Loss/train`: `0.006838007364422083`
- `Loss/val`: `0.12622052431106567`
- `LossPerDim/val_mse_dim0`: `0.009432638064026833`
- `LossPerDim/val_mse_dim1`: `0.005311655346304178`
- `LossPerDim/val_mse_dim2`: `0.01483868807554245`
- `LossPerDim/val_mse_dim3`: `7.974526670295745e-05`
- `LossPerDim/val_mse_dim4`: `0.0002732704160735011`
- `LossPerDim/val_mse_dim5`: `0.00039000390097498894`
- `LossPerDim/val_mse_dim6`: `0.0781158059835434`

## Evaluation Result
- Episodes: `50`
- Successes: `0`
- Success rate: `0.0%`

## Comparison Notes
- Vs baseline `evaluation/baseline_2026-04-01.md`:
- Success rate: `0.0% -> 0.0%` (no change)
- `LossPerDim/val_mse_dim6`: `0.07262297719717026 -> 0.0781158059835434` (worse, +0.00549282878637314)
- `LossPerDim/val_mse_dim0`: `0.01064184121787548 -> 0.009432638064026833` (better)
- `LossPerDim/val_mse_dim1`: `0.00684210704639554 -> 0.005311655346304178` (better)
- `LossPerDim/val_mse_dim2`: `0.017107883468270302 -> 0.01483868807554245` (better)
- `LossPerDim/val_mse_dim3`: `0.0002445052668917924 -> 7.974526670295745e-05` (better)
- `LossPerDim/val_mse_dim4`: `0.000474189524538815 -> 0.0002732704160735011` (better)
- `LossPerDim/val_mse_dim5`: `0.0008148764027282596 -> 0.00039000390097498894` (better)
- Vs dim6-weighted `evaluation/dim6_weighted_2026-04-02.md`:
- `LossPerDim/val_mse_dim6`: `0.09631475806236267 -> 0.0781158059835434` (better than weighted)

## Conclusion
- Standardizing actions improved 6/7 per-dimension validation MSE values versus the original baseline, but dim6 got slightly worse and policy success remained `0/50`.
