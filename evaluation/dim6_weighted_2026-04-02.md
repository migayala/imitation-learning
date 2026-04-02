# Dim6-Weighted Run — 2026-04-02

## Setup
- Train command: `./.venv/bin/python scripts/train.py --config configs/train_dim6_weighted.yaml`
- Eval command: `./.venv/bin/python scripts/evaluate.py --config configs/train_dim6_weighted.yaml --checkpoint models_dim6_weighted/best.pt --episodes 50`
- Dataset: `data/image.hdf5`
- Device: `cuda`
- Loss weights: `[1, 1, 1, 1, 1, 1, 3]`

## Checkpoint And Config Identity
- Best checkpoint: `models_dim6_weighted/best.pt`
- Best epoch: `55`
- Best checkpoint SHA-256: `c95d04c0d6e013e156b6f4731d22c7e8f3c4589562836d3b0b4486e3fcf857a3`
- Config SHA-256: `158ca85f40446cc671516a9cbba51af86f941572116bf7c9f30b767d70542e70`

## Training Metrics (Best Epoch = 55)
- `Loss/train`: `0.0022000421304255724`
- `Loss/val`: `0.04551167041063309`
- `LossPerDim/val_mse_dim0`: `0.012445838190615177`
- `LossPerDim/val_mse_dim1`: `0.00822928175330162`
- `LossPerDim/val_mse_dim2`: `0.022053619846701622`
- `LossPerDim/val_mse_dim3`: `0.00034800180583260953`
- `LossPerDim/val_mse_dim4`: `0.0005731947021558881`
- `LossPerDim/val_mse_dim5`: `0.0012102401815354824`
- `LossPerDim/val_mse_dim6`: `0.09631475806236267`

## Evaluation Result
- Episodes: `50`
- Successes: `0`
- Success rate: `0.0%`

## A/B Comparison Vs Baseline (2026-04-01)
- Baseline reference: `evaluation/baseline_2026-04-01.md`
- Success rate: `0.0% -> 0.0%` (no change)
- Best `Loss/val`: `0.01522385235875845 -> 0.04551167041063309` (worse, +0.03028781805187464)
- `LossPerDim/val_mse_dim6`: `0.07262297719717026 -> 0.09631475806236267` (worse, +0.02369178086519241)
- Other val dimensions also increased:
- `dim0`: `0.01064184121787548 -> 0.012445838190615177`
- `dim1`: `0.00684210704639554 -> 0.00822928175330162`
- `dim2`: `0.017107883468270302 -> 0.022053619846701622`
- `dim3`: `0.0002445052668917924 -> 0.00034800180583260953`
- `dim4`: `0.000474189524538815 -> 0.0005731947021558881`
- `dim5`: `0.0008148764027282596 -> 0.0012102401815354824`

## Conclusion
- This `dim6 x3` weighting did not improve task success and degraded validation losses, including dim6 itself.
