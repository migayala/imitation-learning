# Stacked4 + Standardized Run — 2026-04-03

## Setup
- Train command: `./.venv/bin/python scripts/train.py --config configs/train_stacked4_standardized.yaml`
- Eval command: `./.venv/bin/python scripts/evaluate.py --config configs/train_stacked4_standardized.yaml --checkpoint models_stacked4_standardized/best.pt --episodes 50`
- Dataset: `data/image.hdf5`
- Device: `cuda`
- Action standardization: enabled (`models_stacked4_standardized/action_stats.json`)
- Temporal context: `data.frame_stack = 4` (zero-padded at episode start)

## Checkpoint And Config Identity
- Best checkpoint: `models_stacked4_standardized/best.pt`
- Best epoch: `84`
- Best checkpoint SHA-256: `8f7789eb8b03e0a446eb52e69babdc49eeecb37cbdce21dd1afe35c377cee1c9`
- Config SHA-256: `49f055228fcdfd2ee8cbebafeca44f0d2e20941b1e9b863e3e1d1604a9432ca2`

## Training Metrics (Best Epoch = 84)
- `Loss/train`: `0.00840863399207592`
- `Loss/val`: `0.0957273542881012`
- `LossPerDim/val_mse_dim0`: `0.008392213843762875`
- `LossPerDim/val_mse_dim1`: `0.0036350064910948277`
- `LossPerDim/val_mse_dim2`: `0.009991024620831013`
- `LossPerDim/val_mse_dim3`: `5.1410144806141034e-05`
- `LossPerDim/val_mse_dim4`: `0.00011645510676316917`
- `LossPerDim/val_mse_dim5`: `0.0005136246327310801`
- `LossPerDim/val_mse_dim6`: `0.06861992925405502`

## Evaluation Result
- Episodes: `50`
- Successes: `0`
- Success rate: `0.0%`
- Partial lift (>= `0.030 m`): `0.0%`
- Max lift delta mean / median: `0.0002 m / 0.0000 m`
- Min gripper-to-cube distance mean / median: `0.0263 m / 0.0217 m`

## A/B Comparison Notes
- Vs baseline (`evaluation/baseline_2026-04-01.md`):
- `LossPerDim/val_mse_dim6`: `0.07262297719717026 -> 0.06861992925405502` (better, -0.00400304794311524)
- Success rate: `0.0% -> 0.0%` (no change)
- Vs standardized single-frame (`evaluation/standardized_baseline_2026-04-02.md`):
- `Loss/val`: `0.12622052431106567 -> 0.0957273542881012` (better)
- `LossPerDim/val_mse_dim6`: `0.0781158059835434 -> 0.06861992925405502` (better, -0.00949587672948838)
- Success rate: `0.0% -> 0.0%` (no change)
- Vs dim6-weighted (`evaluation/dim6_weighted_2026-04-02.md`):
- `LossPerDim/val_mse_dim6`: `0.09631475806236267 -> 0.06861992925405502` (better, -0.02769482880830765)

## Conclusion
- Frame stacking (`K=4`) improved validation error broadly, including dim6, but did not move task success off zero.
- Progress metrics show the agent generally reaches near-cube proximity but does not achieve meaningful lift, indicating rollout-level behavior remains brittle despite lower supervised error.
