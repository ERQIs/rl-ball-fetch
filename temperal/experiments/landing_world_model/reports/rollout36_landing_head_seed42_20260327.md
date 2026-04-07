# Rollout36 Landing Head, Seed42

## Goal

Test a first world-model-style downstream recipe:

- observe the first 50% of the trajectory with 8 frames
- roll the latent dynamics forward for a fixed 36 steps
- attach a small landing head on the final rollout state
- predict explicit `landing_px` and `landing_pz`

This is a supervised landing experiment under `landing_world_model`, so it is not directly comparable to the earlier `continuity_prior` downstream results that used the older transfer target convention.

## Setup

- Train split: `700` episodes
- Val split: `150` episodes
- Test split: `150` episodes
- Observation policy: first `50%`, `8` uniformly sampled visible frames
- Rollout steps: `36`
- Init checkpoint:
  [adapter best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/best.pt)
- Training config:
  [rollout36_landing_head_seed42_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/rollout36_landing_head_seed42_20260327.yaml)
- Training script:
  [train_rollout36_landing_head.py](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/scripts/train_rollout36_landing_head.py)

## Outputs

- Run directory:
  [rollout36_landing_head_seed42_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout36_landing_head_seed42_20260327)
- Best checkpoint:
  [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout36_landing_head_seed42_20260327/best.pt)
- Summary:
  [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout36_landing_head_seed42_20260327/summary.json)
- Training history:
  [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout36_landing_head_seed42_20260327/history.csv)
- Curve:
  [mean_l2_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout36_landing_head_seed42_20260327/mean_l2_curve.svg)
- Val predictions:
  [val_predictions.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout36_landing_head_seed42_20260327/val_predictions.csv)
- Test predictions:
  [test_predictions.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout36_landing_head_seed42_20260327/test_predictions.csv)

## Results

- Best epoch: `15`
- Best val mean L2: `0.5621`
- Val success@0.2: `0.1467`
- Test mean L2: `0.5753`
- Test success@0.2: `0.0533`

## Context Against Raw Rollout Probe

The earlier raw rollout-to-threshold probe was:

- [adapter_rollout_probe_20260327 summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/adapter_rollout_probe_20260327/summary.json)
- `landing_xz_l2_mean = 4.2045`

So this fixed-step rollout + landing head training reduced test landing error from roughly `4.20` to `0.58`, which is a large improvement.

## Current Read

- The route is trainable.
- Fixed-step latent rollout is usable as a world-model-style backbone for landing prediction.
- The first seed42 result is promising in absolute improvement over raw rollout probing.
- But `success@0.2` is still low, so this is not yet a strong final recipe.
- The next natural ablations are:
  - rollout steps `32 / 36 / 40`
  - whether the backbone should be partially frozen for the first few epochs
  - whether the landing head should read only the final rollout state or a small stack of late rollout states
