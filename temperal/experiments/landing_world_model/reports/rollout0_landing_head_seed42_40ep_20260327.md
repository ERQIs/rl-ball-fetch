# Rollout0 Landing Head, Seed42, 40 Epochs

## Goal

Continue the stronger `rollout0` landing-head baseline from `20` to `40` epochs and check whether the low success rate mainly came from undertraining.

## Setup

- Same target as before: explicit `landing_px / landing_pz`
- Same observation setting: first `50%`, `8` uniformly sampled visible frames
- Same initialization:
  [adapter best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/best.pt)
- Only major change: `epochs = 40`
- Config:
  [rollout0_landing_head_seed42_40ep_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/rollout0_landing_head_seed42_40ep_20260327.yaml)

## Outputs

- Run directory:
  [rollout0_landing_head_seed42_40ep_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327)
- Best checkpoint:
  [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/best.pt)
- Summary:
  [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/summary.json)
- History:
  [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/history.csv)
- Curve:
  [mean_l2_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/mean_l2_curve.svg)
- Test predictions:
  [test_predictions.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/test_predictions.csv)

## Results

- Best epoch: `40`
- Best val mean L2: `0.4078`
- Val success@0.2: `0.3067`
- Test mean L2: `0.4820`
- Test success@0.2: `0.1933`

## Comparison To 20 Epochs

- `20 epoch` rollout0:
  - test mean L2: `0.5300`
  - test success@0.2: `0.1400`
- `40 epoch` rollout0:
  - test mean L2: `0.4820`
  - test success@0.2: `0.1933`

## Current Read

- The `rollout0` line was undertrained at `20` epochs.
- Longer training helped both error and success noticeably.
- Even after that improvement, this explicit landing task is still much harder than the earlier proxy downstream task.
