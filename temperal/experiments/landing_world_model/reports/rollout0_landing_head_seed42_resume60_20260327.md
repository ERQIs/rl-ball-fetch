# Rollout0 Landing Head, Seed42, Resumed To 60 Epochs

## Goal

Continue the stronger `rollout0` landing-head run from `40` to `60` epochs without restarting from scratch.

## Resume Setup

- Training script now supports `--resume`:
  [train_rollout36_landing_head.py](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/scripts/train_rollout36_landing_head.py)
- Resume config:
  [rollout0_landing_head_seed42_60ep_resume_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/rollout0_landing_head_seed42_60ep_resume_20260327.yaml)
- Resume source checkpoint:
  [last.pt](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/last.pt)

Note:

- the resumed run continued inside the same output directory
- so [rollout0_landing_head_seed42_40ep_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327) now contains the `60 epoch` state, not the old `40 epoch` endpoint

## Latest Results

- Best epoch: `55`
- Best val mean L2: `0.3633`
- Val success@0.2: `0.3133`
- Test mean L2: `0.3866`
- Test success@0.2: `0.2467`

Current artifacts:

- [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/summary.json)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/history.csv)
- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/best.pt)
- [mean_l2_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327/mean_l2_curve.svg)

## Comparison

- `20 epoch` rollout0:
  - test mean L2: `0.5300`
  - test success@0.2: `0.1400`
- `40 epoch` rollout0:
  - test mean L2: `0.4820`
  - test success@0.2: `0.1933`
- `60 epoch` rollout0:
  - test mean L2: `0.3866`
  - test success@0.2: `0.2467`

## Current Read

- The `rollout0` landing-head baseline was clearly undertrained at both `20` and `40` epochs.
- Continued training helped substantially.
- This strengthens the conclusion that the low initial success rate was not only about task difficulty; optimization time also mattered a lot.
