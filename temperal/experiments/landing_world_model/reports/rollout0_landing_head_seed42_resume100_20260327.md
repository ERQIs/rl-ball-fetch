# Rollout0 Landing Head, Seed42, Resumed To 100 Epochs

## Goal

Continue the `rollout0` landing-head baseline from `60` to `100` epochs using resume, to see whether the landing readout was still undertrained.

## Resume Setup

- Resume config:
  [rollout0_landing_head_seed42_100ep_resume_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/rollout0_landing_head_seed42_100ep_resume_20260327.yaml)
- Script:
  [train_rollout36_landing_head.py](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/scripts/train_rollout36_landing_head.py)
- Output directory reused:
  [rollout0_landing_head_seed42_40ep_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_40ep_20260327)

## Latest Results

- Best epoch: `99`
- Best val mean L2: `0.3343`
- Val success@0.2: `0.3733`
- Test mean L2: `0.3350`
- Test success@0.2: `0.3133`

## Comparison

- `20 epoch` rollout0:
  - test mean L2: `0.5300`
  - test success@0.2: `0.1400`
- `60 epoch` rollout0:
  - test mean L2: `0.3866`
  - test success@0.2: `0.2467`
- `100 epoch` rollout0:
  - test mean L2: `0.3350`
  - test success@0.2: `0.3133`

## Current Read

- The `rollout0` landing-head line keeps improving with more optimization.
- At least for this seed42 run, the low initial success rate was substantially an undertraining issue.
- Even without latent rollout, the adapter-initialized temporal backbone contains useful information for explicit landing prediction.
