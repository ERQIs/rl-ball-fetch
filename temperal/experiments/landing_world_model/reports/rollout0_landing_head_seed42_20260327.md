# Rollout0 Landing Head, Seed42

## Goal

Build the cleanest landing-world-model control:

- observe the first 50% of the trajectory with 8 frames
- do **not** latent-rollout beyond the observed state
- attach a landing head directly on the post-observation latent state
- predict explicit `landing_px` and `landing_pz`

This is the direct control for the earlier rollout36 experiment.

## Setup

- Same dataset split and target as rollout36
- Same adapter checkpoint initialization:
  [adapter best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/best.pt)
- Only changed factor: `rollout_steps = 0`
- Config:
  [rollout0_landing_head_seed42_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/rollout0_landing_head_seed42_20260327.yaml)
- Script:
  [train_rollout36_landing_head.py](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/scripts/train_rollout36_landing_head.py)

## Outputs

- Run directory:
  [rollout0_landing_head_seed42_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_20260327)
- Best checkpoint:
  [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_20260327/best.pt)
- Summary:
  [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_20260327/summary.json)
- Training history:
  [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_20260327/history.csv)
- Curve:
  [mean_l2_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_20260327/mean_l2_curve.svg)
- Test predictions:
  [test_predictions.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/rollout0_landing_head_seed42_20260327/test_predictions.csv)

## Results

- Best epoch: `19`
- Best val mean L2: `0.5666`
- Val success@0.2: `0.1533`
- Test mean L2: `0.5300`
- Test success@0.2: `0.1400`

## Comparison To Rollout36

- rollout0:
  - test mean L2: `0.5300`
  - test success@0.2: `0.1400`
- rollout36:
  - test mean L2: `0.5753`
  - test success@0.2: `0.0533`

So in this seed42 control, removing latent rollout improved both landing error and success rate.

## Current Read

- The landing target itself is hard, but latent rollout makes it harder still.
- The current adapter backbone already contains useful landing-related information after the observation window.
- For this first world-model branch, fixed long rollout is not yet helping the final landing readout.
