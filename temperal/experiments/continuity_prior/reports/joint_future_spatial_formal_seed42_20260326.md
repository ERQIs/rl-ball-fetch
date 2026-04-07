# Joint Future + Spatial Pretraining Pilot

Date: `2026-03-26`

## Goal

Run the first non-smoke joint pretraining pilot:

- shared encoder
- future-pretrain objective stays active
- spatial-prior regularization also stays active during temporal pretraining

This was meant to test whether simultaneous pretraining can avoid the "spatial prior gets washed out by later future pretraining" issue.

## Config

Pilot config:

- [multiscale_future_spatial_joint_formal_seed42_20260326.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_future_spatial_joint_formal_seed42_20260326.yaml)

Key settings:

- width-matched `8/8/8`
- seed `42`
- full temporal train/val split
- `20` epochs
- `lambda_spatial_total = 1.0`
- spatial pairs mode = `history`

Output:

- [multiscale_future_spatial_joint_formal_seed42_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatial_joint_formal_seed42_20260326)

Main artifacts:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatial_joint_formal_seed42_20260326/best.pt)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatial_joint_formal_seed42_20260326/history.csv)
- [best_val_metrics.json](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatial_joint_formal_seed42_20260326/best_val_metrics.json)

## Result

The run completed successfully to `20` epochs.

Best validation total loss:

- `2.6104` at epoch `14`

Best validation decomposition:

- future loss: `2.5915`
- future frame: `0.00247`
- future pos: `2.2231`
- future vel: `1.8298`
- spatial raw: `0.01884`
- spatial scaled: `0.01884`

The later epochs are basically flat rather than still improving:

- epoch `14`: `val_total = 2.6104`
- epoch `17`: `2.6112`
- epoch `18`: `2.6107`
- epoch `19`: `2.6125`
- epoch `20`: `2.6128`

So the run plateaued rather than continuing to optimize meaningfully after the first half.

## Comparison to Earlier Width-Matched Sequential Future Pretrain

Reference sequential run:

- [multiscale_future_spatialinit_888_formal_seed42_20260325](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325)
- report:
  [spatial_backbone_plus_future_pretrain_888_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_backbone_plus_future_pretrain_888_20260325.md)

Reference sequential best validation loss:

- `0.3587` at epoch `19`

Current joint pilot best validation total loss:

- `2.6104` at epoch `14`

So this first naive joint version is **much worse** than the earlier sequential spatial-init future-pretrain run.

## Interpretation

The important conclusion is not that simultaneous pretraining is impossible.

The important conclusion is:

1. the implementation is feasible and stable
2. but this first joint recipe is not yet the right one

Even though the logged spatial loss term is numerically small, keeping it active this way appears to strongly interfere with learning the future objective.

A likely explanation is that:

- the spatial branch produces dense encoder gradients
- those gradients are not well balanced against the future objective
- so the encoder is being pulled in a way that hurts predictive dynamics learning

In other words:

- sequential stacked training may wash out spatial structure
- but this first simultaneous version over-corrects and hurts future learning

## Current Takeaway

Simultaneous pretraining is still a promising idea, but this specific first recipe should **not** be treated as the new default.

At the moment:

- it is a valid implementation prototype
- it is not yet a competitive pretraining recipe

## Recommended Next Step

If we continue on the joint line, the next version should probably be gentler than this one, for example:

1. reduce the spatial gradient influence on the shared encoder
2. keep spatial regularization on only a subset of steps or layers
3. try a much smaller effective spatial weight during full-data training

So the next goal is not "run more of the same", but "rebalance the joint objective".
