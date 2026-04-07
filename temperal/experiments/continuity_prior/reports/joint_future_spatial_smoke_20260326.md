# Joint Future + Spatial Pretrain Smoke

Date: `2026-03-26`

## Goal

Validate a first experiment-local implementation of simultaneous pretraining:

- keep the temporal `future pretrain` objective
- keep a shared encoder
- add a persistent spatial-prior regularization branch during temporal pretraining

This is the first smoke test for the idea that spatial structure should stay active while future dynamics are being learned.

## Implementation

New local training entry:

- [train_joint_future_spatial.py](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/scripts/train_joint_future_spatial.py)

This local script:

1. initializes the shared encoder from the compact `8/8/8` spatial-prior checkpoint
2. also initializes a spatial decoder branch from the same checkpoint
3. trains future dynamics and spatial regularization jointly

The joint total loss is:

`L_total = L_future + lambda_spatial_total * L_spatial`

For this smoke run:

- `lambda_spatial_total = 0.2`
- spatial pairs mode = `history`
- spatial decoder mode = `f3`

## Smoke Config

- [multiscale_future_spatial_joint_smoke_20260326.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_future_spatial_joint_smoke_20260326.yaml)

Smoke output:

- [multiscale_future_spatial_joint_smoke_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatial_joint_smoke_20260326)

Main checkpoint:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatial_joint_smoke_20260326/best.pt)

Training history:

- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatial_joint_smoke_20260326/history.csv)

## Result

The implementation works end-to-end:

- spatial encoder init loads cleanly
- spatial decoder init also loads cleanly
- joint training runs on GPU and saves checkpoints normally

The 3-epoch smoke history is:

| epoch | train total | val total | val future | val spatial scaled |
| --- | ---: | ---: | ---: | ---: |
| `1` | `5.2248` | `4.5509` | `4.5433` | `0.0076` |
| `2` | `4.3881` | `4.2266` | `4.2202` | `0.0064` |
| `3` | `4.1609` | `4.0668` | `4.0613` | `0.0055` |

So both train and validation total loss decrease in a healthy way over the smoke run.

## Key Observation

The most important thing learned from this smoke is not just that the code runs, but that the loss scales are very different.

- future loss is around `4.0-5.0`
- scaled spatial contribution is only around `0.005-0.008`

So with `lambda_spatial_total = 0.2`, the spatial branch is currently acting as only a very light regularizer.

## Immediate Interpretation

This smoke says:

1. simultaneous pretraining is feasible in the current codebase
2. the basic joint design is stable enough to train
3. if we want spatial prior to meaningfully shape temporal pretraining, the next real pilot should probably use a stronger spatial weight than this smoke

## Recommended Next Step

Run a short-but-more-serious joint pilot, for example:

- `10-20` epochs
- same `8/8/8` width-matched setup
- increase `lambda_spatial_total` from `0.2` to something like `1.0`

That would let us test whether keeping spatial supervision active during temporal pretraining can improve the downstream stacked transfer story.
