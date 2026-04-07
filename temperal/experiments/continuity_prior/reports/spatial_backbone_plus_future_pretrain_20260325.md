# Spatial Backbone + Temporal Pretrain Follow-up

Date: `2026-03-25`

## Goal

Test the next stage of the `continuity_prior` idea:

1. start from the trained spatial-prior visual backbone
2. use it to initialize the visual encoder of the multiscale future-dynamics model
3. run temporal pretraining on top
4. compare this stacked pretraining setup against the earlier temporal-pretrain-from-scratch reference in `multiscale_future`

This is a temporal-pretraining comparison, not yet the final downstream transfer comparison.

## Runs

Spatial backbone used for encoder init:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/best.pt)

Experiment-local temporal-pretrain script:

- [train_multiscale_future_spatial_init.py](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/scripts/train_multiscale_future_spatial_init.py)

Formal config for spatial-init temporal pretraining:

- [multiscale_future_spatialinit_formal_seed42_20260325.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_future_spatialinit_formal_seed42_20260325.yaml)

Spatial-init temporal-pretrain output:

- [multiscale_future_spatialinit_formal_seed42_20260325](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325)

Historical scratch temporal-pretrain reference:

- config: [multiscale_future_formal_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_future_formal_seed42_20260316_220723.yaml)
- output: [multiscale_future_formal_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723)

## Important Caveat

This first comparison is informative, but not perfectly matched.

- historical scratch future pretrain used `c1/c2/c3 = 8/8/8`
- spatial-init future pretrain uses `c1/c2/c3 = 16/32/64`

So the result below should be read as:

- "can stacked pretraining run and reach a competitive temporal-pretrain solution?"

not yet as:

- "does spatial encoder init beat a width-matched scratch temporal pretrain?"

## Result

Best validation loss comparison:

| run | encoder init | encoder width | best val_loss | best epoch |
| --- | --- | --- | ---: | ---: |
| scratch future pretrain | none | `8/8/8` | `0.3385` | `19` |
| spatial-init future pretrain | spatial-prior backbone | `16/32/64` | `0.3319` | `19` |

Spatial-init artifacts:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/best.pt)
- [last.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/last.pt)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/history.csv)
- [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/loss_curve.svg)
- [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/resolved_config.yaml)

Curve comparison figure:

- [scratch_vs_spatialinit_curve.png](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/scratch_vs_spatialinit_curve.png)

## Observations

1. The stacked setup is trainable.
   The spatial-prior encoder can be loaded cleanly into the temporal-pretrain model and optimized end-to-end without interface issues.

2. Early training is slower than the old scratch temporal-pretrain reference.
   The spatial-init run stays near `val_loss ~ 2.59` through epochs `3-6`, then starts dropping hard from epoch `7` onward.

3. By the end of `20` epochs, the spatial-init run is competitive.
   Its best validation loss is slightly lower than the old scratch reference, although this is still a width-mismatched comparison.

4. The best spatial-init checkpoint is reached at `epoch 19`.
   `epoch 20` drifts a bit upward to `0.3509`, so the saved [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/best.pt) should be the checkpoint used for any downstream follow-up.

## Interpretation

This is encouraging, but still preliminary.

- The positive part:
  stacked pretraining is feasible, and the temporal-pretrain loss does not collapse when starting from the spatial-prior encoder.

- The caution:
  we still do not know whether the gain comes from the spatial prior itself or from the larger encoder width.

## Recommended Next Step

Before claiming a clean temporal-pretrain gain from the spatial prior, the most important control is:

1. run a width-matched scratch temporal pretrain with `c1/c2/c3 = 16/32/64`
2. compare it against the current spatial-init temporal pretrain
3. then carry both checkpoints into the same downstream transfer protocol

If we want to move faster pragmatically, we can already do a first downstream transfer test using:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/best.pt)

but the cleanest story will still need the matched scratch control.
