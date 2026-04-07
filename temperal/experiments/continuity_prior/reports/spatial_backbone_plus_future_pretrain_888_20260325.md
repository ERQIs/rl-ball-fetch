# 8/8/8 Spatial Backbone + 8/8/8 Temporal Pretrain

Date: `2026-03-25`

## Goal

Run a width-matched stacked pretraining experiment:

1. retrain a compact spatial-prior visual backbone with `c1/c2/c3 = 8/8/8`
2. use that checkpoint to initialize the `8/8/8` multiscale future-dynamics encoder
3. compare the resulting temporal pretrain against the historical `8/8/8` scratch future-pretrain baseline

This removes the main width-mismatch caveat from the earlier `16/32/64` stacked run.

## Visual Backbone Run

Config:

- [multiscale_spatial_prior_train_888_15ep_20260325.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_spatial_prior_train_888_15ep_20260325.yaml)

Output:

- [multiscale_spatial_prior_train_888_15ep_20260325](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_spatial_prior_train_888_15ep_20260325)

Best validation loss:

- `val_total = 0.0293` at epoch `15`

Checkpoint used to initialize temporal pretraining:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_spatial_prior_train_888_15ep_20260325/best.pt)

## Temporal Pretrain Run

Spatial-init config:

- [multiscale_future_spatialinit_888_formal_seed42_20260325.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_future_spatialinit_888_formal_seed42_20260325.yaml)

Spatial-init output:

- [multiscale_future_spatialinit_888_formal_seed42_20260325](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325)

Historical scratch baseline:

- config: [multiscale_future_formal_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_future_formal_seed42_20260316_220723.yaml)
- output: [multiscale_future_formal_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723)

## Result

Best validation loss comparison:

| run | encoder init | width | best val_loss | best epoch |
| --- | --- | --- | ---: | ---: |
| scratch future pretrain | none | `8/8/8` | `0.3385` | `19` |
| spatial-init future pretrain | spatial-prior `8/8/8` | `8/8/8` | `0.3587` | `19` |

Curve comparison:

- [scratch_vs_spatialinit_888_curve.png](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/scratch_vs_spatialinit_888_curve.png)

Spatial-init temporal-pretrain artifacts:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/best.pt)
- [last.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/last.pt)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/history.csv)
- [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/loss_curve.svg)
- [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/resolved_config.yaml)

## Observation

1. The width-matched stacked pipeline is fully functional.
   The `8/8/8` spatial-prior encoder loads cleanly into the future-pretrain model with no missing or unexpected keys.

2. The stacked run learns a competitive solution, but does not beat the scratch baseline.
   It improves steadily to `0.3587`, but the historical scratch baseline reaches `0.3385`.

3. The gap is not huge, but it is consistent enough to treat as a negative result for this first matched test.
   At epoch `19`, the stacked run is about `0.0202` worse in validation loss.

4. The best checkpoint is still the temporal-pretrain `best.pt`, not the last epoch.
   Epoch `20` drifts back upward to `0.3718`.

## Interpretation

This matched run is useful because it separates two questions:

- "can the spatial-prior backbone be stacked with temporal pretraining?"  
  Yes.

- "does this compact spatial-prior init improve `8/8/8` temporal pretraining over scratch?"  
  Not in this first matched run.

The current reading is that the `8/8/8` spatial visual pretrain does not yet provide a better starting point for temporal future pretraining than scratch.

## Practical Next Step

The most direct next move is still worthwhile:

1. take the current temporal-pretrain checkpoint [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/best.pt)
2. run the downstream transfer protocol
3. compare against scratch and the historical scratch-future-pretrain downstream runs

Even though temporal pretrain itself is slightly worse, the downstream task can still benefit from a different bias profile, so it is still worth checking.
