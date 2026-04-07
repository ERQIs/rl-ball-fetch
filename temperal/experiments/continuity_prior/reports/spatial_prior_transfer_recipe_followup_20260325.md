# Spatial Prior Transfer Recipe Follow-up

Date: `2026-03-25`

## Problem

After the initial `scratch vs spatial_finetune` transfer experiments, the main issue was no longer whether the spatial-prior backbone had any task-relevant signal.

The real problem became:

- `spatial_finetune` was much more seed-sensitive than `scratch`
- some runs, especially `100% seed43`, showed a very long early plateau
- the backbone seemed transferable, but the finetuning recipe was not yet stable enough

This follow-up report focuses on that transfer-recipe question.

## Question

Can a short frozen-backbone warmup reduce the slow-start / plateau behavior and improve downstream transfer?

The transfer recipe variants compared here are:

- no warmup: `freeze_backbone_epochs = 0`
- short warmup: `freeze_backbone_epochs = 5`
- longer warmup: `freeze_backbone_epochs = 10`

## Shared Setup

All runs in this follow-up keep the same downstream task and architecture:

- visual init: pretrained spatial-prior backbone
- init mode: `finetune`
- encoder widths: `c1=16, c2=32, c3=64`
- temporal state widths: `s1=8, s2=8, s3=8`
- observation length: `8`
- frame stride: `2`
- observation end fraction: `0.5`
- sampling mode: `uniform_visible`
- epochs: `40`
- batch size: `8`
- lr: `1e-3`
- pre-head layernorm: `false`
- seeds: `41, 42, 43`
- data fractions: `20%`, `100%`

Backbone checkpoint used:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/best.pt)

## Output Groups

No-warmup baseline:

- [spatial_prior_transfer_pilot_40ep_20260323](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_40ep_20260323)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_40ep_20260323/summary.csv)

Warmup5 follow-up:

- [spatial_prior_transfer_warmup5_40ep_20260324](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324/detailed_results.csv)

Warmup10 follow-up:

- [spatial_prior_transfer_warmup10_40ep_20260324](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324/detailed_results.csv)

## Aggregate Results

### 20% Data

| recipe | val mean_l2 | val success@0.2 | test mean_l2 | test success@0.2 |
| --- | ---: | ---: | ---: | ---: |
| `scratch` | `0.2179 +- 0.0028` | `0.5000 +- 0.0327` | `0.2042 +- 0.0073` | `0.5311 +- 0.0412` |
| `spatial_finetune` | `0.2736 +- 0.0872` | `0.4600 +- 0.1085` | `0.2817 +- 0.1049` | `0.4578 +- 0.1307` |
| `spatial_finetune + warmup5` | `0.2101 +- 0.0165` | `0.5467 +- 0.0432` | `0.1958 +- 0.0076` | `0.5311 +- 0.0350` |
| `spatial_finetune + warmup10` | `0.2070 +- 0.0128` | `0.5778 +- 0.0409` | `0.1992 +- 0.0133` | `0.5667 +- 0.0163` |

Readout:

- no-warmup finetune is clearly worse than `scratch`
- both warmup variants remove that negative transfer
- `warmup5` gives the best `test mean_l2`
- `warmup10` gives the best `success@0.2`
- both warmup recipes are much more stable than no warmup

### 100% Data

| recipe | val mean_l2 | val success@0.2 | test mean_l2 | test success@0.2 |
| --- | ---: | ---: | ---: | ---: |
| `scratch` | `0.1483 +- 0.0029` | `0.7467 +- 0.0189` | `0.1439 +- 0.0011` | `0.7489 +- 0.0463` |
| `spatial_finetune` | `0.1526 +- 0.0151` | `0.7222 +- 0.0545` | `0.1470 +- 0.0193` | `0.7533 +- 0.0824` |
| `spatial_finetune + warmup5` | `0.1411 +- 0.0084` | `0.7689 +- 0.0354` | `0.1412 +- 0.0073` | `0.7689 +- 0.0600` |
| `spatial_finetune + warmup10` | `0.1510 +- 0.0029` | `0.7333 +- 0.0054` | `0.1443 +- 0.0031` | `0.7533 +- 0.0109` |

Readout:

- no-warmup finetune is roughly tied to `scratch`, but seed variance is larger
- `warmup5` is the strongest overall recipe
- `warmup10` is still better than no warmup on `test mean_l2`, but loses most of the `warmup5` gain
- this suggests some warmup is helpful, but `10` frozen epochs is already a bit too conservative in the high-data regime

## Seed43 Case

The motivating case was `100% seed43`, which previously had a very long plateau.

Useful reference files:

- [no-warmup history](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_40ep_20260323/multiscale_transfer_spatial_finetune_100pct_obs50_seed43_20260323_40ep/history.csv)
- [warmup5 history](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324/multiscale_transfer_spatial_finetune_100pct_obs50_seed43_20260324_warmup5_40ep/history.csv)
- [warmup10 history](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324/multiscale_transfer_spatial_finetune_100pct_obs50_seed43_20260324_warmup10_40ep/history.csv)

Key timing comparison:

- no warmup:
  - first `val_loss < 0.02` at epoch `33`
  - first `val_loss < 0.01` at epoch `38`
- `warmup5`:
  - first `val_loss < 0.02` at epoch `9`
  - first `val_loss < 0.01` at epoch `14`
- `warmup10`:
  - first `val_loss < 0.02` at epoch `17`
  - first `val_loss < 0.01` at epoch `21`

Interpretation:

- the plateau was not evidence that the pretrained backbone was fundamentally incompatible with the task
- it behaved much more like an early alignment problem between the pretrained visual backbone and the random temporal/head layers
- `warmup5` is enough to break that bad early dynamic
- `warmup10` still helps, but delays joint adaptation more than needed

## Figures

Seed-wise curve comparison with `scratch`, no-warmup finetune, `warmup5`, and `warmup10`:

- [100% by-seed curves](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324/transfer_100pct_40ep_curves_by_seed_with_warmup5_warmup10.png)
- [20% by-seed curves](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324/transfer_20pct_40ep_curves_by_seed_with_warmup5_warmup10.png)

Earlier comparison figure before adding `warmup10`:

- [100% by-seed curves with warmup5](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324/transfer_100pct_40ep_curves_by_seed_with_warmup5.png)

## Conclusion

The transfer problem was not “the pretrained backbone is bad”.

The stronger reading is:

- the spatial-prior backbone is transferable
- the transfer recipe is highly optimization-sensitive
- a short frozen-backbone warmup fixes most of the pathological slow-start behavior
- among the tested settings, `freeze_backbone_epochs = 5` is the best default so far

Current practical recommendation:

- use `spatial_finetune + warmup5` as the default transfer recipe
- treat `warmup10` as evidence that warmup is useful, but that too much freezing can start to slow later adaptation
