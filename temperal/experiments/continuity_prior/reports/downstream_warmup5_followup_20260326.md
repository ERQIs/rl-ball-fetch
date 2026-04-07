# Downstream Warmup5 Follow-up

Date: `2026-03-26`

## Goal

Check whether adding a short backbone-freeze warmup improves downstream finetuning for pretrained temporal backbones.

This follow-up covers both pretrained lines:

1. historical `future` pretrain -> downstream finetune
2. new `stacked888` pretrain -> downstream finetune

The user specifically asked to include the `20%` data regime as well, so this report covers `20% / 40% / 100%`.

## Protocol

Shared settings:

- seeds: `42, 43, 44`
- epochs: `20`
- downstream protocol: `obs50`
- warmup recipe: `freeze_backbone_epochs = 5`

Warmup5 outputs:

- future warmup5:
  [future_transfer_warmup5_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/future_transfer_warmup5_20260326)
- stacked888 warmup5:
  [stacked_transfer_888_warmup5_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_warmup5_20260326)

Warmup5 summaries:

- [future warmup5 summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/future_transfer_warmup5_20260326/summary.csv)
- [stacked888 warmup5 summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_warmup5_20260326/summary.csv)

Reference no-warmup reports:

- [stacked_transfer_888_downstream_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/stacked_transfer_888_downstream_20260325.md)
- historical future finetune aggregate summary lives in:
  [summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_obs50_data_efficiency_3seed_20260316_220723/summary.csv)

## Test Results

Historical `future` line:

| data | setting | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: |
| `20%` | future finetune | `0.2435 +/- 0.0103` | `0.4578 +/- 0.0658` |
| `20%` | future warmup5 | `0.2394 +/- 0.0118` | `0.5089 +/- 0.0739` |
| `40%` | future finetune | `0.1789 +/- 0.0103` | `0.6156 +/- 0.0227` |
| `40%` | future warmup5 | `0.1844 +/- 0.0096` | `0.5911 +/- 0.0269` |
| `100%` | future finetune | `0.1493 +/- 0.0053` | `0.7200 +/- 0.0303` |
| `100%` | future warmup5 | `0.1604 +/- 0.0034` | `0.6622 +/- 0.0354` |

`stacked888` line:

| data | setting | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: |
| `20%` | stacked888 finetune | `0.2272 +/- 0.0111` | `0.5111 +/- 0.0638` |
| `20%` | stacked888 warmup5 | `0.2619 +/- 0.0308` | `0.4311 +/- 0.0918` |
| `40%` | stacked888 finetune | `0.1945 +/- 0.0171` | `0.5822 +/- 0.0656` |
| `40%` | stacked888 warmup5 | `0.1873 +/- 0.0097` | `0.5822 +/- 0.0710` |
| `100%` | stacked888 finetune | `0.1658 +/- 0.0045` | `0.6733 +/- 0.0411` |
| `100%` | stacked888 warmup5 | `0.1828 +/- 0.0040` | `0.5622 +/- 0.0175` |

## Main Observations

1. `20%` is now covered, and the result is not symmetric across the two pretrained lines.
   `future warmup5` improves over no-warmup future finetune at low data, but `stacked888 warmup5` gets clearly worse than no-warmup stacked888.

2. The best low-data result still comes from `stacked888` without warmup.
   At `20%`, `stacked888 finetune` remains the strongest setting in this follow-up:
   `0.2272 / 0.5111`.

3. Warmup5 does not look like a good default for the current `stacked888` recipe.
   It hurts `20%` substantially, leaves `40%` roughly flat, and hurts `100%` clearly.

4. Warmup5 is only conditionally useful for the historical `future` line.
   It helps at `20%`, but it does not improve the `40%` and `100%` regimes.

## Interpretation

The current evidence suggests that the two pretrained lines have different finetune dynamics.

- The historical `future` backbone sometimes benefits from a short readout-alignment phase in low data.
- The `stacked888` checkpoint already seems more task-ready at low data, so freezing it for `5` epochs may be over-constraining the useful adaptation path.

This matches the earlier observation that stacked pretraining was most promising as a low-data prior, not necessarily as a high-data recipe.

## Current Recommendation

For the next round, the most sensible default is:

1. keep `stacked888` downstream finetune **without** warmup as the main low-data candidate
2. keep `future warmup5` only as a side comparison if we specifically care about rescuing the historical future line at `20%`
3. do not adopt `warmup5` as a universal default across all pretrained backbones
