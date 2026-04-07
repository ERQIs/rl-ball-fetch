# Downstream Transfer of Adapter-Based Temporal Pretraining

Date: `2026-03-27`

## Goal

Test whether the new temporal-pretrain checkpoint from the shared-decoder adapter line gives downstream benefit on the same `obs50` transfer benchmark used for:

- scratch
- historical future-pretrain finetune
- earlier `stacked888` finetune

The main question is:

- does the better adapter-based temporal-pretrain bias translate into better downstream transfer?

## Checkpoint Used

Adapter temporal-pretrain checkpoint:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/best.pt)

Source report:

- [future_decoder_adapter_unfreeze_formal_seed42_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/future_decoder_adapter_unfreeze_formal_seed42_20260326.md)

## Downstream Protocol

New configs:

- [adapter_transfer_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/adapter_transfer_20260327)

New output root:

- [adapter_transfer_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/adapter_transfer_20260327)

Shared settings:

- seeds: `42, 43, 44`
- epochs: `20`
- protocol: `obs50`
- data fractions: `20%`, `40%`, `100%`
- init mode: `finetune`
- no warmup

Main artifacts:

- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/adapter_transfer_20260327/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/adapter_transfer_20260327/detailed_results.csv)
- [adapter_transfer_vs_baselines_curves.png](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/adapter_transfer_20260327/adapter_transfer_vs_baselines_curves.png)

Reference baselines:

- historical scratch / future summary:
  [summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_obs50_data_efficiency_3seed_20260316_220723/summary.csv)
- earlier stacked888 summary:
  [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325/summary.csv)

## Test Results

| data | method | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: |
| `20%` | scratch | `0.2654 +/- 0.0130` | `0.3844 +/- 0.0257` |
| `20%` | future finetune | `0.2435 +/- 0.0103` | `0.4578 +/- 0.0658` |
| `20%` | stacked888 finetune | `0.2272 +/- 0.0111` | `0.5111 +/- 0.0638` |
| `20%` | adapter finetune | `0.2290 +/- 0.0058` | `0.4956 +/- 0.0166` |
| `40%` | scratch | `0.2070 +/- 0.0069` | `0.5156 +/- 0.0503` |
| `40%` | future finetune | `0.1789 +/- 0.0103` | `0.6156 +/- 0.0227` |
| `40%` | stacked888 finetune | `0.1945 +/- 0.0171` | `0.5822 +/- 0.0656` |
| `40%` | adapter finetune | `0.1737 +/- 0.0008` | `0.6289 +/- 0.0031` |
| `100%` | scratch | `0.1632 +/- 0.0067` | `0.6711 +/- 0.0371` |
| `100%` | future finetune | `0.1493 +/- 0.0053` | `0.7200 +/- 0.0303` |
| `100%` | stacked888 finetune | `0.1658 +/- 0.0045` | `0.6733 +/- 0.0411` |
| `100%` | adapter finetune | `0.1479 +/- 0.0086` | `0.7400 +/- 0.0237` |

## Main Takeaway

This adapter-pretrained downstream line is clearly useful.

The regime story is:

1. at `20%`, adapter transfer is strong but not the best
- it is better than scratch and historical future finetune
- but it still trails the earlier `stacked888` low-data winner by a small margin

2. at `40%`, adapter transfer is the new best setting
- it beats scratch
- it beats historical future finetune
- it also beats stacked888

3. at `100%`, adapter transfer is also the new best setting
- it slightly beats historical future finetune
- and clearly beats scratch and stacked888

## Interpretation

This result is important because it lines up with the temporal-pretrain finding:

- the adapter-based temporal-pretrain was only slightly below the historical scratch future-pretrain in validation loss
- but its downstream transfer bias turns out to be very competitive

So the current picture is:

- `stacked888` is still the strongest low-data specialist at `20%`
- `adapter finetune` is now the best general-purpose pretrained downstream recipe at `40%` and `100%`

## Current Recommendation

If we continue from here, the most sensible default candidates are:

1. keep `stacked888` as the main low-data (`20%`) downstream prior
2. promote `adapter finetune` as the main mid/high-data downstream candidate
3. if we want one single pretrained recipe to push further, `adapter` is now the best one to prioritize
