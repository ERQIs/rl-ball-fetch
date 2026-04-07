# Downstream Transfer of 8/8/8 Stacked Pretraining

Date: `2026-03-25`

## Goal

Check whether the new width-matched stacked pretraining line gives downstream benefit:

1. `8/8/8` spatial-prior visual backbone
2. `8/8/8` temporal future pretrain initialized from that visual backbone
3. downstream finetune on the `obs50` transfer benchmark

The main question is not whether stacked temporal pretraining beat scratch pretraining loss, but whether its final downstream transfer is useful.

## Checkpoint Used

Stacked temporal-pretrain checkpoint:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325/best.pt)

Source report for that checkpoint:

- [spatial_backbone_plus_future_pretrain_888_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_backbone_plus_future_pretrain_888_20260325.md)

## Downstream Protocol

New configs for this run:

- [stacked_transfer_888_20260325](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/stacked_transfer_888_20260325)

New output root:

- [stacked_transfer_888_20260325](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325)

Shared settings:

- seeds: `42, 43, 44`
- epochs: `20`
- protocol: `obs50`
- data fractions: `20%`, `40%`, `100%`
- init mode: `finetune`
- no extra freeze warmup in this first comparison

Reused historical baselines:

- scratch aggregate outputs from `temperal/outputs/multiscale_transfer_scratch_*_obs50_3seed_20260316_220723`
- future-pretrain finetune aggregate outputs from `temperal/outputs/multiscale_transfer_finetune_*_obs50_3seed_20260316_220723`

## New Results

New stacked-run summaries:

- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325/detailed_results.csv)
- [stacked_transfer_vs_baselines_curves.png](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_20260325/stacked_transfer_vs_baselines_curves.png)

Test-set comparison:

| data | method | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: |
| `20%` | scratch | `0.2654 ± 0.0130` | `0.3844 ± 0.0257` |
| `20%` | future finetune | `0.2435 ± 0.0103` | `0.4578 ± 0.0658` |
| `20%` | stacked888 finetune | `0.2272 ± 0.0111` | `0.5111 ± 0.0638` |
| `40%` | scratch | `0.2070 ± 0.0069` | `0.5156 ± 0.0503` |
| `40%` | future finetune | `0.1789 ± 0.0103` | `0.6156 ± 0.0227` |
| `40%` | stacked888 finetune | `0.1945 ± 0.0171` | `0.5822 ± 0.0656` |
| `100%` | scratch | `0.1632 ± 0.0067` | `0.6711 ± 0.0371` |
| `100%` | future finetune | `0.1493 ± 0.0053` | `0.7200 ± 0.0303` |
| `100%` | stacked888 finetune | `0.1658 ± 0.0045` | `0.6733 ± 0.0411` |

## Main Takeaway

Yes, there is downstream benefit, but it depends on data regime.

1. At `20%`, the stacked checkpoint is the best of the three.
   It beats both scratch and the historical future-pretrain finetune baseline.

2. At `40%`, stacked transfer is still helpful relative to scratch.
   It does not beat the historical future-pretrain baseline, but it stays clearly above scratch.

3. At `100%`, stacked transfer is basically scratch-level.
   It is no longer clearly better than scratch, and it still trails the historical future-pretrain finetune baseline.

## Interpretation

This is the most interesting part of the current story:

- the width-matched stacked temporal pretrain was slightly worse than scratch in temporal-pretrain validation loss
- but that did not prevent it from helping downstream at low data

So for this experiment line, the important signal is:

- stacked pretraining may be especially useful as a low-data transfer prior
- pretrain loss ranking and downstream transfer ranking are not identical

## Recommended Next Step

The most valuable follow-up now is:

1. keep this `stacked888` checkpoint as a serious downstream candidate
2. if we want to improve mid/high-data performance, test the same downstream protocol with a small warmup recipe
3. compare `stacked888` directly against the earlier `16/32/64` stacked line if we want to know whether compactness is helping transfer
