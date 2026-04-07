# Stacked888 Smaller-Backbone-LR Follow-up

Date: `2026-03-26`

## Goal

Test a gentler finetune recipe for the `stacked888` downstream transfer line.

Instead of freezing the pretrained backbone, use parameter groups:

- head / readout lr: `1e-3`
- backbone lr: `1e-4` (`0.1x`)

This was meant to preserve the stacked pretrained representation while still allowing end-to-end adaptation.

## Setup

Only the most informative minimal matrix was run:

- data fractions: `20%`, `100%`
- seeds: `42, 43, 44`
- epochs: `20`
- init: `stacked888` downstream finetune
- no freeze warmup
- backbone lr scale: `0.1`

Configs:

- [stacked_transfer_888_blr01_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/stacked_transfer_888_blr01_20260326)

Outputs:

- [stacked_transfer_888_blr01_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_blr01_20260326)

Summaries:

- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_blr01_20260326/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/stacked_transfer_888_blr01_20260326/detailed_results.csv)

Local experiment-only training entry:

- [train_multiscale_downstream_backbone_lr.py](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/scripts/train_multiscale_downstream_backbone_lr.py)

## Comparison Against Default Stacked888 Finetune

| data | setting | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: |
| `20%` | stacked888 default | `0.2272 +/- 0.0111` | `0.5111 +/- 0.0638` |
| `20%` | stacked888 backbone-lr `0.1x` | `0.2620 +/- 0.0252` | `0.4444 +/- 0.0607` |
| `100%` | stacked888 default | `0.1658 +/- 0.0045` | `0.6733 +/- 0.0411` |
| `100%` | stacked888 backbone-lr `0.1x` | `0.1747 +/- 0.0048` | `0.6467 +/- 0.0331` |

## Conclusion

This first smaller-backbone-lr test did **not** help.

1. At `20%`, it clearly underperforms the current default stacked888 finetune.
2. At `100%`, it is also worse than default stacked888 finetune.
3. So for the current stacked line, reducing backbone lr to `0.1x` is not a good default recipe.

## Interpretation

This looks different from the earlier intuition that the stacked backbone might need stronger protection.

The current result suggests:

- the stacked checkpoint may already need fairly active joint adaptation
- slowing the backbone too much can behave a bit like a softer freeze
- and that soft freeze is still enough to hurt transfer quality

So the main low-data positive result remains:

- `stacked888` finetune with the original single-lr end-to-end recipe
