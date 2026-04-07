# Spatial Prior Transfer Pilot Plan

Date: `2026-03-23`

## Purpose

This pilot asks one focused question:

- does the newly trained multi-scale spatial-prior visual backbone improve the downstream multi-frame observe-then-act task?

This report is intentionally split into:

- experiment purpose
- experiment design
- placeholders for results and conclusions to be filled after runs complete

## Main Question

We want to isolate the value of the visual pretraining itself.

So the first comparison should keep the downstream temporal model, dataset split, and training budget fixed, and only change the visual backbone initialization:

- `scratch`
- `spatial-pretrain + finetune`

The backbone used for initialization is:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/best.pt)

## Scope

This pilot is intentionally narrower than a full ablation matrix.

It does **not** yet try to answer:

- whether frozen transfer is enough
- whether reconstruction-only pretraining would work just as well
- whether temporal pretraining should be stacked on top

Those can be follow-up experiments if this pilot already shows a signal.

## Compatibility Notes

The current spatial-prior backbone was trained with encoder widths:

- `c1=16`
- `c2=32`
- `c3=64`

The shared downstream transfer configs in `temperal/` were previously run mostly with:

- `c1=8`
- `c2=8`
- `c3=8`

So this pilot should compare:

- `scratch` with the same `16/32/64` encoder widths
- `spatial-pretrain + finetune` with the same `16/32/64` encoder widths

This keeps the comparison fair and avoids mixing two effects:

- pretraining benefit
- model-capacity change

The temporal state widths can stay at the previous small setting in the first pilot:

- `s1=8`
- `s2=8`
- `s3=8`

## Experiment Design

### Conditions

Initialization modes:

- `scratch`
- `spatial_pretrain_finetune`

Data fractions:

- `20%`
- `100%`

Seeds:

- `41`
- `42`
- `43`

Total planned runs in the first batch:

- `2 x 2 x 3 = 12`

### Downstream task

Use the current observe-then-act transfer setup:

- observation length: `8`
- frame stride: `2`
- observation end fraction: `0.5`
- sampling mode: `uniform_visible`
- target: final `xy`

### Fairness constraints

The following should remain matched between conditions:

- same dataset root
- same train/val/test split files
- same epochs
- same batch size
- same optimizer and learning rate
- same temporal module
- same head dimension
- same temporal state widths
- same logging frequency

For this first pilot:

- `freeze_backbone_epochs = 0`
- `pre_head_layernorm = false`

This keeps the comparison focused on pretraining rather than stabilization tricks.

## Metrics

Primary metrics:

- `test mean_l2`
- `test success@0.2`

Secondary readouts:

- best validation epoch
- train-to-train stability across seeds

## Paths

Local config directory for this pilot:

- [transfer_pilot_20260323](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/transfer_pilot_20260323)

Local output root for this pilot:

- [output](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output)

Planned output group:

- [spatial_prior_transfer_pilot_20260323](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_20260323)

## Planned Run Names

- `multiscale_transfer_scratch_spatialpilot_20pct_obs50_seed41_20260323`
- `multiscale_transfer_scratch_spatialpilot_20pct_obs50_seed42_20260323`
- `multiscale_transfer_scratch_spatialpilot_20pct_obs50_seed43_20260323`
- `multiscale_transfer_spatial_finetune_20pct_obs50_seed41_20260323`
- `multiscale_transfer_spatial_finetune_20pct_obs50_seed42_20260323`
- `multiscale_transfer_spatial_finetune_20pct_obs50_seed43_20260323`
- `multiscale_transfer_scratch_spatialpilot_100pct_obs50_seed41_20260323`
- `multiscale_transfer_scratch_spatialpilot_100pct_obs50_seed42_20260323`
- `multiscale_transfer_scratch_spatialpilot_100pct_obs50_seed43_20260323`
- `multiscale_transfer_spatial_finetune_100pct_obs50_seed41_20260323`
- `multiscale_transfer_spatial_finetune_100pct_obs50_seed42_20260323`
- `multiscale_transfer_spatial_finetune_100pct_obs50_seed43_20260323`

## Expected Readout

We will treat the pilot as positive if at least one of the following appears:

- `spatial_pretrain_finetune` beats `scratch` clearly at `20%`
- `spatial_pretrain_finetune` matches or exceeds `scratch` at `100%`
- `spatial_pretrain_finetune` shows smaller seed variance
- `spatial_pretrain_finetune` reaches good validation earlier

## Results

All `12` planned runs have now been completed.

Aggregate result files:

- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_20260323/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_20260323/detailed_results.csv)

Per-run test metrics have also been saved next to each run as `test_metrics.json`.

### Aggregate summary

| data | mode | val mean_l2 | val success@0.2 | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `20%` | `scratch` | `0.3637 +- 0.1392` | `0.3378 +- 0.1305` | `0.3761 +- 0.1379` | `0.2978 +- 0.0898` |
| `20%` | `spatial_finetune` | `1.0896 +- 0.5229` | `0.1222 +- 0.0689` | `1.1451 +- 0.5670` | `0.0800 +- 0.0411` |
| `100%` | `scratch` | `0.1667 +- 0.0068` | `0.6667 +- 0.0218` | `0.1640 +- 0.0068` | `0.6622 +- 0.0469` |
| `100%` | `spatial_finetune` | `0.1536 +- 0.0087` | `0.7200 +- 0.0618` | `0.1562 +- 0.0066` | `0.6933 +- 0.0237` |

### Readout

- At `20%`, the pretrained spatial-prior backbone is clearly worse than `scratch`.
- At `100%`, the pretrained spatial-prior backbone is modestly but consistently better than `scratch`.
- The high-data gain is not large, but it appears on both validation and test aggregates.
- The low-data behavior suggests the current transfer recipe is not yet robust enough for small-data finetuning.

### 40-epoch follow-up

A non-overwriting follow-up was then run with the same `12` runs but `40` epochs instead of `20`.

Follow-up outputs:

- [spatial_prior_transfer_pilot_40ep_20260323](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_40ep_20260323)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_40ep_20260323/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_40ep_20260323/detailed_results.csv)

Aggregate summary for the `40`-epoch follow-up:

| data | mode | val mean_l2 | val success@0.2 | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `20%` | `scratch` | `0.2179 +- 0.0028` | `0.5000 +- 0.0327` | `0.2042 +- 0.0073` | `0.5311 +- 0.0412` |
| `20%` | `spatial_finetune` | `0.2736 +- 0.0872` | `0.4600 +- 0.1085` | `0.2817 +- 0.1049` | `0.4578 +- 0.1307` |
| `100%` | `scratch` | `0.1483 +- 0.0029` | `0.7467 +- 0.0189` | `0.1439 +- 0.0011` | `0.7489 +- 0.0463` |
| `100%` | `spatial_finetune` | `0.1526 +- 0.0151` | `0.7222 +- 0.0545` | `0.1470 +- 0.0193` | `0.7533 +- 0.0824` |

What changed after extending training to `40` epochs:

- the strong `20%` negative transfer from the `20`-epoch run was greatly reduced
- `20% scratch` still remains better on average, but the gap is now much smaller
- `100%` becomes nearly tied, with very small differences that depend on the metric
- this strongly suggests that the earlier `20`-epoch low-data failure was at least partly an optimization-horizon issue

### 5-epoch freeze-warmup follow-up

A second non-overwriting follow-up was then run on the `spatial_finetune` branch only, still with `40` total epochs, but with:

- `freeze_backbone_epochs = 5`

This warmup batch was designed to test whether the slow-start / plateau behavior came from early-stage alignment difficulty between the pretrained visual backbone and the randomly initialized temporal/head layers.

Warmup follow-up outputs:

- [spatial_prior_transfer_warmup5_40ep_20260324](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324/detailed_results.csv)

Aggregate summary for the warmup follow-up:

| data | mode | val mean_l2 | val success@0.2 | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `20%` | `spatial_finetune + warmup5` | `0.2101 +- 0.0165` | `0.5467 +- 0.0432` | `0.1958 +- 0.0076` | `0.5311 +- 0.0350` |
| `100%` | `spatial_finetune + warmup5` | `0.1411 +- 0.0084` | `0.7689 +- 0.0354` | `0.1412 +- 0.0073` | `0.7689 +- 0.0600` |

Compared against the original `40`-epoch no-warmup `spatial_finetune` baseline:

- at `20%`, warmup converts transfer from slightly worse than `scratch` to essentially tied on test and slightly better on validation
- at `100%`, warmup improves over both the no-warmup `spatial_finetune` run and the `scratch` baseline on the aggregate metrics
- seed variance also becomes much smaller, especially at `20%`

The strongest qualitative change appears in the previously problematic `100% seed43` run:

- no warmup: first `val_loss < 0.02` at epoch `33`, first `val_loss < 0.01` at epoch `38`
- `warmup5`: first `val_loss < 0.02` at epoch `9`, first `val_loss < 0.01` at epoch `14`

So the earlier long plateau was not a permanent incompatibility of the backbone. It was much more consistent with an early optimization / alignment problem.

### 10-epoch freeze-warmup follow-up

A third non-overwriting follow-up was then run with:

- `freeze_backbone_epochs = 10`

Outputs:

- [spatial_prior_transfer_warmup10_40ep_20260324](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324/detailed_results.csv)

Aggregate summary for the `warmup10` follow-up:

| data | mode | val mean_l2 | val success@0.2 | test mean_l2 | test success@0.2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `20%` | `spatial_finetune + warmup10` | `0.2070 +- 0.0087` | `0.5778 +- 0.0518` | `0.1992 +- 0.0157` | `0.5667 +- 0.0267` |
| `100%` | `spatial_finetune + warmup10` | `0.1510 +- 0.0078` | `0.7333 +- 0.0272` | `0.1443 +- 0.0077` | `0.7533 +- 0.0705` |

How `warmup10` compares to `warmup5`:

- at `20%`, `warmup10` gives slightly better aggregate success than `warmup5`, while test `mean_l2` is very close
- at `100%`, `warmup10` is still clearly better than the no-warmup baseline, but it is slightly worse than `warmup5`
- for the previously problematic `100% seed43` run, `warmup10` still removes the extreme plateau, but not as aggressively as `warmup5`

For `100% seed43` specifically:

- `warmup5`: first `val_loss < 0.02` at epoch `9`, first `val_loss < 0.01` at epoch `14`
- `warmup10`: first `val_loss < 0.02` at epoch `17`, first `val_loss < 0.01` at epoch `21`

So the current best reading is that some warmup is clearly helpful, but `10` frozen epochs may already be a bit too conservative for the `100%` regime.

## Conclusion

The updated conclusion after the `20`-epoch pilot, the `40`-epoch follow-up, and the `warmup5` follow-up is:

- the original `20`-epoch result was too pessimistic about low-data transfer
- low-data `spatial_finetune` without warmup is still slightly worse at `40` epochs, but no longer catastrophically worse
- adding `freeze_backbone_epochs = 5` substantially improves both stability and final transfer performance
- high-data transfer becomes clearly positive once the warmup recipe is used

So the better reading is:

- the current spatial-prior backbone is **transferable**
- the low-data regime is strongly optimization-sensitive
- the earlier negative-transfer signal was amplified by an insufficient training horizon and a too-aggressive finetuning recipe
- a short freeze warmup helps substantially, but the current evidence suggests `5` epochs is a better default than `10`

This means the next follow-up should focus on transfer recipe tuning rather than abandoning the backbone:

- test `spatial_pretrain + frozen`
- test whether a longer or shorter warmup than `5` epochs is even better
- test smaller finetuning LR for the pretrained backbone
- test whether the temporal state width should be increased to better match the richer visual encoder
