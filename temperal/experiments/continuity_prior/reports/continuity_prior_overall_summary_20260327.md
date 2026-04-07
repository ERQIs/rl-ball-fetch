# Continuity Prior Overall Summary

Date: `2026-03-27`

## Research Question

This experiment line was designed to answer one main question:

- can **spatial prior + temporal pretraining** learn representations that help the final landing-point downstream task **without directly using landing labels during pretraining**?

In this project, "without landing labels during pretraining" means:

- spatial-prior visual training uses reconstruction / transport / warp-decode / smoothness style losses
- temporal pretraining uses future-frame reconstruction and intermediate position / velocity supervision
- the final landing-point supervision is only used in the downstream transfer benchmark

So this line is trying to test whether useful landing-prediction representations can emerge indirectly, without pretraining directly on the final landing target.

## What Was Tried

The main branches explored in this period were:

1. **spatial prior only -> downstream transfer**
- first with the original `16/32/64` spatial backbone
- then with transfer-recipe follow-ups such as frozen-backbone warmup

2. **sequential stacked pretraining**
- `8/8/8` spatial-prior visual backbone
- then `8/8/8` temporal future pretraining initialized from that backbone
- then downstream finetune

3. **naive joint future + spatial loss**
- future pretraining and spatial-prior loss active together during temporal pretraining

4. **shared visual decoder + adapter**
- spatial-prior encoder init
- temporal rollout state passed through a small adapter
- shared visual decoder reused during future pretraining
- then downstream finetune

Main detailed reports from these branches:

- [spatial_prior_transfer_pilot_plan_20260323.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_prior_transfer_pilot_plan_20260323.md)
- [spatial_prior_transfer_recipe_followup_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_prior_transfer_recipe_followup_20260325.md)
- [spatial_backbone_plus_future_pretrain_888_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_backbone_plus_future_pretrain_888_20260325.md)
- [stacked_transfer_888_downstream_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/stacked_transfer_888_downstream_20260325.md)
- [joint_future_spatial_formal_seed42_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/joint_future_spatial_formal_seed42_20260326.md)
- [future_decoder_adapter_unfreeze_formal_seed42_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/future_decoder_adapter_unfreeze_formal_seed42_20260326.md)
- [adapter_transfer_downstream_20260327.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/adapter_transfer_downstream_20260327.md)

## Main Findings by Branch

### 1. Spatial Prior Alone Is Transferable, but Recipe-Sensitive

The first spatial-only transfer pilot showed:

- `20%`: naive spatial finetune was clearly worse than scratch
- `100%`: naive spatial finetune was modestly better than scratch

Later follow-up showed that this was mostly an optimization issue rather than a fundamentally bad visual prior:

- `warmup5` and `warmup10` largely removed the low-data negative transfer
- `warmup5` became the best default recipe for the spatial-only transfer line

Key result from the follow-up:

| setting | `20%` test mean_l2 / success | `100%` test mean_l2 / success |
| --- | --- | --- |
| scratch | `0.2042 / 0.5311` | `0.1439 / 0.7489` |
| spatial finetune | `0.2817 / 0.4578` | `0.1470 / 0.7533` |
| spatial finetune + warmup5 | `0.1958 / 0.5311` | `0.1412 / 0.7689` |

So the visual spatial prior by itself already contains task-relevant signal, but its transfer requires a careful finetune recipe.

### 2. Sequential Stacked Pretraining Helps Low-Data Downstream

The first width-matched sequential stacked line used:

- `8/8/8` spatial-prior visual backbone
- `8/8/8` temporal future pretrain initialized from that backbone

On temporal-pretrain validation loss, it did **not** beat scratch:

- scratch future pretrain: `0.3385`
- sequential spatial-init future pretrain: `0.3587`

But downstream told a more useful story:

| data | scratch | future finetune | stacked888 finetune |
| --- | --- | --- | --- |
| `20%` | `0.2654 / 0.3844` | `0.2435 / 0.4578` | `0.2272 / 0.5111` |
| `40%` | `0.2070 / 0.5156` | `0.1789 / 0.6156` | `0.1945 / 0.5822` |
| `100%` | `0.1632 / 0.6711` | `0.1493 / 0.7200` | `0.1658 / 0.6733` |

This is one of the most important results of the whole line:

- even though stacked temporal pretraining was slightly worse than scratch in pretrain loss
- it became the **best low-data downstream prior** at `20%`

So pretrain ranking and downstream ranking were clearly not the same.

### 3. Naive Joint Future + Spatial Loss Failed

The first simultaneous joint-loss pilot asked the temporal-pretrain stage to optimize:

- future objective
- spatial-prior objective

at the same time on the shared encoder.

This worked as an implementation, but not as a recipe:

- joint-loss best validation total loss: `2.6104`
- sequential width-matched reference: `0.3587`

So the first naive joint-loss recipe over-corrected and strongly hurt future learning.

This branch is important mostly as a negative result:

- simultaneous pretraining is possible
- but the first direct loss-coupling version is **not** the right way to keep spatial structure alive

### 4. Shared Decoder + Adapter Is the Best Coupling Tried So Far

The adapter branch changed the coupling idea:

- initialize from the spatial-prior encoder
- reuse the spatial-prior visual decoder
- insert a small adapter from temporal rollout state to decoder space

Its `20`-epoch temporal-pretrain result was:

- adapter future pretrain: `0.3436`
- scratch future pretrain: `0.3385`
- sequential stacked future pretrain: `0.3587`
- naive joint-loss: `2.6104`

So this was:

- much better than naive joint-loss
- better than sequential stacked temporal pretraining
- very close to the scratch temporal-pretrain baseline

Then downstream transfer gave the strongest overall result of this period:

| data | scratch | future finetune | stacked888 finetune | adapter finetune |
| --- | --- | --- | --- | --- |
| `20%` | `0.2654 / 0.3844` | `0.2435 / 0.4578` | `0.2272 / 0.5111` | `0.2290 / 0.4956` |
| `40%` | `0.2070 / 0.5156` | `0.1789 / 0.6156` | `0.1945 / 0.5822` | `0.1737 / 0.6289` |
| `100%` | `0.1632 / 0.6711` | `0.1493 / 0.7200` | `0.1658 / 0.6733` | `0.1479 / 0.7400` |

This means:

- `20%`: adapter is strong, but still slightly below `stacked888`
- `40%`: adapter is the new best method
- `100%`: adapter is also the new best method

## Direct Answer to the Research Question

The answer is now:

- **Yes.**

Spatial prior + temporal pretraining **can** learn representations that improve the final landing-point downstream task **without directly using landing labels during pretraining**.

The evidence is not just one isolated run:

- spatial prior alone can help downstream after recipe stabilization
- stacked pretraining gives the strongest low-data result
- adapter-based coupling gives the strongest mid/high-data result

So the combined line is not "white enhancement" or null.

It is genuinely learning transferable task-relevant structure without being directly pretrained on the final landing target.

## What the Current Evidence Also Says

The more nuanced answer is:

1. **The answer depends heavily on how spatial prior is coupled to temporal pretraining.**
- naive joint-loss coupling fails
- shared-decoder adapter coupling works much better

2. **The answer depends on data regime.**
- `stacked888` is best at `20%`
- `adapter` is best at `40%` and `100%`

3. **The pretraining target is still only partially aligned with the downstream task.**
- current future pretraining mainly teaches short-horizon state estimation and short-horizon prediction
- downstream asks for final landing-point prediction from partial observation
- so useful transfer is possible, but the alignment is not perfect

## Current Best Recipes

If the goal is downstream performance with the methods already tested:

- **best low-data (`20%`) recipe:** `stacked888 finetune`
- **best overall mid/high-data (`40%`, `100%`) recipe:** `adapter finetune`

If the goal is "best temporal-pretrain coupling strategy tried so far":

- **best current answer:** shared visual decoder + adapter

If the goal is "best pure visual transfer recipe":

- **best current answer:** spatial finetune + short warmup (`warmup5`)

## Suggested Next Steps

The current line is already strong enough that new work should probably focus on one of these directions rather than reopening old branches:

1. **Prioritize the adapter line**
- it is the strongest overall pretrained downstream recipe tested so far

2. **Preserve the stacked888 line as the low-data specialist**
- it remains the best `20%` result

3. **If continuing research rather than just maximizing score**
- investigate why low-data prefers sequential stacked while mid/high-data prefers adapter
- this looks like a real representation-bias difference, not just noise

## Bottom Line

The main result of this period is:

- spatial prior + temporal pretraining **does** help downstream landing prediction without directly pretraining on landing labels
- but the gain depends strongly on the coupling recipe
- the best current overall solution is the **adapter-based temporal-pretrain + downstream finetune** line
- the best current low-data specialist is still **stacked888**
