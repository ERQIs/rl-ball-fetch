# Hierarchical Short-To-Long Landing Design

## Motivation

The current adapter temporal pretraining learns a **short-horizon motion model**:

- observe `8` frames
- predict the next `4` future steps
- training stride is `frame_stride = 2`

So it is better interpreted as a **short-term motion encoder** than as a reliable long-range rollout engine.

This suggests a more aligned downstream design:

1. sample the first `50%` of the trajectory at the same temporal frequency used in pretraining
2. break that visible prefix into multiple local `8`-frame motion windows
3. encode each local window into a short-term motion token using the pretrained temporal backbone
4. aggregate those tokens with a lighter long-range module
5. decode the final landing point from the aggregated long-range trajectory representation

## First-Version Design

### Local motion token

- input: one local window of `8` frames
- local frame spacing: `2`
- backbone: pretrained adapter temporal encoder+dynamics
- token readout: pooled final `h3`

This token is meant to summarize a short segment of motion that matches the temporal pretraining regime.

### Long-range aggregation

- visible prefix: first `50%` of the episode
- prefix sampled at `frame_stride = 2`
- local window length: `8`
- local window hop: `4` sampled frames
- long-range module: single-layer `GRU`
- final head: MLP predicting explicit `landing_px / landing_pz`

### Training policy

First version:

- freeze the pretrained local temporal backbone
- train only:
  - token projection
  - long-range GRU
  - landing head

This keeps the short-term motion representation stable and tests whether long-range aggregation alone can extract better landing information.

## Expected Advantage Over Direct Rollout

Compared with direct latent rollout to landing:

- avoids long-range error accumulation
- respects the short-horizon nature of temporal pretraining
- uses the whole visible trajectory prefix more faithfully
- separates:
  - short-term motion encoding
  - long-term trajectory aggregation

## Implementation Plan

- experiment-local dataset producing padded local-window tensors
- experiment-local hierarchical model
- experiment-local training script
- smoke run first, then formal run if the pipeline is healthy

## Status

- design approved
- implementation completed

## Implemented Files

- dataset: [hierarchical_motion_windows_dataset.py](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/src/datasets/hierarchical_motion_windows_dataset.py)
- model: [hierarchical_motion_landing.py](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/src/models/hierarchical_motion_landing.py)
- training script: [train_hierarchical_motion_landing.py](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/scripts/train_hierarchical_motion_landing.py)
- smoke config: [hierarchical_motion_landing_smoke_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/hierarchical_motion_landing_smoke_20260327.yaml)

## Smoke Result

First smoke run settings:

- local stride: `2`
- local window length: `8`
- local window hop: `4`
- visible prefix: first `50%`
- local temporal backbone: pretrained adapter checkpoint initialization
- local temporal backbone training policy: frozen
- long-range module: `1-layer GRU`
- data budget: `128 / 64 / 64` train/val/test episodes
- epochs: `3`

Output:

- [hierarchical_motion_landing_smoke_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_smoke_20260327)
- [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_smoke_20260327/summary.json)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_smoke_20260327/history.csv)

Observed metrics:

- best epoch: `3`
- best val mean L2: `4.4549`
- test mean L2: `4.6363`
- test success@0.2: `0.0000`

Interpretation:

- the hierarchical pipeline is technically healthy: dataset, padding, local token extraction, GRU aggregation, and landing readout all run end-to-end
- but this first frozen-backbone smoke is not yet competitive
- compared with the direct `rollout0 landing head` line, this first version is dramatically undertrained or mismatched

Immediate next options:

- train the same hierarchical design longer before judging it
- unfreeze part of the local temporal backbone
- change the local token readout or use multiple local summary tokens per window

## 20-Epoch Follow-Up

The same smoke run was resumed from epoch `3` to epoch `20` with the same output directory and the same frozen-local-backbone recipe.

Resume config:

- [hierarchical_motion_landing_smoke_resume20_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/hierarchical_motion_landing_smoke_resume20_20260327.yaml)

Updated output:

- [hierarchical_motion_landing_smoke_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_smoke_20260327)
- [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_smoke_20260327/summary.json)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_smoke_20260327/history.csv)

Updated metrics at `20` epochs:

- best epoch: `20`
- best val mean L2: `1.8322`
- test mean L2: `2.0621`
- test success@0.2: `0.0469`

Compared with the original `3`-epoch smoke:

- val mean L2 improved from `4.4549` to `1.8322`
- test mean L2 improved from `4.6363` to `2.0621`
- test success moved from `0.0000` to `0.0469`

Interpretation of the follow-up:

- this confirms the hierarchical line was strongly undertrained at `3` epochs
- the model is capable of steady optimization under the current recipe
- but even at `20` epochs it is still much weaker than the direct `rollout0 landing head` baseline, so the current frozen-local-backbone design is not yet competitive

## Unfreeze-Dynamics Follow-Up

We then changed the local-backbone policy to:

- `freeze_local_encoder = true`
- `freeze_local_dynamics = false`

This keeps the visual encoder fixed while allowing the local short-term temporal module to adapt to the long-range landing objective.

Config:

- [hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327.yaml)

Output:

- [hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327)
- [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/summary.json)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/history.csv)

Observed metrics at `20` epochs:

- best epoch: `19`
- best val mean L2: `1.0097`
- test mean L2: `1.2713`
- test success@0.2: `0.0000`

Comparison against the frozen-local-backbone `20`-epoch run:

- val mean L2 improved from `1.8322` to `1.0097`
- test mean L2 improved from `2.0621` to `1.2713`
- success@0.2 did not improve and remained effectively unusable on this small test slice

Interpretation:

- unfreezing only `dynamics` clearly helps optimization in terms of average landing error
- this supports the hypothesis that the local motion-state organization, not only the long-range GRU, needs adaptation
- however, the model is still far from the strict `0.2` success threshold, so the improvement has not yet translated into strong success-rate gains

## Resume-To-100 Follow-Up

The `encoder frozen + dynamics unfrozen` run was then resumed from epoch `20` to epoch `100`.

Resume config:

- [hierarchical_motion_landing_unfreeze_dynamics_100ep_resume_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/hierarchical_motion_landing_unfreeze_dynamics_100ep_resume_20260327.yaml)

Important note:

- the resumed run reused the same output directory
- so [hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327) now contains the full `1..100` epoch trajectory

Intermediate data saved during training:

- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/history.csv)
- [mean_l2_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/mean_l2_curve.svg)
- [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/summary.json)
- [val_predictions.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/val_predictions.csv)
- [test_predictions.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/test_predictions.csv)

Updated metrics at `100` epochs:

- best epoch: `97`
- best val mean L2: `0.3378`
- val success@0.2: `0.4062`
- test mean L2: `0.3669`
- test success@0.2: `0.2969`

Comparison against the earlier `20`-epoch unfreeze-dynamics run:

- val mean L2 improved from `1.0097` to `0.3378`
- test mean L2 improved from `1.2713` to `0.3669`
- test success@0.2 improved from `0.0000` to `0.2969`

Interpretation:

- this hierarchical route was also substantially undertrained at `20` epochs
- once allowed to run long enough, the encoder-frozen+dynamics-unfrozen recipe becomes much more competitive
- the result is still behind the strongest direct `rollout0 landing head` line, but it is no longer obviously failing

## Resume-To-300 Follow-Up

The same `encoder frozen + dynamics unfrozen` run was resumed again from epoch `100` to epoch `300`.

Resume config:

- [hierarchical_motion_landing_unfreeze_dynamics_300ep_resume_20260327.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/configs/hierarchical_motion_landing_unfreeze_dynamics_300ep_resume_20260327.yaml)

Output directory remained the same:

- [hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327)

This means all intermediate data for the full `1..300` epoch run are now stored together in:

- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/history.csv)
- [mean_l2_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/mean_l2_curve.svg)
- [summary.json](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/summary.json)
- [val_predictions.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/val_predictions.csv)
- [test_predictions.csv](/d:/projects/rl-ball-fetch/temperal/experiments/landing_world_model/output/hierarchical_motion_landing_unfreeze_dynamics_20ep_20260327/test_predictions.csv)

Updated metrics at `300` epochs:

- best epoch: `285`
- best val mean L2: `0.2993`
- val success@0.2: `0.4688`
- test mean L2: `0.3598`
- test success@0.2: `0.3438`

Comparison against the earlier `100`-epoch checkpoint:

- val mean L2 improved from `0.3378` to `0.2993`
- test mean L2 improved from `0.3669` to `0.3598`
- test success@0.2 improved from `0.2969` to `0.3438`

Interpretation:

- the run still benefits from longer training, but gains are now much smaller than the `20 -> 100` jump
- the best checkpoint appears late (`epoch 285`), so this recipe is not obviously overtrained at `300` epochs
- the hierarchical route now reaches a usable regime, though it still does not clearly surpass the simpler direct landing-head baseline
