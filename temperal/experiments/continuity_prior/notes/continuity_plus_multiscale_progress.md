# Continuity + Multiscale Progress

This file records the current integration progress for combining the earlier Part I continuity-style visual prior with the current `multiscale` pipeline in `temperal/`.

## Status Note (2026-03-23)

This note mainly records an earlier integration route where the Part I continuity idea was approximated inside the shared `temperal/` codebase by reusing the current `MultiScaleEncoder` and supervising only the coarsest feature `f3`.

That route is no longer the main implementation target for the current `continuity_prior` experiment.

The current target is now:

- keep the experiment self-contained under `temperal/experiments/continuity_prior`
- build a dedicated multi-scale spatial-prior visual head
- preserve the Part I spatial continuity intuition
- make the visual interface natively compatible with the Part II temporal module

The latest design note for the current direction is:

- [spatial_prior_visual_head_alignment_plan_20260323.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/notes/spatial_prior_visual_head_alignment_plan_20260323.md)

The current downstream pilot plan for testing whether the pretrained visual backbone helps transfer is:

- [spatial_prior_transfer_pilot_plan_20260323.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_prior_transfer_pilot_plan_20260323.md)

The follow-up report focused on transfer-recipe tuning is:

- [spatial_prior_transfer_recipe_followup_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_prior_transfer_recipe_followup_20260325.md)

The first full `scratch vs spatial_finetune` transfer batch has now completed.

Aggregated outputs:

- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_20260323/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_20260323/detailed_results.csv)

A non-overwriting `40`-epoch follow-up has also been completed:

- [spatial_prior_transfer_pilot_40ep_20260323](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_40ep_20260323)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_pilot_40ep_20260323/summary.csv)

A focused `freeze_backbone_epochs = 5` warmup follow-up has now also been completed for the `spatial_finetune` branch:

- [spatial_prior_transfer_warmup5_40ep_20260324](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup5_40ep_20260324/summary.csv)

This warmup run is currently the strongest transfer recipe tested so far:

- at `20%`, it roughly closes the gap to `scratch`
- at `100%`, it improves over both the no-warmup `spatial_finetune` run and the `scratch` baseline
- it also removes most of the long plateau seen previously in the problematic `100% seed43` run

A second warmup sweep with `freeze_backbone_epochs = 10` has now also been completed:

- [spatial_prior_transfer_warmup10_40ep_20260324](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324)
- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/spatial_prior_transfer_warmup10_40ep_20260324/summary.csv)

Current reading:

- `warmup10` is still much better than no warmup
- but on the current aggregate numbers it does not beat `warmup5`
- so `freeze_backbone_epochs = 5` remains the best default transfer recipe tested so far

The rest of this file is still useful as background on the older `temperal`-internal continuity-pretrain route, but should now be read as historical/control context rather than the current implementation plan.

### Current experiment-local visual backbone status

The current experiment-local multi-scale spatial-prior visual backbone is now implemented and has been trained in a first non-smoke run.

Main training config:

- [multiscale_spatial_prior_train_20260323.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_spatial_prior_train_20260323.yaml)

Output directory:

- [multiscale_spatial_prior_train_20260323](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323)

Main backbone checkpoint for later loading:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/best.pt)

Training-process data saved for later plotting/inspection:

- [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/history.csv)
- [batch_log.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/batch_log.csv)
- [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/loss_curve.svg)
- [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/resolved_config.yaml)

## 1. Goal

The goal is to test whether the earlier spatial continuity prior from:

- `D:/projects/rl-ball-fetch/vis_backbone/v0.0.1`

can be used to pretrain the current `MultiScaleEncoder`, and whether that encoder pretraining improves the downstream `observe-then-act` regression task.

The intended comparison is:

- `scratch`
- `future-dynamics finetune`
- `continuity-encoder finetune`

## 2. Reference Method Being Reused

The reused idea comes from Part I `v0.0.1`:

- encoder/decoder backbone
- feature warp consistency
- warp-decode consistency
- optional neighbor smoothness

Relevant original files:

- [README.md](/d:/projects/rl-ball-fetch/vis_backbone/v0.0.1/README.md)
- [model.py](/d:/projects/rl-ball-fetch/vis_backbone/v0.0.1/model.py)
- [losses.py](/d:/projects/rl-ball-fetch/vis_backbone/v0.0.1/losses.py)
- [warp.py](/d:/projects/rl-ball-fetch/vis_backbone/v0.0.1/warp.py)

## 3. Current Integration Design

The current plan is the minimal integration path:

1. keep the current `MultiScaleEncoder`
2. apply the Part I continuity-style pretraining only to the encoder
3. use the coarsest feature map `f3` as the feature space for continuity losses
4. save `encoder_state_dict`
5. let downstream `multiscale_transfer` load this encoder-only checkpoint

This intentionally avoids changing the current downstream architecture at first.

## 4. New Files Added

### Pair dataset for continuity pretraining

- [frame_pair_dataset.py](/d:/projects/rl-ball-fetch/temperal/src/datasets/frame_pair_dataset.py)

Purpose:

- build adjacent frame pairs from the existing episode splits
- return `i_t`, `i_t1`, and a placeholder `flow_t`

Current assumption:

- flow is currently zero flow

### Continuity pretraining model

- [multiscale_continuity_pretrain.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_continuity_pretrain.py)

Current structure:

- reuse current `MultiScaleEncoder`
- take only the coarsest feature `f3`
- decode `f3` back to image space
- warp `f3` in feature space
- decode warped `f3`

Implemented losses:

- reconstruction loss
- transport consistency loss
- warp-decode consistency loss
- neighbor smoothness loss

### Continuity training entry

- [train_multiscale_continuity.py](/d:/projects/rl-ball-fetch/temperal/scripts/train_multiscale_continuity.py)

Implemented features:

- train/val split support
- TensorBoard logging
- `history.csv`
- `loss_curve.svg`
- `best.pt`
- `last.pt`
- `resolved_config.yaml`

### Smoke config

- [multiscale_continuity_smoke_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_continuity_smoke_20260316_220723.yaml)

## 5. Downstream Loading Update

The downstream transfer model was updated so it can load encoder-only checkpoints.

Relevant file:

- [multiscale_transfer_regressor.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_transfer_regressor.py)

Current behavior:

- if a checkpoint contains full `encoder + dynamics`, it loads both
- if a checkpoint contains only `encoder_state_dict`, it loads just the encoder

This means the continuity checkpoint can already be plugged into the current transfer code path.

## 6. Smoke Run Status

A continuity smoke run has already been completed.

Output directory:

- [multiscale_continuity_smoke_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_smoke_20260316_220723)

Saved artifacts:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_smoke_20260316_220723/best.pt)
- [last.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_smoke_20260316_220723/last.pt)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_smoke_20260316_220723/history.csv)
- [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_smoke_20260316_220723/loss_curve.svg)
- [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_smoke_20260316_220723/resolved_config.yaml)

Validation loss trend:

- epoch 1: `val_total = 0.0902`
- epoch 2: `val_total = 0.0624`
- epoch 3: `val_total = 0.0513`

Interpretation:

- the training pipeline is functional
- the continuity loss stack decreases cleanly
- checkpoint export works

## 7. Confirmed Working

The following has already been verified:

- the new continuity pretrain code compiles
- the smoke config parses
- training runs on GPU
- `best.pt` contains `encoder_state_dict`
- downstream transfer can load the saved encoder checkpoint successfully

## 8. Current Assumptions

The current version uses several simplifying assumptions:

- zero flow instead of real estimated flow
- continuity pretraining only on the coarsest encoder feature `f3`
- the continuity weight is currently set to `lambda_nb = 0.1`

## 9. Formal Continuity Pretrain Status

A formal continuity pretrain run has now been completed.

Config:

- [multiscale_continuity_formal_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_continuity_formal_20260316_220723.yaml)

Output directory:

- [multiscale_continuity_formal_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723)

Best validation loss:

- `val_total = 0.0169` at epoch `5`

Saved artifacts:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723/best.pt)
- [last.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723/last.pt)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723/history.csv)
- [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723/loss_curve.svg)
- [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723/resolved_config.yaml)

## 10. Downstream Comparison Status

Matched downstream finetune runs using the continuity checkpoint have now been completed under the same `obs50` protocol used for the earlier transfer data-efficiency experiments.

Aggregate directories:

- [20pct 3seed aggregate](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_20pct_obs50_3seed_20260316_220723)
- [40pct 3seed aggregate](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_40pct_obs50_3seed_20260316_220723)

High-level outcome:

- the continuity-pretrained encoder can sometimes finetune to a reasonable downstream solution
- however, the result is much less stable across seeds than the existing future-dynamics pretraining
- on 3-seed aggregate, continuity finetune is worse than both `scratch` and `future-dynamics finetune`

## 11. Next Planned Experiment

The next intended comparison is:

1. `scratch`
2. `future-dynamics finetune`
3. `continuity-encoder finetune`

This comparison now exists for the matched `obs50` downstream benchmark at `20%` and `40%` data.

The next useful follow-up is no longer "run the first downstream experiment", but rather:

1. diagnose why continuity finetune is unstable across seeds
2. test whether real flow, lighter LR, or staged freezing improves stability
3. keep using the current result as a control ablation against the future-dynamics pretraining line

## 12. Current Status Summary

The continuity prior has already been migrated into the current multiscale codebase at the encoder-pretraining level.

What is done:

- continuity pretrain dataset
- continuity pretrain model
- continuity pretrain training script
- smoke run
- formal pretrain run
- downstream encoder loading support
- matched downstream continuity finetune runs

What is not done yet:

- stability diagnosis and follow-up ablation
- improved continuity finetune recipe if we want it to compete with future-dynamics pretraining

## 13. Spatial Backbone + Temporal Pretrain

The next-stage stacked pretraining experiment has now been run:

1. start from the trained spatial-prior visual backbone
2. initialize the multiscale future-dynamics encoder with it
3. run temporal pretraining on top

Main report:

- [spatial_backbone_plus_future_pretrain_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_backbone_plus_future_pretrain_20260325.md)

Current status:

- the spatial-prior encoder loads cleanly into the temporal-pretrain model
- the stacked pretraining run finishes successfully to `20` epochs
- best validation loss reaches `0.3319` at epoch `19`

Current best stacked temporal-pretrain checkpoint:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_formal_seed42_20260325/best.pt)

Important caveat:

- this first comparison is still against the older `multiscale_future` scratch run with encoder width `8/8/8`
- the new spatial-init run uses encoder width `16/32/64`
- so the next clean control is a width-matched scratch temporal pretrain

## 14. Width-Matched 8/8/8 Stacked Pretraining

A stricter width-matched follow-up has now also been completed:

1. retrain a compact spatial-prior visual backbone with `8/8/8`
2. use it to initialize an `8/8/8` future-pretrain run
3. compare directly against the historical `8/8/8` scratch future-pretrain baseline

Main report:

- [spatial_backbone_plus_future_pretrain_888_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_backbone_plus_future_pretrain_888_20260325.md)

Current result:

- compact spatial visual backbone best `val_total = 0.0293`
- width-matched stacked temporal pretrain best `val_loss = 0.3587`
- historical `8/8/8` scratch future-pretrain reference best `val_loss = 0.3385`

So the first width-matched result says:

- stacked pretraining is feasible
- but this `8/8/8` spatial-prior init does not yet beat scratch temporal pretraining

## 15. Width-Matched Stacked Checkpoint on Downstream Transfer

The `8/8/8` stacked checkpoint has now been tested on the downstream `obs50` transfer benchmark.

Main report:

- [stacked_transfer_888_downstream_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/stacked_transfer_888_downstream_20260325.md)

High-level result:

- at `20%`, stacked transfer is the best of `scratch / future finetune / stacked888 finetune`
- at `40%`, stacked transfer is still better than scratch, but not better than historical future finetune
- at `100%`, stacked transfer is roughly scratch-level and trails historical future finetune

This means the current `8/8/8` stacked line is most promising as a low-data downstream prior.

## 16. Warmup5 Follow-up for Future vs Stacked888 Downstream

The warmup follow-up has now been completed for both pretrained downstream lines, including the requested `20%` setting.

Main report:

- [downstream_warmup5_followup_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/downstream_warmup5_followup_20260326.md)

High-level result:

- `future warmup5` helps the historical future-pretrain line at `20%`, but not at `40%/100%`
- `stacked888 warmup5` does not help overall and is clearly worse than no-warmup stacked888 at `20%`
- the best current low-data downstream recipe remains `stacked888` finetune without warmup

## 17. Smaller-Backbone-LR Follow-up for Stacked888

We also tested a gentler stacked888 finetune recipe using parameter groups:

- main lr `1e-3`
- backbone lr `1e-4` (`0.1x`)

Main report:

- [stacked_transfer_backbone_lr_followup_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/stacked_transfer_backbone_lr_followup_20260326.md)

High-level result:

- this `0.1x` backbone-lr recipe is worse than the default stacked888 finetune at both `20%` and `100%`
- so neither `warmup5` nor `backbone lr = 0.1x` improves the current stacked downstream line
- the strongest current low-data recipe still remains plain `stacked888` end-to-end finetune

## 18. Joint Future + Spatial Pretraining Smoke

We have now implemented the first simultaneous-pretraining version inside `continuity_prior`.

Main report:

- [joint_future_spatial_smoke_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/joint_future_spatial_smoke_20260326.md)

High-level result:

- the joint future+spatial training line runs end-to-end and saves checkpoints correctly
- both train and validation total loss decrease over the smoke run
- the first smoke also revealed that the spatial term is currently much smaller than the future term, so the next real pilot should likely increase the spatial weight

## 19. Joint Future + Spatial 20-Epoch Pilot

A first full-data `20`-epoch joint pilot has now been completed for seed `42`.

Main report:

- [joint_future_spatial_formal_seed42_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/joint_future_spatial_formal_seed42_20260326.md)

High-level result:

- the full joint run completes successfully with resume support
- but this first naive joint recipe is much worse than the earlier sequential `spatial-init -> future-pretrain` line
- best joint validation total loss is `2.6104`, versus `0.3587` for the earlier width-matched sequential future-pretrain reference

So the current conclusion is:

- simultaneous pretraining is feasible
- but the first `lambda_spatial_total = 1.0` recipe is too disruptive and should not be treated as a good default

## 20. Shared Visual Decoder Adapter Smoke

We have now implemented a second way to couple the spatial prior with temporal pretraining:

- keep the spatial-prior encoder init
- reuse the spatial-prior decoder during future pretraining
- insert a small adapter between rollout state and shared visual decoder

Main report:

- [future_decoder_adapter_smoke_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/future_decoder_adapter_smoke_20260326.md)

High-level result:

- both frozen-decoder and unfrozen-decoder adapter smoke runs train successfully
- the unfrozen adapter variant is slightly better than the frozen one
- the adapter line currently gives very strong future frame reconstruction, but still trails the earlier joint smoke on total future loss because position prediction is weaker

So the current conclusion is:

- the adapter idea is structurally valid
- but the first smoke does not yet make it the new best temporal-pretrain recipe
- if we continue here, the unfrozen-decoder branch is the better one to push further

## 21. Shared Visual Decoder Adapter 20-Epoch Formal Pilot

We have now completed a first full-data `20`-epoch temporal-pretrain run for the better adapter branch:

- unfrozen shared visual decoder
- `8/8/8`
- seed `42`

Main report:

- [future_decoder_adapter_unfreeze_formal_seed42_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/future_decoder_adapter_unfreeze_formal_seed42_20260326.md)

High-level result:

- best validation loss is `0.3436` at epoch `19`
- this is clearly better than the earlier sequential `spatial-init -> future-pretrain` result (`0.3587`)
- it is also dramatically better than the naive joint-loss pilot (`2.6104`)
- but it still sits slightly above the historical scratch baseline (`0.3385`)

So the current conclusion is:

- the adapter route is the most promising temporal-pretrain coupling strategy tried so far
- it is now close enough to scratch that downstream transfer is worth testing directly

## 22. Adapter Temporal-Pretrain on Downstream Transfer

We have now completed the downstream transfer benchmark for the new adapter-based temporal-pretrain checkpoint.

Main report:

- [adapter_transfer_downstream_20260327.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/adapter_transfer_downstream_20260327.md)

Main artifacts:

- [summary.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/adapter_transfer_20260327/summary.csv)
- [detailed_results.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/adapter_transfer_20260327/detailed_results.csv)
- [adapter_transfer_vs_baselines_curves.png](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/adapter_transfer_20260327/adapter_transfer_vs_baselines_curves.png)

High-level result:

- at `20%`, adapter transfer is strong but still slightly below `stacked888`
- at `40%`, adapter transfer is the new best setting across `scratch / future / stacked888 / adapter`
- at `100%`, adapter transfer is also the new best setting

So the current conclusion is:

- `stacked888` remains the most promising low-data specialist
- `adapter` is now the strongest overall pretrained downstream recipe tried so far

## 23. Overall Summary Report

We now also have a single synthesis report that answers the main research question for this whole period:

- can spatial prior + temporal pretraining help the downstream landing task without directly using landing labels during pretraining?

Main report:

- [continuity_prior_overall_summary_20260327.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/continuity_prior_overall_summary_20260327.md)

High-level answer:

- yes
- the effect is real but recipe-dependent
- `stacked888` is the strongest low-data specialist
- `adapter` is the strongest overall mid/high-data recipe tried so far
