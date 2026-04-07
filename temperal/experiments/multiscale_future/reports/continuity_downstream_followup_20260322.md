# Continuity Downstream Follow-up

Date: `2026-03-22`

## 1. Purpose

This note continues the `temperal/experiments/continuity_prior` line by checking why the first downstream `continuity-encoder finetune` runs were unstable and whether a lightweight stabilization recipe can recover them.

## 2. Original Downstream Status

The original downstream continuity runs already existed for:

- `20%` data, seeds `42/43/44`
- `40%` data, seeds `42/43/44`

Two seeds clearly collapsed:

- `20% seed43`: test `mean_l2 = 1.0534`, `success@0.2 = 0.0867`
- `40% seed44`: test `mean_l2 = 1.9606`, `success@0.2 = 0.0067`

Original aggregate summaries:

- `20%`: test `mean_l2 = 0.5321 ± 0.3686`, success `0.2733 ± 0.1338`
- `40%`: test `mean_l2 = 0.7852 ± 0.8311`, success `0.3711 ± 0.2578`

These summaries were dominated by failure cases rather than normal variance.

## 3. Diagnosis

Using the same input mini-batch, the encoder feature scale was compared between the future-dynamics checkpoint and the continuity checkpoint.

Observed `f3` feature statistics:

- future pretrain: `mean_abs ≈ 0.7899`, `std ≈ 1.1374`
- continuity pretrain: `mean_abs ≈ 0.0631`, `std ≈ 0.0807`

Interpretation:

- the continuity encoder produces a much smaller coarse feature magnitude
- the downstream transfer stack pools the hidden `h3` state directly into the regression head
- this makes optimization much more sensitive to initialization when the continuity encoder is loaded

## 4. Stabilization Changes Added

Two lightweight controls were added to the downstream code path:

1. `model.pre_head_layernorm`
2. `training.freeze_backbone_epochs`

Relevant updates:

- [multiscale_transfer_regressor.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_transfer_regressor.py)
- [train_multiscale_downstream.py](/d:/projects/rl-ball-fetch/temperal/scripts/train_multiscale_downstream.py)
- [run_multiscale_transfer_grid.py](/d:/projects/rl-ball-fetch/temperal/scripts/run_multiscale_transfer_grid.py)

New retry configs:

- [multiscale_transfer_continuity_finetune_40pct_obs50_seed44_ln_20260322.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_transfer_continuity_finetune_40pct_obs50_seed44_ln_20260322.yaml)
- [multiscale_transfer_continuity_finetune_20pct_obs50_seed43_ln_20260322.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_transfer_continuity_finetune_20pct_obs50_seed43_ln_20260322.yaml)
- [multiscale_transfer_continuity_finetune_20pct_obs50_seed43_ln_warmup5_20260322.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_transfer_continuity_finetune_20pct_obs50_seed43_ln_warmup5_20260322.yaml)

## 5. Retry Results

### `40% seed44`

Original:

- test `mean_l2 = 1.9606`
- test `success@0.2 = 0.0067`

Retry with `pre_head_layernorm = true`:

- [multiscale_transfer_continuity_finetune_40pct_obs50_seed44_ln_20260322](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_40pct_obs50_seed44_ln_20260322)
- test `mean_l2 = 0.2475`
- test `success@0.2 = 0.4533`

### `20% seed43`

Original:

- test `mean_l2 = 1.0534`
- test `success@0.2 = 0.0867`

Retry with only `pre_head_layernorm = true`:

- [multiscale_transfer_continuity_finetune_20pct_obs50_seed43_ln_20260322](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_20pct_obs50_seed43_ln_20260322)
- test `mean_l2 = 1.9671`
- test `success@0.2 = 0.0067`

Retry with `pre_head_layernorm = true` and `freeze_backbone_epochs = 5`:

- [multiscale_transfer_continuity_finetune_20pct_obs50_seed43_ln_warmup5_20260322](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_20pct_obs50_seed43_ln_warmup5_20260322)
- test `mean_l2 = 0.3547`
- test `success@0.2 = 0.2467`

Conclusion from retries:

- `LayerNorm` alone is enough to rescue the `40%` failure case
- the lower-data `20%` failure case also needs a short backbone-freeze warmup

## 6. Stabilized Aggregate Comparison

Stabilized continuity aggregate outputs:

- [multiscale_transfer_continuity_finetune_20pct_obs50_stabilized_3seed_20260322](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_20pct_obs50_stabilized_3seed_20260322)
- [multiscale_transfer_continuity_finetune_40pct_obs50_stabilized_3seed_20260322](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_40pct_obs50_stabilized_3seed_20260322)

### `20%` data

- continuity stabilized: test `mean_l2 = 0.2992 ± 0.0394`, success `0.3267 ± 0.0606`
- scratch baseline: test `mean_l2 = 0.2654 ± 0.0130`, success `0.3844 ± 0.0257`
- future-dynamics finetune: test `mean_l2 = 0.2435 ± 0.0103`, success `0.4578 ± 0.0658`

### `40%` data

- continuity stabilized: test `mean_l2 = 0.2142 ± 0.0235`, success `0.5200 ± 0.0475`
- scratch baseline: test `mean_l2 = 0.2070 ± 0.0069`, success `0.5156 ± 0.0503`
- future-dynamics finetune: test `mean_l2 = 0.1789 ± 0.0103`, success `0.6156 ± 0.0227`

## 7. Current Takeaway

The continuity encoder can be made trainable downstream with a small stabilization recipe, but under the current zero-flow and `f3`-only setup it still does not beat the existing future-dynamics pretraining and does not clearly beat `scratch`.

What improved:

- downstream continuity runs are no longer dominated by catastrophic bad seeds
- the continuation path is now configurable and reproducible

What remains true:

- continuity is still behind the future-dynamics line
- at `20%`, continuity stabilized is still worse than `scratch`

## 8. Recommended Next Step

The clean next continuation would be:

1. run a matched stabilized continuity grid for `100%`
2. if continuity still trails `scratch`, stop scaling this exact zero-flow variant
3. move the method change to either real/estimated flow, multi-scale continuity supervision, or stronger coarse-feature variance control during pretraining
