# Future Pretrain with Shared Visual Decoder Adapter

Date: `2026-03-26`

## Goal

Run a first full-data `20`-epoch temporal-pretrain pilot using:

- spatial-prior `8/8/8` encoder initialization
- the same spatial-prior visual decoder as the future-frame decoder
- a small adapter between temporal rollout state and shared visual decoder
- the shared decoder left **unfrozen**

This was meant to test a cleaner coupling strategy than the earlier joint-loss recipe.

## Config

Formal config:

- [multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326.yaml)

Output:

- [multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326)

Main artifacts:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/best.pt)
- [last.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/last.pt)
- [history.csv](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/history.csv)
- [best_val_metrics.json](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/best_val_metrics.json)
- [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/loss_curve.svg)
- [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/resolved_config.yaml)

Key settings:

- width-matched `8/8/8`
- seed `42`
- full temporal train/val split
- `20` epochs
- shared visual decoder init from
  [multiscale_spatial_prior_train_888_15ep_20260325/best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_spatial_prior_train_888_15ep_20260325/best.pt)
- decoder is trainable, not frozen

## Result

Best validation loss:

- `0.3436` at epoch `19`

Best validation decomposition:

- frame: `0.00237`
- pos: `0.21462`
- vel: `0.63282`

The run still improves late into training:

- epoch `16`: `0.4058`
- epoch `17`: `0.3843`
- epoch `18`: `0.3799`
- epoch `19`: `0.3436`
- epoch `20`: `0.3451`

So unlike the earlier naive joint-loss pilot, this adapter version does **not** collapse into an early plateau.

## Comparison

Reference width-matched sequential run:

- [multiscale_future_spatialinit_888_formal_seed42_20260325](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatialinit_888_formal_seed42_20260325)
- report:
  [spatial_backbone_plus_future_pretrain_888_20260325.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/spatial_backbone_plus_future_pretrain_888_20260325.md)

Reference earlier joint-loss run:

- [multiscale_future_spatial_joint_formal_seed42_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_spatial_joint_formal_seed42_20260326)
- report:
  [joint_future_spatial_formal_seed42_20260326.md](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/reports/joint_future_spatial_formal_seed42_20260326.md)

Historical scratch baseline:

- [multiscale_future_formal_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723)

Best validation loss comparison:

| run | recipe | best val_loss | best epoch |
| --- | --- | ---: | ---: |
| scratch future pretrain | scratch `8/8/8` | `0.3385` | `19` |
| sequential spatial-init | spatial encoder init only | `0.3587` | `19` |
| joint future + spatial loss | simultaneous loss coupling | `2.6104` | `14` |
| adapter + shared decoder | spatial encoder + shared decoder + adapter | `0.3436` | `19` |

## Interpretation

This is the first adapter result that looks genuinely promising.

The adapter version:

1. is much better than the earlier naive joint-loss pilot
2. improves on the earlier sequential `spatial-init -> future-pretrain` result
3. gets very close to the historical scratch baseline

The remaining gap to scratch is small:

- `0.3436 - 0.3385 = 0.0051`

So the current reading is:

- shared-decoder coupling is a much healthier way to keep spatial structure in the loop than the first joint-loss recipe
- but this first adapter run still does not quite beat the scratch future-pretrain baseline

## Current Takeaway

The adapter idea should stay alive.

At the moment it is:

- clearly better than the naive joint-loss version
- better than the earlier sequential spatial-init future-pretrain
- close enough to scratch that downstream transfer is now worth checking

## Recommended Next Step

The most useful next experiment is now straightforward:

1. take [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_formal_seed42_20260326/best.pt)
2. run the downstream transfer benchmark
3. compare against:
   - scratch
   - historical future finetune
   - earlier stacked888 finetune

This will tell us whether the adapter-based temporal-pretrain bias is more downstream-friendly than what the temporal-pretrain validation loss alone suggests.
