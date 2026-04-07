# Future Pretrain with Shared Visual Decoder Adapter Smoke

Date: `2026-03-26`

## Goal

Test a cleaner alternative to the earlier joint-loss recipe:

- keep the spatial-prior visual encoder initialization
- reuse the spatial-prior visual decoder during future pretraining
- insert a small adapter between temporal rollout states and the shared visual decoder

The hope is that this would encourage rollout states to stay in a decoder-readable visual feature space, without directly adding a second spatial loss term to the encoder.

## Setup

Trainer:

- [train_future_with_decoder_adapter.py](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/scripts/train_future_with_decoder_adapter.py)

We ran two smoke settings on the same `8/8/8` backbone and the same small temporal split:

1. frozen shared decoder
- config:
  [multiscale_future_decoder_adapter_smoke_20260326.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_future_decoder_adapter_smoke_20260326.yaml)
- output:
  [multiscale_future_decoder_adapter_smoke_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_smoke_20260326)

2. unfrozen shared decoder
- config:
  [multiscale_future_decoder_adapter_unfreeze_smoke_20260326.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_future_decoder_adapter_unfreeze_smoke_20260326.yaml)
- output:
  [multiscale_future_decoder_adapter_unfreeze_smoke_20260326](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_future_decoder_adapter_unfreeze_smoke_20260326)

Shared settings:

- width-matched `8/8/8`
- spatial init from
  [best.pt](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/output/multiscale_spatial_prior_train_888_15ep_20260325/best.pt)
- `train_max_clips = 512`
- `val_max_clips = 128`
- `3` epochs
- no extra spatial regularization loss

## Result

Validation loss after `3` epochs:

| variant | val_loss | val_frame | val_pos | val_vel |
| --- | ---: | ---: | ---: | ---: |
| adapter + frozen decoder | `4.3101` | `0.0221` | `3.5648` | `3.6160` |
| adapter + unfrozen decoder | `4.2990` | `0.0029` | `3.5838` | `3.5617` |
| earlier joint smoke reference | `4.0668` total | `0.0222` frame | `3.3196` | `3.5978` |

So the key observations are:

1. the adapter implementation is valid and trains stably
2. unfreezing the shared decoder is slightly better than freezing it
3. the adapter line already drives frame reconstruction very low when the decoder can co-adapt
4. but after `3` epochs it still does **not** beat the earlier joint smoke on total future loss

## What Looks Encouraging

The interesting signal is not the total loss yet.

The encouraging part is that the unfrozen adapter run gets:

- `val_frame = 0.0029`

which is much lower than the frozen-adapter run and also much lower than the earlier joint smoke frame term.

This suggests that:

- the temporal rollout state can in fact be translated into a visual-decoder-readable space
- the adapter idea is structurally plausible

## What Still Looks Weak

The total loss is still dominated by position and velocity terms, especially position:

- frozen adapter: `val_pos = 3.5648`
- unfrozen adapter: `val_pos = 3.5838`
- joint smoke reference: `val_pos = 3.3196`

So right now the adapter branch is better at future frame decoding than at building the latent state needed for accurate probe prediction.

## Current Takeaway

The adapter idea is worth keeping.

But the first smoke says:

- "shared visual decoder + adapter" is **not yet** a drop-in improvement over the earlier joint smoke
- its most obvious strength is cleaner frame decoding, not better overall predictive state quality

## Recommended Next Step

If we continue on the adapter line, the more promising branch is:

1. keep the shared decoder **unfrozen**
2. train longer than `3` epochs
3. watch whether the better frame-space alignment eventually helps the position/velocity terms catch up

So the next useful question is not "does adapter compile", but:

- can the unfrozen-adapter recipe turn its strong frame reconstruction into better future-state prediction when trained longer?
