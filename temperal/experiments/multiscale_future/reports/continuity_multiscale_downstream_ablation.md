# Continuity Prior + Multiscale Downstream Ablation

## Question

Can the earlier Part I continuity-style visual prior serve as a useful pretraining initialization for the current `multiscale` transfer model on the `observe-then-act` downstream benchmark?

The concrete matched comparison is:

- `scratch`
- `future-dynamics finetune`
- `continuity-encoder finetune`

## Setup

### Continuity pretrain

- Config: [multiscale_continuity_formal_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/configs/multiscale_continuity_formal_20260316_220723.yaml)
- Output: [multiscale_continuity_formal_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723)
- Dataset: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Train/val split: `temperal/data/splits_20260316_220723`
- Pairs: adjacent frame pairs, `pair_step=1`
- Input: grayscale replicated to 3 channels, `64 x 64`
- Encoder width: `c1=c2=c3=8`
- Losses:
  - `lambda_rec=1.0`
  - `lambda_trans=1.0`
  - `lambda_wd=1.0`
  - `lambda_nb=0.1`
- Training:
  - `seed=42`
  - `epochs=5`
  - `batch_size=64`
  - `lr=1e-3`

### Downstream protocol

The downstream protocol is matched to the earlier `obs50` transfer experiments:

- Task: current `observe-then-act` terminal target regression
- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Observation protocol:
  - `observation_length=8`
  - `frame_stride=2`
  - `observation_end_fraction=0.5`
  - `sampling_mode=uniform_visible`
- Model width:
  - `c1=c2=c3=8`
  - `s1=s2=s3=8`
  - `head_hidden_dim=64`
- Downstream training:
  - `epochs=20`
  - `batch_size=8`
  - `lr=1e-3`
  - `seeds=42, 43, 44`

Continuity finetune outputs:

- [20pct aggregate](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_20pct_obs50_3seed_20260316_220723)
- [40pct aggregate](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_40pct_obs50_3seed_20260316_220723)

Reference baselines:

- [scratch 20pct aggregate](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_3seed_20260316_220723)
- [scratch 40pct aggregate](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_40pct_obs50_3seed_20260316_220723)
- [future finetune 20pct aggregate](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_3seed_20260316_220723)
- [future finetune 40pct aggregate](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_40pct_obs50_3seed_20260316_220723)

## Pretrain Result

The formal continuity pretrain converged cleanly:

- best `val_total = 0.0169`
- checkpoint: [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723/best.pt)
- history: [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723/history.csv)
- curve: [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_continuity_formal_20260316_220723/loss_curve.svg)

## Downstream Results

### 20% data, 3 seeds

| Method | Val mean L2 | Val success@0.2 | Test mean L2 | Test success@0.2 |
| --- | ---: | ---: | ---: | ---: |
| scratch | `0.2606 ± 0.0165` | `0.4311 ± 0.0220` | `0.2654 ± 0.0130` | `0.3844 ± 0.0257` |
| future-dynamics finetune | `0.2437 ± 0.0020` | `0.5000 ± 0.0393` | `0.2435 ± 0.0103` | `0.4578 ± 0.0658` |
| continuity finetune | `0.5501 ± 0.4031` | `0.2711 ± 0.1420` | `0.5321 ± 0.3686` | `0.2733 ± 0.1338` |

### 40% data, 3 seeds

| Method | Val mean L2 | Val success@0.2 | Test mean L2 | Test success@0.2 |
| --- | ---: | ---: | ---: | ---: |
| scratch | `0.2103 ± 0.0028` | `0.4800 ± 0.0196` | `0.2070 ± 0.0069` | `0.5156 ± 0.0503` |
| future-dynamics finetune | `0.1829 ± 0.0056` | `0.6000 ± 0.0272` | `0.1789 ± 0.0103` | `0.6156 ± 0.0227` |
| continuity finetune | `0.7325 ± 0.7467` | `0.3578 ± 0.2157` | `0.7852 ± 0.8311` | `0.3711 ± 0.2578` |

## Seed-Level Behavior

Continuity finetune is highly unstable.

At `20%`, seeds `42` and `44` converged to moderate solutions:

- seed `42`: test `mean_l2=0.2681`, `success@0.2=0.3933`
- seed `44`: test `mean_l2=0.2748`, `success@0.2=0.3400`

but seed `43` collapsed badly:

- seed `43`: test `mean_l2=1.0534`, `success@0.2=0.0867`

At `40%`, seeds `42` and `43` were reasonable:

- seed `42`: test `mean_l2=0.1982`, `success@0.2=0.5600`
- seed `43`: test `mean_l2=0.1969`, `success@0.2=0.5467`

but seed `44` failed almost immediately and selected epoch `1` as best:

- seed `44`: test `mean_l2=1.9606`, `success@0.2=0.0067`

Per-seed summaries:

- [20pct per_seed_summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_20pct_obs50_3seed_20260316_220723/per_seed_summary.csv)
- [40pct per_seed_summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_continuity_finetune_40pct_obs50_3seed_20260316_220723/per_seed_summary.csv)

## Conclusion

The current continuity-style encoder pretraining does not provide a competitive downstream initialization under matched `obs50` settings.

What the result supports:

- the continuity pretrain objective itself is trainable
- the continuity-pretrained encoder can sometimes support a usable downstream finetune run

What the result does not support:

- it does not beat `scratch` on 3-seed aggregate
- it is far less stable than `future-dynamics finetune`
- it is not currently a better pretraining recipe for this downstream task

The most likely interpretation is that the current continuity pretrain is too weak or too mismatched as an initialization for the recurrent downstream dynamics stack. The `future-dynamics` pretraining remains the stronger and more reliable prior in the current codebase.

## Most Useful Next Fixes

If we want to rescue this line rather than retire it, the highest-value follow-ups are:

1. replace zero flow with real flow or a learned motion estimate
2. lower downstream LR or use staged unfreezing for continuity finetune
3. test continuity pretraining plus a short downstream warmup with frozen encoder before full finetune
