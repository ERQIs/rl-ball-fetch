# Formal Results

This file tracks longer-running baseline experiments intended to serve as formal reference runs.

## Summary

| Exp ID | Dataset | Model | Seed | Epochs | Best Epoch | Val mean_l2 | Val success@0.2 | Test mean_l2 | Test success@0.2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gru_formal_3seed_20260316_220723` | `20260316_220723` | `cnn_gru` | `42,43,44` | `30` | `28 / 25 / 30` | `0.1106 ± 0.0077` | `0.8822 ± 0.0310` | `0.1241 ± 0.0083` | `0.8400 ± 0.0237` |
| `gru_formal_seed42_20260316_220723` | `20260316_220723` | `cnn_gru` | `42` | `30` | `28` | `0.1044` | `0.9133` | `0.1232` | `0.8200` |
| `gru_formal_seed43_20260316_220723` | `20260316_220723` | `cnn_gru` | `43` | `30` | `25` | `0.1059` | `0.8933` | `0.1145` | `0.8733` |
| `gru_formal_seed44_20260316_220723` | `20260316_220723` | `cnn_gru` | `44` | `30` | `30` | `0.1214` | `0.8400` | `0.1347` | `0.8267` |
| `multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723` | `20260316_220723` | `multiscale_transfer_scratch` | `42` | `20` | `20` | `0.2464` | `0.4467` | `0.2520` | `0.4200` |
| `multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723` | `20260316_220723` | `multiscale_transfer_frozen` | `42` | `20` | `20` | `0.2891` | `0.3933` | `0.3049` | `0.3667` |
| `multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723` | `20260316_220723` | `multiscale_transfer_finetune` | `42` | `20` | `20` | `0.2455` | `0.4467` | `0.2530` | `0.3667` |

---

## Queued Experiments

### `multiscale_transfer_20pct_obs50_seed42_suite`

This experiment group has now been run to test whether the multiscale future-dynamics representation improves downstream regression in a lower-data regime. Detailed results are recorded below.

- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Split files:
  - train: `temperal/data/splits_20260316_220723/train.txt`
  - val: `temperal/data/splits_20260316_220723/val.txt`
  - test: `temperal/data/splits_20260316_220723/test.txt`
- Downstream task: terminal target regression on `target_xy`
- Data-efficiency setting:
  - `train_fraction = 0.2`
  - val/test use full split
- Observation setting:
  - `observation_length = 8`
  - `frame_stride = 2`
  - `sampling_mode = uniform_visible`
  - `observation_end_fraction = 0.5`
- Backbone width:
  - `c1 = c2 = c3 = 8`
  - `s1 = s2 = s3 = 8`
- Head:
  - `head_hidden_dim = 64`
  - output dim = `2`
- Training:
  - seed: `42`
  - epochs: `20`
  - batch size: `8`
  - lr: `1e-3`
  - device: `auto`
- Modes:
  - scratch: [multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723.yaml)
  - frozen: [multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723.yaml)
  - finetune: [multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723.yaml)
- Pretrained checkpoint for transfer:
  - [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723/best.pt)

Expected saved artifacts per run:

- `best.pt`
- `last.pt`
- `history.csv`
- `loss_curve.svg`
- `success_curve.svg`
- `best_val_metrics.json`
- `test_metrics.json`
- `resolved_config.yaml`
- TensorBoard events under `tb/`

---

### `multiscale_transfer_full_obs67_seed42_suite`

This experiment group is queued to align the multiscale transfer setting with the formal GRU baseline observation protocol.

- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Split files:
  - train: `temperal/data/splits_20260316_220723/train.txt`
  - val: `temperal/data/splits_20260316_220723/val.txt`
  - test: `temperal/data/splits_20260316_220723/test.txt`
- Downstream task: terminal target regression on `target_xy`
- Data setting:
  - full train split
  - val/test use full split
- Observation setting:
  - `observation_length = 16`
  - `frame_stride = 1`
  - `sampling_mode = uniform_visible`
  - `observation_end_fraction = 0.67`
- Backbone width:
  - `c1 = c2 = c3 = 8`
  - `s1 = s2 = s3 = 8`
- Head:
  - `head_hidden_dim = 64`
  - output dim = `2`
- Training:
  - seed: `42`
  - epochs: `30`
  - batch size: `8`
  - lr: `1e-3`
  - device: `auto`
- Modes:
  - scratch: [multiscale_transfer_scratch_full_obs67_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_scratch_full_obs67_seed42_20260316_220723.yaml)
  - finetune: [multiscale_transfer_finetune_full_obs67_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_finetune_full_obs67_seed42_20260316_220723.yaml)
- Pretrained checkpoint for transfer:
  - [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723/best.pt)

Expected saved artifacts per run:

- `best.pt`
- `last.pt`
- `history.csv`
- `loss_curve.svg`
- `success_curve.svg`
- `best_val_metrics.json`
- `test_metrics.json`
- `resolved_config.yaml`
- TensorBoard events under `tb/`

---

### `multiscale_future_formal_seed42_20260316_220723`

This experiment is queued as the first long-run trial for the multi-scale future-dynamics line.

- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Split files:
  - train: `temperal/data/splits_20260316_220723/train.txt`
  - val: `temperal/data/splits_20260316_220723/val.txt`
  - test: `temperal/data/splits_20260316_220723/test.txt`
- Task type: clip-based future dynamics with 3D probes
- Probe targets:
  - position: `(ball_px, ball_py, ball_pz)`
  - velocity: `(ball_vx, ball_vy, ball_vz)`
- Clip config:
  - `seq_len = 12`
  - `history_len = 8`
  - `future_len = 4`
  - `frame_stride = 2`
  - `clip_start_stride = 6`
- Model width:
  - `c1 = c2 = c3 = 8`
  - `s1 = s2 = s3 = 8`
  - `decoder_proj_ch = 8`
  - `decoder_hidden_ch = 8`
- Reconstruction:
  - `enable_frame_recon = true`
  - `future_recon_len = 4`
- Loss weights:
  - `lambda_frame = 1.0`
  - `lambda_p = 1.0`
  - `lambda_v = 0.2`
- Training:
  - seed: `42`
  - epochs: `20`
  - batch size: `8`
  - lr: `3e-4`
  - device: `auto`
- Config: [multiscale_future_formal_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_future_formal_seed42_20260316_220723.yaml)
- Output dir: [multiscale_future_formal_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723)

Expected saved artifacts:

- `best.pt`
- `last.pt`
- `history.csv`
- `loss_curve.svg`
- TensorBoard events under `tb/`

Note:

- This experiment does not define a catch `success rate` yet.
- The main saved intermediate signals are train/val losses and probe MAEs.

---

## Experiment Group: `multiscale_transfer_20pct_obs50_seed42_suite`

### Shared Setup

- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Split files:
  - train: `temperal/data/splits_20260316_220723/train.txt`
  - val: `temperal/data/splits_20260316_220723/val.txt`
  - test: `temperal/data/splits_20260316_220723/test.txt`
- Downstream task: terminal target regression on `target_xy`
- Data-efficiency setting:
  - `train_fraction = 0.2`
  - val/test use full split
- Observation setting:
  - `observation_length = 8`
  - `frame_stride = 2`
  - `sampling_mode = uniform_visible`
  - `observation_end_fraction = 0.5`
- Backbone width:
  - `c1 = c2 = c3 = 8`
  - `s1 = s2 = s3 = 8`
- Head:
  - `head_hidden_dim = 64`
  - output dim = `2`
- Training:
  - seed: `42`
  - epochs: `20`
  - batch size: `8`
  - lr: `1e-3`
  - device: `auto`
- Transfer source checkpoint:
  - [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723/best.pt)

### Main Result

- In this `20%` data regime with `observation_end_fraction = 0.5`, `scratch` and `finetune` were nearly tied on validation.
- `frozen` underperformed both, so this pretrained representation is not yet a strong fixed feature extractor for the current downstream target.
- `finetune` achieved the best validation `mean_l2`, while `scratch` achieved the best test `success@0.2`.

### Experiment: `multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723`

#### Best Validation Snapshot

- Best epoch: `20`
- train_loss: `0.01347`
- val_loss: `0.02326`
- val_mean_l2: `0.24640`
- val_median_l2: `0.22076`
- val_success@0.2: `0.44667`

#### Test Result Using `best.pt`

- loss: `0.02073`
- mean_l2: `0.25204`
- median_l2: `0.23054`
- success_at_0.2: `0.42000`

#### Artifacts

- Config: [multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723.yaml)
- Output dir: [multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723)
- Resolved config: [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/resolved_config.yaml)
- Best checkpoint: [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/best.pt)
- Last checkpoint: [last.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/last.pt)
- Best val metrics: [best_val_metrics.json](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/best_val_metrics.json)
- Test metrics: [test_metrics.json](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/test_metrics.json)
- Training history: [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/history.csv)
- Loss curve: [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/loss_curve.svg)
- Success curve: [success_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/success_curve.svg)
- TensorBoard: [tb](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723/tb)

### Experiment: `multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723`

#### Best Validation Snapshot

- Best epoch: `20`
- train_loss: `0.04865`
- val_loss: `0.03175`
- val_mean_l2: `0.28906`
- val_median_l2: `0.25138`
- val_success@0.2: `0.39333`

#### Test Result Using `best.pt`

- loss: `0.03338`
- mean_l2: `0.30491`
- median_l2: `0.25807`
- success_at_0.2: `0.36667`

#### Artifacts

- Config: [multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723.yaml)
- Output dir: [multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723)
- Resolved config: [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/resolved_config.yaml)
- Best checkpoint: [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/best.pt)
- Last checkpoint: [last.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/last.pt)
- Best val metrics: [best_val_metrics.json](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/best_val_metrics.json)
- Test metrics: [test_metrics.json](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/test_metrics.json)
- Training history: [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/history.csv)
- Loss curve: [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/loss_curve.svg)
- Success curve: [success_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/success_curve.svg)
- TensorBoard: [tb](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723/tb)

### Experiment: `multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723`

#### Best Validation Snapshot

- Best epoch: `20`
- train_loss: `0.01876`
- val_loss: `0.02416`
- val_mean_l2: `0.24551`
- val_median_l2: `0.20996`
- val_success@0.2: `0.44667`

#### Test Result Using `best.pt`

- loss: `0.02060`
- mean_l2: `0.25302`
- median_l2: `0.24811`
- success_at_0.2: `0.36667`

#### Artifacts

- Config: [multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723.yaml)
- Output dir: [multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723)
- Resolved config: [resolved_config.yaml](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/resolved_config.yaml)
- Best checkpoint: [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/best.pt)
- Last checkpoint: [last.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/last.pt)
- Best val metrics: [best_val_metrics.json](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/best_val_metrics.json)
- Test metrics: [test_metrics.json](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/test_metrics.json)
- Training history: [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/history.csv)
- Loss curve: [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/loss_curve.svg)
- Success curve: [success_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/success_curve.svg)
- TensorBoard: [tb](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723/tb)

---

## Aggregate: `gru_formal_3seed_20260316_220723`

### Setup

- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Data setting: `360` degrees, radius `0-6`
- Model: `cnn_gru`
- Seeds: `42, 43, 44`
- Epochs per run: `30`
- Observation setting: first `67%` of the trajectory, uniformly sampled sequence

### Aggregate Metrics

- val_mean_l2: `0.11061 ± 0.00768`
- val_success@0.2: `0.88222 ± 0.03095`
- test_mean_l2: `0.12415 ± 0.00826`
- test_success@0.2: `0.84000 ± 0.02373`

### Per-Seed Best Epochs

- seed `42`: best epoch `28`
- seed `43`: best epoch `25`
- seed `44`: best epoch `30`

### Aggregate Artifacts

- Output dir: [gru_formal_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_3seed_20260316_220723)
- Per-seed summary: [per_seed_summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_3seed_20260316_220723/per_seed_summary.csv)
- Aggregate summary: [aggregate_summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_3seed_20260316_220723/aggregate_summary.csv)
- Aggregate history: [aggregate_history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_3seed_20260316_220723/aggregate_history.csv)
- Average loss curve: [aggregate_loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_3seed_20260316_220723/aggregate_loss_curve.svg)
- Average success curve: [aggregate_success_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_3seed_20260316_220723/aggregate_success_curve.svg)

---

## Experiment: `gru_formal_seed42_20260316_220723`

### Setup

- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Data setting: `360` degrees, radius `0-6`
- Split: `700 / 150 / 150`
- Model: `cnn_gru`
- Seed: `42`
- Epochs: `30`
- Observation setting: first `67%` of the trajectory, uniformly sampled sequence

### Best Validation Snapshot

- Best epoch: `28`
- train_loss: `0.00353`
- val_loss: `0.00416`
- val_mean_l2: `0.10445`
- val_median_l2: `0.09016`
- val_success@0.2: `0.91333`

### Test Result Using `best.pt`

- mean_l2: `0.12321`
- median_l2: `0.10419`
- success_at_0.2: `0.82000`

### Artifacts

- Config: [gru_formal_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/gru_formal_seed42_20260316_220723.yaml)
- Output dir: [gru_formal_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed42_20260316_220723)
- Best checkpoint: [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed42_20260316_220723/best.pt)
- Last checkpoint: [last.pt](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed42_20260316_220723/last.pt)
- Training history: [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed42_20260316_220723/history.csv)
- Loss curve: [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed42_20260316_220723/loss_curve.svg)
- Success curve: [success_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed42_20260316_220723/success_curve.svg)

### Notes

- Compared with the shorter 10-epoch smoke baseline on the same dataset, the longer run improved both validation and test performance.
- The validation curve remained somewhat noisy, but the run continued to improve through late epochs rather than clearly overfitting early.

---

## Experiment: `gru_formal_seed43_20260316_220723`

### Setup

- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Data setting: `360` degrees, radius `0-6`
- Split: `700 / 150 / 150`
- Model: `cnn_gru`
- Seed: `43`
- Epochs: `30`
- Observation setting: first `67%` of the trajectory, uniformly sampled sequence

### Best Validation Snapshot

- Best epoch: `25`
- val_loss: `0.00443`
- val_mean_l2: `0.10595`
- val_median_l2: `0.08683`
- val_success@0.2: `0.89333`

### Test Result Using `best.pt`

- mean_l2: `0.11453`
- median_l2: `0.09702`
- success_at_0.2: `0.87333`

### Artifacts

- Config: [gru_formal_seed43_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/gru_formal_seed43_20260316_220723.yaml)
- Output dir: [gru_formal_seed43_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed43_20260316_220723)

---

## Experiment: `gru_formal_seed44_20260316_220723`

### Setup

- Dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- Data setting: `360` degrees, radius `0-6`
- Split: `700 / 150 / 150`
- Model: `cnn_gru`
- Seed: `44`
- Epochs: `30`
- Observation setting: first `67%` of the trajectory, uniformly sampled sequence

### Best Validation Snapshot

- Best epoch: `30`
- val_loss: `0.00560`
- val_mean_l2: `0.12144`
- val_median_l2: `0.10018`
- val_success@0.2: `0.84000`

### Test Result Using `best.pt`

- mean_l2: `0.13470`
- median_l2: `0.11495`
- success_at_0.2: `0.82667`

### Artifacts

- Config: [gru_formal_seed44_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/gru_formal_seed44_20260316_220723.yaml)
- Output dir: [gru_formal_seed44_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed44_20260316_220723)

---

## Template For Future Runs

Copy the block below when adding another formal experiment:

```md
## Experiment: `<exp_id>`

### Setup

- Dataset root:
- Data setting:
- Split:
- Model:
- Seed:
- Epochs:
- Observation setting:

### Best Validation Snapshot

- Best epoch:
- train_loss:
- val_loss:
- val_mean_l2:
- val_median_l2:
- val_success@0.2:

### Test Result Using `best.pt`

- mean_l2:
- median_l2:
- success_at_0.2:

### Artifacts

- Config:
- Output dir:
- Best checkpoint:
- Last checkpoint:
- Training history:
- Loss curve:
- Success curve:

### Notes

- 
```
