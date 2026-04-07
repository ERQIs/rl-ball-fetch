# Observe-Then-Act Experiment Report

This report summarizes the recent `observe-then-act` experiments in `temperal/`, with the goal of testing whether a structured spatiotemporal prior is useful for downstream target prediction.

## 1. Goal

The core question is:

- does a structured spatiotemporal representation help downstream target regression beyond standard supervised baselines?
- does it help more clearly in low-data settings?

The current working hypothesis is:

- a model that preserves spatial layout over time and is pretrained with future rollout constraints should learn a stronger temporal prior than a plain framewise encoder plus vector GRU.

## 2. Current Task Definition

All experiments in this report use the current `observe-then-act` target definition implemented in [trajectory_dataset.py](/d:/projects/rl-ball-fetch/temperal/src/datasets/trajectory_dataset.py):

- input: a prefix of the visual trajectory
- output: `target_xy`
- current label source: the last row of `frames.csv`
- current label fields: effectively `(ball_px, ball_py)`

Important caveat:

- this is not yet the true ground-plane landing point `(ball_px, ball_pz)`
- it is better interpreted as the final exported ball position near the end of flight

So the experiments below evaluate whether the model can infer the terminal 2D target used by the current benchmark, not the final intended interception label.

## 3. Dataset And Evaluation Protocol

The main dataset used for the recent formal experiments is:

- dataset root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- collection setting: `360` degree azimuth coverage, radius `0-6`
- split: `700 / 150 / 150`
- image size: `64 x 64`
- images: grayscale inputs

Evaluation metrics:

- `mean_l2`
- `median_l2`
- `success@0.2`

The two main observation protocols used recently are:

| Setting | Observation length | Frame stride | Observation end fraction | Sampling mode | Intended meaning |
| --- | --- | --- | --- | --- | --- |
| `obs50` | `8` | `2` | `0.5` | `uniform_visible` | observe the first half of the trajectory |
| `obs67` | `16` | `1` | `0.67` | `uniform_visible` | align with the formal GRU baseline |

## 4. Models

### 4.1 Single-Frame Baseline

Implementation:

- [single_frame_regressor.py](/d:/projects/rl-ball-fetch/temperal/src/models/single_frame_regressor.py)
- [cnn_encoder.py](/d:/projects/rl-ball-fetch/temperal/src/models/cnn_encoder.py)

Structure:

- `SmallCNN` encoder with three `conv + relu + maxpool` stages
- flatten final feature map
- fully connected layer to hidden feature
- MLP regression head
- only uses the last observed frame

Parameter count:

- `556,098`

Role:

- non-temporal baseline
- tests how much can be inferred from appearance alone

### 4.2 CNN+GRU Baseline

Implementation:

- [cnn_gru_regressor.py](/d:/projects/rl-ball-fetch/temperal/src/models/cnn_gru_regressor.py)

Structure:

- same `SmallCNN` encoder as above
- each frame is encoded into a `128`-dimensional vector
- sequence of vectors is processed by a standard `GRU`
- take the final hidden state
- MLP regression head predicts `target_xy`

Parameter count:

- `655,170`

Role:

- main temporal supervised baseline
- current strongest direct benchmark baseline

### 4.3 Multi-Scale Transfer Model

Implementation:

- [multiscale_transfer_regressor.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_transfer_regressor.py)
- [multiscale_future_dynamics.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_future_dynamics.py)

Structure:

- `MultiScaleEncoder` produces three spatial feature levels `f1 / f2 / f3`
- `MultiScaleDynamics` maintains three hidden states `h1 / h2 / h3`
- each hidden state is updated with a `ConvGRU`-style cell, so time is modeled on spatial feature maps rather than vectors
- final coarsest hidden map `h3` is globally averaged
- MLP head predicts `target_xy`

Default width used here:

- `c1 = c2 = c3 = 8`
- `s1 = s2 = s3 = 8`
- `head_hidden_dim = 64`

Parameter count:

- `22,978`

Interpretation:

- this is not a larger model than the GRU baseline
- it is a smaller but more structured spatiotemporal model

### 4.4 Pretraining Source For Transfer

Transfer uses a pretrained checkpoint from the multi-scale future-dynamics experiment:

- config: [multiscale_future_formal_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_future_formal_seed42_20260316_220723.yaml)
- checkpoint: [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723/best.pt)

Pretraining task summary:

- short contiguous clips
- observed history plus future rollout
- 3D probe supervision on position and velocity
- optional future-frame reconstruction

This is the operational form of the current spatiotemporal prior.

### 4.5 Transfer Modes

The downstream transfer experiments use three initialization modes:

- `scratch`: same multiscale architecture, random initialization
- `frozen`: load pretrained encoder+dynamics, freeze backbone, only train head
- `finetune`: load pretrained encoder+dynamics, then train backbone and head jointly

## 5. Experiments

### 5.1 Smoke Sanity Check On The Hard Dataset

Purpose:

- verify that temporal information matters on the hardest dataset before running formal transfer

Dataset:

- `20260316_220723`

Protocol:

- `observation_end_fraction = 0.67`
- single-frame uses `8` frames with `stride=2`
- GRU uses `16` frames with `stride=1`

Results on validation split:

| Model | Epochs | mean_l2 | median_l2 | success@0.2 |
| --- | --- | --- | --- | --- |
| `single_frame` | `1` | `0.63026` | `0.33193` | `0.30000` |
| `cnn_gru` | `10` | `0.15115` | `0.12964` | `0.76000` |

Interpretation:

- temporal information is clearly useful
- a purely visual single-frame predictor is much weaker than a temporal model

### 5.2 Formal CNN+GRU Baseline

Purpose:

- establish the main supervised reference baseline

Config:

- [gru_formal_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/gru_formal_seed42_20260316_220723.yaml)

Training setup:

- full train split
- `observation_length = 16`
- `frame_stride = 1`
- `observation_end_fraction = 0.67`
- `sampling_mode = uniform_visible`
- seed `42`
- epochs `30`
- batch size `8`
- lr `5e-4`

Seed-42 result:

| Model | Best epoch | Val mean_l2 | Val success@0.2 | Test mean_l2 | Test success@0.2 |
| --- | --- | --- | --- | --- | --- |
| `cnn_gru` | `28` | `0.10445` | `0.91333` | `0.12321` | `0.82000` |

3-seed aggregate:

| Seeds | Val mean_l2 | Val success@0.2 | Test mean_l2 | Test success@0.2 |
| --- | --- | --- | --- | --- |
| `42,43,44` | `0.11061 ± 0.00768` | `0.88222 ± 0.03095` | `0.12415 ± 0.00826` | `0.84000 ± 0.02373` |

Artifacts:

- seed-42 output: [gru_formal_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_seed42_20260316_220723)
- 3-seed aggregate: [gru_formal_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/gru_formal_3seed_20260316_220723)

### 5.3 Data-Efficiency Transfer Experiment

Purpose:

- test whether the pretrained spatiotemporal prior helps across matched data regimes under one fixed observation protocol

Shared training setup:

- train fractions: `0.2 / 0.4 / 1.0`
- `observation_length = 8`
- `frame_stride = 2`
- `observation_end_fraction = 0.5`
- seeds: `42 / 43 / 44`
- epochs `20`
- batch size `8`
- lr `1e-3`

Configs:

- scratch: [multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723.yaml)
- frozen: [multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723.yaml)
- finetune: [multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723.yaml)

Aggregate results across 3 seeds:

| Train fraction | Init mode | Val mean_l2 | Val success@0.2 | Test mean_l2 | Test success@0.2 |
| --- | --- | --- | --- | --- | --- |
| `20%` | `scratch` | `0.26063 ± 0.01654` | `0.43111 ± 0.02200` | `0.26541 ± 0.01298` | `0.38444 ± 0.02572` |
| `20%` | `frozen` | `0.28110 ± 0.00563` | `0.39333 ± 0.00000` | `0.30404 ± 0.00155` | `0.34444 ± 0.01571` |
| `20%` | `finetune` | `0.24372 ± 0.00197` | `0.50000 ± 0.03925` | `0.24348 ± 0.01034` | `0.45778 ± 0.06585` |
| `40%` | `scratch` | `0.21033 ± 0.00281` | `0.48000 ± 0.01963` | `0.20696 ± 0.00689` | `0.51556 ± 0.05028` |
| `40%` | `frozen` | `0.25724 ± 0.00371` | `0.42444 ± 0.01571` | `0.26297 ± 0.00575` | `0.43778 ± 0.03281` |
| `40%` | `finetune` | `0.18293 ± 0.00558` | `0.60000 ± 0.02722` | `0.17889 ± 0.01028` | `0.61556 ± 0.02266` |
| `100%` | `scratch` | `0.17348 ± 0.00942` | `0.62889 ± 0.02740` | `0.16325 ± 0.00668` | `0.67111 ± 0.03705` |
| `100%` | `frozen` | `0.24828 ± 0.00713` | `0.42222 ± 0.02200` | `0.25294 ± 0.00841` | `0.44889 ± 0.02455` |
| `100%` | `finetune` | `0.15534 ± 0.00942` | `0.72889 ± 0.03624` | `0.14934 ± 0.00527` | `0.72000 ± 0.03031` |

Artifacts:

- scratch: [multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_seed42_20260316_220723)
- frozen: [multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_seed42_20260316_220723)
- finetune: [multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_seed42_20260316_220723)
- combined 3-seed summary: [summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_obs50_data_efficiency_3seed_20260316_220723/summary.csv)
- per-combination aggregates:
  - [multiscale_transfer_scratch_20pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_3seed_20260316_220723)
  - [multiscale_transfer_frozen_20pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_3seed_20260316_220723)
  - [multiscale_transfer_finetune_20pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_3seed_20260316_220723)
  - [multiscale_transfer_scratch_40pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_40pct_obs50_3seed_20260316_220723)
  - [multiscale_transfer_frozen_40pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_40pct_obs50_3seed_20260316_220723)
  - [multiscale_transfer_finetune_40pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_40pct_obs50_3seed_20260316_220723)
  - [multiscale_transfer_scratch_100pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_100pct_obs50_3seed_20260316_220723)
  - [multiscale_transfer_frozen_100pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_100pct_obs50_3seed_20260316_220723)
  - [multiscale_transfer_finetune_100pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_100pct_obs50_3seed_20260316_220723)

Interpretation:

- `frozen` is consistently the weakest, even as data increases
- `finetune` is already best at `20%`, and the gap becomes clearer at `40%` and `100%`
- `scratch` improves smoothly with more data, but `finetune` improves faster and stays ahead
- under the matched `obs50` protocol, the current pretrained prior is useful mainly as a finetuned initialization, not as a frozen feature extractor

### 5.4 Full-Data Transfer Aligned With The Formal GRU Protocol

Purpose:

- compare `scratch` and `finetune` under the same observation protocol as the formal GRU baseline

Shared training setup:

- full train split
- `observation_length = 16`
- `frame_stride = 1`
- `observation_end_fraction = 0.67`
- `sampling_mode = uniform_visible`
- seed `42`
- epochs `30`
- batch size `8`
- lr `1e-3`

Configs:

- scratch: [multiscale_transfer_scratch_full_obs67_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_scratch_full_obs67_seed42_20260316_220723.yaml)
- finetune: [multiscale_transfer_finetune_full_obs67_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_transfer_finetune_full_obs67_seed42_20260316_220723.yaml)

Results:

| Model | Init mode | Best epoch | Val mean_l2 | Val success@0.2 | Test mean_l2 | Test success@0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| `multiscale_transfer` | `scratch` | `18` | `0.16443` | `0.65333` | `0.16549` | `0.66667` |
| `multiscale_transfer` | `finetune` | `28` | `0.10341` | `0.90667` | `0.11866` | `0.86000` |

Artifacts:

- scratch: [multiscale_transfer_scratch_full_obs67_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_full_obs67_seed42_20260316_220723)
- finetune: [multiscale_transfer_finetune_full_obs67_seed42_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_full_obs67_seed42_20260316_220723)

Direct comparison with the formal GRU baseline:

| Model | Val mean_l2 | Val success@0.2 | Test mean_l2 | Test success@0.2 |
| --- | --- | --- | --- | --- |
| `cnn_gru` seed42 | `0.10445` | `0.91333` | `0.12321` | `0.82000` |
| `multiscale_transfer scratch` | `0.16443` | `0.65333` | `0.16549` | `0.66667` |
| `multiscale_transfer finetune` | `0.10341` | `0.90667` | `0.11866` | `0.86000` |

Interpretation:

- `scratch` is clearly worse than the GRU baseline
- `finetune` reaches essentially the same validation level as the GRU baseline
- on this run, `finetune` slightly exceeds the GRU baseline on test

## 6. Main Findings

### 6.1 Temporal Modeling Matters

Evidence:

- on the hardest dataset, `single_frame` is far behind `cnn_gru`
- this confirms that the task cannot be solved well from a single image alone

### 6.2 The Multi-Scale Model Is Not Winning Because It Is Bigger

Evidence:

- `cnn_gru`: `655,170` parameters
- `multiscale_transfer`: `22,978` parameters

So:

- the multiscale model is much smaller
- the relevant difference is inductive bias and initialization, not brute-force capacity

### 6.3 The Current Prior Helps Most When Used As A Finetuned Initialization

Evidence:

- `frozen` underperforms on the `20%` experiment
- `scratch` underperforms on the `full + obs67` formal comparison
- `finetune` is the only transfer setting that becomes competitive with the formal GRU baseline

Interpretation:

- the current pretrained representation is not yet a strong universal fixed feature extractor
- but it is a useful initialization for downstream optimization

### 6.4 Current Evidence For Data Efficiency Is Now Stronger

What is supported:

- at `20%`, `finetune` is already ahead of `scratch` in the 3-seed aggregate
- at `40%`, the gap becomes clearer
- at `100%`, `finetune` remains clearly better than `scratch`

What is still limited:

- the evidence is strong within the current `obs50` protocol, but still based on the current terminal `(x, y)` target
- the claim should be strengthened further by repeating the same analysis on the final landing/interception label

### 6.5 Current Best Evidence For The Prior

The strongest current result is:

- under the formal `obs67` protocol, `multiscale_transfer finetune` matches or slightly exceeds the formal GRU baseline while using a much smaller downstream model

This is the cleanest current argument that the structured spatiotemporal prior is meaningful.

## 7. Current Limitations

- the label is still the final exported `(ball_px, ball_py)` target, not the true landing point
- transfer has only been tested with one seed so far
- the strongest data-efficiency evidence currently comes from the `obs50` protocol rather than the formal `obs67` protocol
- the `frozen` result suggests the representation itself is not yet fully task-ready without adaptation

## 8. Recommended Next Steps

- run `10%` as an even harder low-data point
- optionally repeat the `obs67` protocol with multiple seeds for `scratch` and `finetune`
- repeat the transfer comparison with at least `3` seeds
- change the downstream label from terminal `(x, y)` to the real landing target
- keep `cnn_gru` as the direct supervised baseline
- use `multiscale_transfer finetune` as the main prior-based model

## 9. Suggested One-Sentence Summary

The recent observe-then-act experiments show that a pretrained multi-scale ConvGRU-style spatiotemporal representation is not yet a strong frozen feature extractor, but it becomes highly competitive after finetuning and can match or slightly surpass the formal CNN+GRU baseline with a much smaller downstream model.
