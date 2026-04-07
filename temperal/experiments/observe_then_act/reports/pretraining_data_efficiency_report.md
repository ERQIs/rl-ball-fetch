# Pretraining Data Efficiency Report

This note focuses on one question only:

- does the current multiscale spatiotemporal pretraining bring downstream data-efficiency gains?

## 1. Question And Short Answer

Question:

- compared with training the same downstream architecture from scratch, does pretraining help the model reach better performance with the same amount of labeled data?

Short answer:

- yes, in the current `obs50` protocol, pretraining helps when used as `finetune`
- no, pretraining does not help when used as a frozen backbone
- the evidence is now reasonably strong because it holds across `20% / 40% / 100%` data fractions and `3` random seeds

## 2. Task Definition

These experiments use the current `observe-then-act` downstream benchmark in [trajectory_dataset.py](/d:/projects/rl-ball-fetch/temperal/src/datasets/trajectory_dataset.py).

Current target:

- `target_xy`
- read from the last row of `frames.csv`
- effectively the final exported `(ball_px, ball_py)` position

Important caveat:

- this is not yet the true landing point on the ground plane
- so the conclusions below are about data efficiency on the current benchmark target, not yet about the final interception label

## 3. Dataset And Split

Dataset:

- root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- setting: `360` degree coverage, radius `0-6`

Split:

- train: `700`
- val: `150`
- test: `150`

## 4. Experimental Protocol

To answer the data-efficiency question cleanly, all compared runs use the same downstream protocol:

- `observation_length = 8`
- `frame_stride = 2`
- `observation_end_fraction = 0.5`
- `sampling_mode = uniform_visible`
- image size: `64`
- epochs: `20`
- batch size: `8`
- learning rate: `1e-3`
- seeds: `42 / 43 / 44`

Train fractions:

- `20%`
- `40%`
- `100%`

This keeps the observation protocol fixed and changes only the amount of labeled training data.

## 5. Compared Models

### 5.1 Shared Downstream Architecture

All three compared transfer modes use the same downstream architecture:

- [multiscale_transfer_regressor.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_transfer_regressor.py)
- [multiscale_future_dynamics.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_future_dynamics.py)

Structure:

- multi-scale visual encoder
- multi-scale hidden state updated by `ConvGRU`-style dynamics
- global average pooling on the coarsest hidden state
- MLP regression head for `target_xy`

Width used here:

- `c1 = c2 = c3 = 8`
- `s1 = s2 = s3 = 8`
- `head_hidden_dim = 64`

Parameter count:

- about `22,978`

### 5.2 Initialization Modes

The only difference between the three compared methods is initialization and whether the backbone is trainable.

`scratch`

- same downstream architecture
- all weights randomly initialized

`frozen`

- load pretrained `encoder + dynamics`
- freeze the backbone
- only train the regression head

`finetune`

- load pretrained `encoder + dynamics`
- train both backbone and regression head jointly

## 6. Pretraining Source

The pretrained weights come from the multiscale future-dynamics model:

- config: [multiscale_future_formal_seed42_20260316_220723.yaml](/d:/projects/rl-ball-fetch/temperal/configs/multiscale_future_formal_seed42_20260316_220723.yaml)
- checkpoint: [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_future_formal_seed42_20260316_220723/best.pt)

That pretraining stage uses:

- short contiguous clips
- multi-scale `ConvGRU` temporal state
- 3D position and velocity probe supervision
- future rollout
- future-frame reconstruction

## 7. Results

3-seed aggregate summary:

- source file: [summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_obs50_data_efficiency_3seed_20260316_220723/summary.csv)

### 7.1 Validation Results

| Train fraction | Scratch mean_l2 | Frozen mean_l2 | Finetune mean_l2 | Scratch success@0.2 | Frozen success@0.2 | Finetune success@0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| `20%` | `0.2606 ± 0.0165` | `0.2811 ± 0.0056` | `0.2437 ± 0.0020` | `0.4311 ± 0.0220` | `0.3933 ± 0.0000` | `0.5000 ± 0.0393` |
| `40%` | `0.2103 ± 0.0028` | `0.2572 ± 0.0037` | `0.1829 ± 0.0056` | `0.4800 ± 0.0196` | `0.4244 ± 0.0157` | `0.6000 ± 0.0272` |
| `100%` | `0.1735 ± 0.0094` | `0.2483 ± 0.0071` | `0.1553 ± 0.0094` | `0.6289 ± 0.0274` | `0.4222 ± 0.0220` | `0.7289 ± 0.0362` |

### 7.2 Test Results

| Train fraction | Scratch mean_l2 | Frozen mean_l2 | Finetune mean_l2 | Scratch success@0.2 | Frozen success@0.2 | Finetune success@0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| `20%` | `0.2654 ± 0.0130` | `0.3040 ± 0.0016` | `0.2435 ± 0.0103` | `0.3844 ± 0.0257` | `0.3444 ± 0.0157` | `0.4578 ± 0.0658` |
| `40%` | `0.2070 ± 0.0069` | `0.2630 ± 0.0057` | `0.1789 ± 0.0103` | `0.5156 ± 0.0503` | `0.4378 ± 0.0328` | `0.6156 ± 0.0227` |
| `100%` | `0.1632 ± 0.0067` | `0.2529 ± 0.0084` | `0.1493 ± 0.0053` | `0.6711 ± 0.0371` | `0.4489 ± 0.0245` | `0.7200 ± 0.0303` |

## 8. Interpretation

### 8.1 Does Pretraining Help?

If pretraining is used as `finetune`, the answer is yes.

Evidence:

- at `20%`, `finetune` is already better than `scratch`
- at `40%`, the margin becomes larger
- at `100%`, `finetune` still stays ahead

This holds on both:

- `mean_l2`
- `success@0.2`

### 8.2 Does Frozen Transfer Help?

No.

Evidence:

- `frozen` is consistently the weakest of the three
- it is worse than `scratch` at `20%`, `40%`, and `100%`

So the current pretrained representation is not yet strong enough to be used as a fixed feature extractor for this downstream task.

### 8.3 Is This Really A Data-Efficiency Gain?

Within the current benchmark setup, yes.

Why:

- data efficiency means getting better downstream performance with the same amount of labeled data
- at each matched data fraction, `finetune` beats `scratch`
- therefore the pretrained model is extracting more value from the same labeled dataset size

This is strongest at:

- `20%`: test `0.2435` vs `0.2654`, success `0.4578` vs `0.3844`
- `40%`: test `0.1789` vs `0.2070`, success `0.6156` vs `0.5156`

### 8.4 How Strong Is The Evidence?

The evidence is stronger than before because:

- it uses `3` seeds instead of just one
- it covers multiple data fractions
- the trend is consistent rather than accidental

But it is still not the final word because:

- the downstream target is still the current terminal `(x, y)` benchmark target
- it is not yet the real landing/interception label

## 9. Main Conclusion

The current multiscale spatiotemporal pretraining **does bring downstream data-efficiency gains**, but mainly in the `finetune` setting.

More precisely:

- `finetune > scratch` at `20% / 40% / 100%`
- `frozen < scratch` at `20% / 40% / 100%`

So the best-supported conclusion is:

- pretraining is useful as a better initialization for downstream learning
- pretraining is not yet useful as a frozen generic feature extractor

## 10. Practical Takeaway

If the question is "what should we use going forward?" then the answer is:

- keep `scratch` as the no-pretraining control
- stop emphasizing `frozen` as the main transfer story
- use `finetune` as the primary evidence that the spatiotemporal prior helps

## 11. Supporting Artifacts

Combined summary:

- [summary.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_obs50_data_efficiency_3seed_20260316_220723/summary.csv)

Per-combination 3-seed aggregates:

- [multiscale_transfer_scratch_20pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_20pct_obs50_3seed_20260316_220723)
- [multiscale_transfer_frozen_20pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_20pct_obs50_3seed_20260316_220723)
- [multiscale_transfer_finetune_20pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_20pct_obs50_3seed_20260316_220723)
- [multiscale_transfer_scratch_40pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_40pct_obs50_3seed_20260316_220723)
- [multiscale_transfer_frozen_40pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_40pct_obs50_3seed_20260316_220723)
- [multiscale_transfer_finetune_40pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_40pct_obs50_3seed_20260316_220723)
- [multiscale_transfer_scratch_100pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_scratch_100pct_obs50_3seed_20260316_220723)
- [multiscale_transfer_frozen_100pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_frozen_100pct_obs50_3seed_20260316_220723)
- [multiscale_transfer_finetune_100pct_obs50_3seed_20260316_220723](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_transfer_finetune_100pct_obs50_3seed_20260316_220723)

## 12. One-Sentence Claim

Under a matched `observe-then-act` protocol and across `3` seeds, the current multiscale pretraining consistently improves downstream data efficiency when used for finetuning, but does not provide gains as a frozen backbone.
