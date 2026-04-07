# Multi-Scale Future Dynamics Experiment v1

## 1. Purpose

This document specifies the first transfer experiment that adapts the external project under:

`D:\projects\re-ball-fetch-other-people\code(1)\code`

to the current ball-catching dataset in `temperal/`.

The goal is not to replace the existing landing-regression baseline immediately, but to build a stronger temporal visual model that:

- consumes a short history of frames,
- rolls its hidden state into the future,
- optionally reconstructs future frames,
- predicts physical state probes at every step.

This experiment is intended as a representation-learning / temporal-modeling benchmark that can later be connected to downstream landing prediction or control.

---

## 2. Reference Method Summary

The external project uses four main parts:

1. `MultiScaleEncoder`
2. `MultiScaleDynamics` with multi-scale `ConvGRU`
3. `PyramidDecoder` for future-frame rollout
4. probe heads on the coarsest hidden state

The probe design is the most important part for transfer:

- pool the coarsest hidden map `h3`
- apply linear heads to predict physical quantities
- supervise both history steps and rollout steps

This is the core idea preserved in the current transfer plan.

---

## 3. High-Level Design Choice

### 3.1 Default Stance

The default experiment should include height information.

Reasoning:

- height is visually salient and easier to infer directly from images,
- height helps disambiguate the same apparent planar location under different depths,
- height is important for future time-to-impact reasoning,
- even if the current simplified task does not yet use full interception timing, keeping height in the state makes the representation more future-proof.

### 3.2 Default Probe Target

The default probe target will therefore be:

- position: `(ball_px, ball_py, ball_pz)`
- velocity: `(ball_vx, ball_vy, ball_vz)`

This makes both heads 3D regression heads.

### 3.3 Important Ablation

A later ablation should also be run with:

- position: `(ball_px, ball_pz)`
- velocity: `(ball_vx, ball_vz)`

This 2D planar version is useful as a comparison, but it is not the default.

---

## 4. Dataset Setting

### 4.1 Recommended Dataset

The first transfer experiment should use:

`D:\projects\rl-ball-fetch\ball_fetch\vis_backbone\datasets\manual_capture\20260316_220723`

Reasoning:

- it is currently the hardest collected dataset,
- it contains `1000` trajectories,
- it is already used by the strongest formal baseline,
- it provides the clearest comparison point.

### 4.2 Split

Use the existing split:

- train: `temperal/data/splits_20260316_220723/train.txt`
- val: `temperal/data/splits_20260316_220723/val.txt`
- test: `temperal/data/splits_20260316_220723/test.txt`

### 4.3 Clip Construction

Unlike the current observe-then-act regression baseline, this model should train on short contiguous clips, not sparse prefix samples across the whole trajectory.

Recommended default:

- `seq_len = 12`
- `history_len = 8`
- `future_len = 4`
- `frame_stride = 2`

Interpretation:

- the model receives `8` observed steps,
- then rolls out `4` future steps,
- the total physical coverage is `24` raw frames because of stride `2`.

### 4.4 Clip Sampling Rule

For each trajectory:

- sample a valid contiguous window of length `seq_len * frame_stride`,
- decode the selected frames into a clip,
- align each selected frame with the corresponding CSV row.

This is different from the existing landing-regression dataset and should be implemented as a new dataset class, rather than trying to overload the current `TrajectoryDataset`.

---

## 5. Input Format

### 5.1 Default Input Mode

Default input should be grayscale.

Reasoning:

- the existing dataset is effectively grayscale already,
- the current benchmarks also use grayscale,
- color is not an important signal in the current task.

### 5.2 Encoder Compatibility

To reduce initial code churn, v1 can still feed grayscale images into a 3-channel encoder by repeating the grayscale channel three times.

Recommended implementation priority:

1. v1: grayscale replicated to 3 channels
2. later cleanup: make encoder `in_channels` configurable and support `1` directly

---

## 6. Model Adaptation Plan

### 6.1 Components To Keep

Keep the following design from the reference project:

- multi-scale encoder
- multi-scale hidden state
- `ConvGRU` observation update
- autonomous future rollout
- coarsest-state probes

### 6.2 Components To Change

Change the following:

1. input channel handling
2. dataset interface
3. probe head output dimensions
4. optional frame-reconstruction branch

### 6.3 Probe Head Shapes

Default:

- `pos_head: Linear(s3, 3)`
- `vel_head: Linear(s3, 3)`

Planar ablation:

- `pos_head: Linear(s3, 2)`
- `vel_head: Linear(s3, 2)`

### 6.4 Frame Reconstruction As A Configurable Option

Future-frame reconstruction should be configurable rather than always on.

Required options:

- `enable_frame_recon: bool`
- `future_recon_len: int`
- `lambda_frame: float`

Behavior:

- if `enable_frame_recon = false`, skip decoder loss entirely
- if `enable_frame_recon = true`, reconstruct only the first `future_recon_len` rollout frames
- require `future_recon_len <= future_len`

Recommended default for v1:

- `enable_frame_recon = true`
- `future_recon_len = 4`
- `lambda_frame = 1.0`
- use a narrow first-pass model width:
  - `c1 = c2 = c3 = 8`
  - `s1 = s2 = s3 = 8`
  - `decoder_proj_ch = 8`
  - `decoder_hidden_ch = 8`

Important ablation:

- same setting but `enable_frame_recon = false`

This ablation is useful because it directly tests whether image reconstruction actually helps the temporal state, or whether the probes alone are enough.

---

## 7. Supervision Targets

### 7.1 Default Supervision

At every selected step in the clip, supervise:

- `pos_seq = (ball_px, ball_py, ball_pz)`
- `vel_seq = (ball_vx, ball_vy, ball_vz)`

These targets are available directly from `frames.csv`.

### 7.2 Loss Terms

Recommended v1 loss:

- `loss_frame`: future-frame reconstruction loss
- `loss_pos`: all-step position regression loss
- `loss_vel`: all-step velocity regression loss

Total:

`L = lambda_frame * L_frame + lambda_p * L_pos + lambda_v * L_vel`

Recommended default weights:

- `lambda_frame = 1.0`
- `lambda_p = 1.0`
- `lambda_v = 0.2`

### 7.3 Monotone Loss

Do not include the reference project's monotone loss in v1.

Reasoning:

- our trajectories are not one-dimensional monotonic motion,
- 360-degree sampling invalidates that prior,
- adding it now would inject a likely-wrong inductive bias.

---

## 8. Training Setup

### 8.1 Default Optimization

Recommended v1 training setup:

- optimizer: `AdamW`
- learning rate: `3e-4`
- batch size: `8`
- epochs: `30`
- seed set: `42, 43, 44`
- checkpoint rule: best validation total loss

### 8.2 Validation Outputs

Track at least:

- total validation loss
- frame reconstruction loss
- position loss
- velocity loss
- per-dimension MAE for position and velocity

If possible, also track:

- history-step probe error
- rollout-step probe error

The split between history and rollout is important because the value of this model is not just state encoding, but stable future rollout.

---

## 9. Evaluation Plan

### 9.1 Primary Questions

The first transfer experiment should answer:

1. can the model predict physically meaningful state sequences from image history?
2. does future-frame reconstruction improve probe quality?
3. is 3D probe supervision better than planar-only supervision?

### 9.2 Recommended First Matrix

Run these in order:

#### Exp A: Default 3D + reconstruction

- probe dims: 3D
- frame reconstruction: on
- `future_recon_len = 4`

#### Exp B: 3D without reconstruction

- probe dims: 3D
- frame reconstruction: off

#### Exp C: Planar-only with reconstruction

- probe dims: 2D `(x,z)` and `(vx,vz)`
- frame reconstruction: on

If compute budget is limited, run only Exp A first.

### 9.3 Baseline For Comparison

The most relevant current comparison target is the formal GRU baseline on:

`20260316_220723`

This comparison will not be one-to-one at the task-output level, but it will tell us whether the richer temporal state is worth carrying forward.

---

## 10. Artifacts To Save

Each run should save:

- config
- `best.pt`
- `last.pt`
- training history CSV
- loss curves
- if reconstruction is enabled:
  - a diagnostic image panel showing
    - observed history frames
    - true future frames
    - predicted future frames
- a probe diagnostic figure showing
  - true vs predicted position curves
  - true vs predicted velocity curves

Recommended per-run summary fields:

- best epoch
- val total loss
- val position MAE
- val velocity MAE
- rollout-only position MAE
- rollout-only velocity MAE

---

## 11. Engineering Plan

### 11.1 New Files Expected

Recommended additions:

- `temperal/src/datasets/temporal_clip_dataset.py`
- `temperal/src/models/multiscale_future_dynamics.py`
- `temperal/scripts/train_multiscale_future.py`
- `temperal/scripts/evaluate_multiscale_future.py`

### 11.2 Reuse

Can reuse:

- split files
- existing seed utilities
- existing output organization pattern
- existing formal-results reporting style

### 11.3 Keep Separate From Current Landing Baseline

Do not overload the current regression code path at first.

Reasoning:

- the new task is clip-based, not whole-trajectory-to-terminal-target,
- the output structure is different,
- separate code makes debugging and interpretation easier.

---

## 12. Default v1 Decision

Unless changed later, the first implementation should use:

- dataset: `20260316_220723`
- clip mode: contiguous clip
- `seq_len = 12`
- `history_len = 8`
- `future_len = 4`
- `frame_stride = 2`
- grayscale input replicated to 3 channels
- `c1 = c2 = c3 = 8`
- `s1 = s2 = s3 = 8`
- `decoder_proj_ch = 8`
- `decoder_hidden_ch = 8`
- 3D position probe `(px, py, pz)`
- 3D velocity probe `(vx, vy, vz)`
- future-frame reconstruction enabled
- `future_recon_len = 4`
- loss weights: `frame=1.0, pos=1.0, vel=0.2`
- seeds: `42, 43, 44`

---

## 13. What This Experiment Is Not Yet

This v1 experiment is not yet:

- direct landing-point regression
- direct action regression
- RL transfer
- a final claim about downstream sample efficiency

It is a controlled temporal visual modeling experiment designed to determine whether the multi-scale rollout state is worth carrying into later downstream tasks.

---

## 14. Immediate Next Step

Implement Exp A first:

- 3D probes
- frame reconstruction on
- `seq_len = 12, history = 8, future = 4`
- single seed smoke run first

If that works and diagnostics look sane, then continue to:

- remove reconstruction for ablation,
- compare 3D vs planar probes,
- connect the learned state to downstream landing prediction.
