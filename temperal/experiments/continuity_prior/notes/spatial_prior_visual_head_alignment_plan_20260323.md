# Spatial Prior Visual Head Alignment Plan

Date: `2026-03-23`

## 1. Purpose

This note is for the actual `continuity_prior` goal in `temperal/`:

- first train a visual head with the earlier Part I spatial continuity prior
- then connect that visual head to the Part II temporal module
- finally test whether `spatial prior + temporal module` is better than the current temporal-only setup

This document is intentionally a design note only.
It does **not** start implementation.

## 2. What This Experiment Really Means

There are two different `continuity` stories in the repo:

### Part I: spatial continuity prior

Location:

- [vis_backbone/README.md](/d:/projects/rl-ball-fetch/vis_backbone/README.md)
- [vis_backbone/v0.0.1/README.md](/d:/projects/rl-ball-fetch/vis_backbone/v0.0.1/README.md)

Core idea:

- learn a visual representation with spatial structure
- use feature warp consistency, warp-decode consistency, and reconstruction
- output a spatial feature map rather than a single global vector

### Part II: temporal module in `temperal/`

Location:

- [multiscale_future_dynamics.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_future_dynamics.py)
- [multiscale_transfer_regressor.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_transfer_regressor.py)

Core idea:

- use a multi-scale CNN encoder
- feed multi-scale features into ConvGRU-style temporal dynamics
- solve multi-frame downstream prediction

Therefore, the real `continuity_prior` experiment is **not**:

- temporal continuity inside `temperal/` only

It is:

- reuse the Part I spatial-prior visual backbone
- adapt it to the Part II temporal interface
- evaluate the combined system

## 3. Current Mismatch

The key mismatch is interface shape, not research intent.

### Part I `v0.0.1`

Current encoder output:

- one feature map only
- shape roughly `(C, 8, 8)`

Reference:

- [model.py](/d:/projects/rl-ball-fetch/vis_backbone/v0.0.1/model.py)

### Part II `temperal`

Current temporal encoder output:

- `f1`: `(c1, 32, 32)`
- `f2`: `(c2, 16, 16)`
- `f3`: `(c3, 8, 8)`

Reference:

- [multiscale_future_dynamics.py](/d:/projects/rl-ball-fetch/temperal/src/models/multiscale_future_dynamics.py)

The temporal module is written around this three-scale contract.

So the real question is:

- how do we keep the Part I spatial prior,
- while producing a visual interface the temporal module can consume cleanly?

## 4. Constraint We Should Respect

At this stage, the safest constraint is:

- keep work scoped under `temperal/experiments/continuity_prior`
- avoid editing the existing Part I code directly unless later approved
- avoid large changes to the current temporal stack before we know the interface plan is right

This suggests the first target should be:

- a new training design under `temperal/` that borrows the Part I loss idea
- but trains a temporal-compatible visual encoder

## 5. Candidate Alignment Strategies

There are three realistic strategies.

### Option A. Train the current `MultiScaleEncoder` with spatial-prior losses

Idea:

- keep the current `temperal` encoder structure
- pretrain it using Part I continuity losses
- use `f3` for warp/transport losses
- optionally add reconstruction from `f3` or fused pyramid features

Pros:

- easiest checkpoint compatibility
- no channel mismatch at load time
- minimal downstream integration risk

Cons:

- this is only a migration of the Part I idea
- it is not literally the same visual head as `v0.0.1`
- if channels stay tiny (`c1=c2=c3=8`), visual capacity may be too weak

Assessment:

- good as a minimal engineering bridge
- not ideal if the real goal is to preserve the stronger Part I visual prior as faithfully as possible

### Option B. Build a new spatial-prior multi-scale visual head

Idea:

- design a visual encoder that outputs `f1, f2, f3`
- train it with spatial-prior losses adapted from Part I
- make `f3` the main continuity feature space
- let `f1/f2` act as supporting features for the temporal module later

Pros:

- preserves the Part I research story better
- produces native `temperal`-compatible outputs
- avoids brittle post-hoc shape hacks

Cons:

- requires a new visual-head training implementation
- slightly more design work up front

Assessment:

- this is the cleanest long-term route
- this is the current recommended direction

### Option C. Keep the exact Part I head and add an adapter into the temporal module

Idea:

- train or reuse the original single-scale Part I encoder
- add an adapter that converts its single `(C, 8, 8)` feature map into the temporal module inputs

Possible adapter forms:

- derive only `f3` and redesign temporal module to work from one scale
- upsample and project to synthesize `f2` and `f1`
- add a new pre-encoder stem before dynamics

Pros:

- closest to the original Part I encoder

Cons:

- the interface is unnatural
- synthetic `f1/f2` may be semantically weak
- likely creates more confusion than value in ablations

Assessment:

- not recommended as the first route

## 6. Recommended Direction

The recommended plan is **Option B**:

- create a new spatial-prior multi-scale visual head inside `temperal`
- make its outputs natively match `f1, f2, f3`
- keep the Part I training logic, especially on the coarsest continuity feature space

This gives us:

1. the right research story
2. the right tensor interface
3. the least awkward downstream loading path

## 7. Recommended Visual Head Design

The new visual head should be thought of as:

- "Part I objective"
- with a "Part II-compatible encoder interface"

### Encoder contract

Recommended output:

- `f1`: high-resolution feature map for temporal shallow scale
- `f2`: mid-resolution feature map
- `f3`: coarse feature map and main continuity feature space

The actual architecture can stay simple at first:

- stride-2 stem -> `32x32`
- second downsample -> `16x16`
- third downsample -> `8x8`
- lightweight residual blocks at each scale

### Continuity supervision target

Recommended first choice:

- apply continuity losses primarily on `f3`

Reason:

- this stays closest to Part I
- keeps the warp space simple
- avoids turning the first implementation into a large multi-loss system

### Decoder contract

Recommended first choice:

- reconstruct from `f3` only, or from a small `f3 -> image` decoder

Reason:

- reconstruction is mainly anti-collapse supervision
- we do not yet need a large high-fidelity image decoder

Possible later upgrade:

- fuse `f1/f2/f3` in the decoder if `f3`-only reconstruction is too weak

## 8. Channel Alignment Recommendation

This is the main issue the current note should settle before implementation.

The safest recommendation is:

- do **not** force the new spatial-prior visual head to use the current tiny default widths immediately if that weakens the visual model too much
- do **not** keep the exact old Part I `C=64 only` interface either

Instead, separate two concepts:

1. visual pretraining width
2. temporal loading width

### Recommended first practical scheme

Train the spatial-prior visual head with a modest but expressive pyramid, for example:

- `c1 = 16`
- `c2 = 32`
- `c3 = 64`

Then define one of two loading paths later:

#### Path 1. Direct-width temporal model

- widen the temporal model to the same widths

Pros:

- cleanest semantics
- no projection loss

Cons:

- requires editing the temporal model outside this experiment folder

#### Path 2. Learned projection adapter

- keep the visual head widths above
- add `1x1` projections to map them into a smaller temporal width if needed

Pros:

- preserves stronger visual representation
- gives a controlled compatibility layer

Cons:

- adds one more component to ablate

### What to avoid first

Avoid this first:

- training the spatial prior directly at `8/8/8` just because the current temporal defaults happen to be small

Reason:

- that answers the engineering mismatch quickly
- but may throw away exactly the visual capacity that made Part I useful

## 9. Training Plan for the Visual Head

Before any temporal integration, the visual head should have its own self-contained training and check phase.

### Stage VP-1. Smoke training

Goal:

- verify data loading
- verify warp path
- verify checkpoint save/load

Expected outputs:

- `best.pt`
- `last.pt`
- `history.csv`
- reconstruction visualization

### Stage VP-2. Formal visual pretraining

Goal:

- train the spatial-prior visual head on the existing capture dataset

Core losses:

- `L_rec`
- `L_trans`
- `L_wd`
- optional `L_nb`

### Stage VP-3. Visual-head-only evaluation

Before mixing in time, do a lightweight visual check.

Recommended checks:

- reconstruction grid
- feature transport sanity checks
- single-frame probe or small downstream probe if needed

This stage matters because otherwise a later temporal failure is ambiguous:

- did the visual prior fail?
- or did the temporal integration fail?

## 10. Integration Plan After Visual Pretraining

Only after the visual head itself looks reasonable should we integrate it into the temporal stack.

Recommended sequence:

1. load pretrained visual encoder into a temporal-compatible model
2. freeze visual encoder first
3. train temporal dynamics + downstream head
4. then run a finetune version with encoder unfrozen

This gives a clean comparison:

- scratch
- frozen spatial-prior visual head + temporal module
- finetuned spatial-prior visual head + temporal module
- future-dynamics pretraining baseline

## 11. Suggested First Experimental Question

The first combined question should be narrow:

- does a spatial-prior visual head improve the current `observe-then-act` multi-frame downstream task?

The first matrix should stay small:

- one downstream protocol
- one or two data fractions
- a small number of seeds

This is enough to tell whether the combination is promising before broader sweeps.

## 12. Recommended Deliverables Before Coding

Before implementation starts, we should make these explicit:

1. final encoder interface
   - exact `f1/f2/f3` shapes
2. visual pretraining width choice
   - whether to use direct-width or projected-width loading
3. checkpoint contract
   - what keys must be saved for later loading
4. first evaluation matrix
   - which downstream benchmark is the first target

## 13. Current Recommendation

The most defensible next implementation target is:

- build a new multi-scale visual head in `temperal`
- train it with Part I spatial continuity losses
- keep `f3` as the main continuity space
- defer temporal integration until the visual head is independently checked

In short:

- first make the spatial prior strong and interface-correct
- then plug it into the temporal module
- do not start from post-hoc channel hacks

## 14. Loss Placement Decision (Updated)

The agreed first-pass loss placement is now:

- `f3` is the main continuity space
- `f2` gets a small auxiliary transport loss
- `f1` does not get a strong continuity loss in the first pass

Rationale:

- `f3` is closest to the original Part I `8x8xC` feature-map setting
- `f2` may still benefit from a weak continuity bias
- `f1` is most sensitive to zero-flow mismatch and coarse warp error, so it should not be heavily constrained yet

Recommended initial weighting style:

- `lambda_trans_f3 = 1.0`
- `lambda_trans_f2 = 0.1 ~ 0.2`

All other continuity-heavy terms should stay centered on `f3` first.

## 15. Immediate Next Discussion Items

Before coding, the next discussion should settle just two decisions:

1. Should the new visual head be trained at a richer width such as `16/32/64`, or should it match the current temporal widths directly?
2. When integrating into the temporal model later, do we want:
   - direct width matching, or
   - a learned projection adapter?

Those two decisions determine almost all of the implementation shape.

## 16. Current Training Artifact

The first trained experiment-local multi-scale spatial-prior visual backbone currently lives at:

- [best.pt](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/best.pt)

Training output directory:

- [multiscale_spatial_prior_train_20260323](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323)

Saved training-process data:

- [history.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/history.csv)
- [batch_log.csv](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/batch_log.csv)
- [loss_curve.svg](/d:/projects/rl-ball-fetch/temperal/outputs/multiscale_spatial_prior_train_20260323/loss_curve.svg)

This run used:

- multi-scale visual widths `16/32/64`
- `f3` as the main continuity space
- `f2` as a small auxiliary transport term
