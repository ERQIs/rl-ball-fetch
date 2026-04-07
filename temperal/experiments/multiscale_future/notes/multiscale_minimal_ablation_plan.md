# Multiscale Minimal Ablation Plan

This note is intended to answer one focused question:

- is the current `multiscale` model design worth keeping as a research object?

The goal is not to exhaustively ablate every component.
The goal is to run the smallest set of experiments that can support or reject the strongest useful claim.

## 1. What We Can Already Say From Existing Results

Based on the current `observe-then-act` experiments:

1. `multiscale scratch` is not a stronger supervised predictor than the existing `cnn_gru` baseline.
2. `multiscale finetune` is consistently better than `multiscale scratch`.
3. `multiscale frozen` is consistently weak.
4. In the matched `obs50` protocol, `multiscale finetune` shows data-efficiency gains over `scratch`.
5. In the `full + obs67` protocol, `multiscale finetune` becomes competitive with the formal `cnn_gru` baseline.

Therefore, the most defensible current claim is:

- the `multiscale` model is valuable as a carrier for pretraining and structured priors
- it is not currently justified as a stronger scratch-only downstream architecture

## 2. What Is Still Missing

Current evidence still does not tell us:

1. whether the gain comes specifically from the `multiscale future-dynamics` pretraining design
2. whether a simpler continuity-style encoder pretraining would be enough
3. whether the future-frame reconstruction branch is actually necessary

These are the three most important missing causal questions.

## 3. Minimal Questions To Answer

The minimal ablation package should answer:

### Q1. Is the temporal rollout pretraining better than a simpler spatial continuity prior?

If yes:

- the `multiscale future-dynamics` idea is not just “any pretraining works”

If no:

- the simpler continuity prior may already explain most of the benefit

### Q2. Inside the future-dynamics framework, is future-frame reconstruction necessary?

If yes:

- the decoder branch is carrying useful supervision

If no:

- the main value may come from state probes and recurrent rollout alone

### Q3. Is the model’s value really “better as a pretrained carrier” rather than “better as a scratch model”?

This is mostly already answered, so we should avoid rerunning large redundant experiments unless needed for matching conditions.

## 4. Smallest Useful Ablation Package

## Experiment A: Prior Type Comparison

Purpose:

- compare two different kinds of pretraining under the same downstream evaluation

Models:

- `scratch`
- `continuity-encoder finetune`
- `future-dynamics finetune`

Fixed downstream protocol:

- use the same `obs50` data-efficiency protocol already established
- dataset: `20260316_220723`
- fractions: `20%` and `40%`
- seeds: `42, 43, 44`

Why only `20%` and `40%`:

- `20%` is where data-efficiency should be most visible
- `40%` tells us whether the ranking is stable
- `100%` is less informative for this particular question and can be skipped initially

Interpretation:

- if `future-dynamics finetune > continuity-encoder finetune > scratch`, then the temporal pretraining mechanism has added value beyond a spatial continuity prior
- if `continuity-encoder finetune ≈ future-dynamics finetune`, then the simpler prior may already explain most of the gain

## Experiment B: Reconstruction Ablation Within Future-Dynamics

Purpose:

- test whether the future-frame reconstruction branch is necessary

Models:

- `future-dynamics pretrain + finetune` with reconstruction `on`
- `future-dynamics pretrain + finetune` with reconstruction `off`

Fixed downstream protocol:

- use `obs50`
- fraction: `20%`
- seeds: `42, 43, 44`

Why only `20%`:

- if reconstruction helps, the gain should be easiest to observe in the lower-data regime
- this keeps the experiment small

Interpretation:

- if reconstruction `on > off`, then image prediction is a real contributor
- if reconstruction `on ≈ off`, then the decoder branch may be unnecessary complexity

## Experiment C: No New Run Required, Just Explicitly Cite Existing Evidence

Purpose:

- establish the role of the architecture itself without new compute

Use existing results:

- `multiscale scratch` vs `cnn_gru baseline`
- `multiscale finetune` vs `multiscale scratch`

Interpretation:

- this already shows that the architecture is not justified by scratch performance alone
- its value must be argued through representation learning and finetuning benefit

## 5. Recommended Execution Order

### Step 1

Run Experiment A first.

Reason:

- this is the most important question
- it tests whether the more complicated temporal pretraining is actually better than the earlier continuity prior

### Step 2

Run Experiment B second.

Reason:

- only worth doing if the future-dynamics line itself still looks promising after Experiment A

### Step 3

Do not rerun large scratch-vs-GRU sweeps unless a reviewer later asks for a stricter matched comparison.

Reason:

- current evidence is already sufficient to reject the “scratch architecture is simply better” interpretation

## 6. Exact Minimal Matrix

This is the minimal matrix I recommend.

| Group | Fraction | Seeds | Methods | Why |
| --- | --- | --- | --- | --- |
| A | `20%` | `42,43,44` | `scratch`, `continuity finetune`, `future-dynamics finetune` | strongest low-data comparison |
| A | `40%` | `42,43,44` | `scratch`, `continuity finetune`, `future-dynamics finetune` | check stability of ranking |
| B | `20%` | `42,43,44` | `future-dynamics finetune (recon on/off)` | test necessity of decoder loss |

Total new downstream runs:

- Group A: `2 fractions x 3 methods x 3 seeds = 18 runs`
- Group B: `1 fraction x 2 methods x 3 seeds = 6 runs`
- total = `24` downstream runs

This is small enough to be realistic, and large enough to answer the key questions.

## 7. What We Do Not Need Yet

To keep the ablation package minimal, I do not recommend doing these yet:

- `100%` continuity-vs-future comparison
- planar-probe vs 3D-probe comparison
- many different `lambda_nb` sweeps
- many different hidden widths
- frozen continuity transfer as a main line

These can all wait unless the first minimal package gives ambiguous results.

## 8. Decision Rules

### Keep the future-dynamics model as the main story if:

- `future-dynamics finetune` is consistently better than `continuity finetune`
- or reconstruction `on` is consistently better than reconstruction `off`

### Downgrade the future-dynamics model to a secondary story if:

- `future-dynamics finetune` and `continuity finetune` are basically tied
- and reconstruction `on/off` makes little difference

In that case, the stronger story may become:

- “a simpler spatial continuity prior is already enough”

## 9. Best Current Framing Before New Ablations

Right now, before running the new ablations, the safest framing is:

- the `multiscale` model is worth keeping because it is an effective pretraining carrier
- but we have not yet proven that its specific future-dynamics design is necessary

That missing proof is exactly what Experiments A and B are meant to provide.
