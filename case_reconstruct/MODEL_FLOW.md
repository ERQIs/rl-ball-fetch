# Model Flow Reconstruction

This note reconstructs how one training step runs in this project.

## 1) Unity side (environment step)

1. Agent receives action `[forward, lateral]`.
2. `CarCatcherAgent.OnActionReceived()` applies target planar velocity.
3. Reward shaping is added every decision step.
4. Episode can end by:
   - caught ball (`OnBallCaught`)
   - missed ball (`OnBallMissed`)
   - out-of-arena
   - max-step timeout

## 2) Observation pipeline

1. Camera sensor `CarCam` produces grayscale stacked frames (64x64, stacks=6).
2. If `useVectorObs` is enabled, `CollectObservations()` appends vector features.
3. Behavior Parameters determine observation spec sent to Python trainer.

## 3) Python trainer (ML-Agents PPO)

1. Observation encoder receives visual (and optional vector) observations.
2. Encoded features are concatenated in `ObservationEncoder` and sent to policy/value heads.
3. PPO computes:
   - policy loss
   - value loss
   - entropy regularization
4. Gradients update actor/critic network.

## 4) Custom experiment hooks used here

- `exp_vb_0001`: replace visual encoder with pretrained Stage-A backbone wrapper.
- `exp_vb_0002_blind`: replace visual encoder with zero-output blind encoder.

Both hooks are runtime patches in experiment launchers, not global package edits.

## 5) Why reward can disagree with success rate

Success rate only tracks catch/miss events. Cumulative reward includes all step rewards and penalties.
If many episodes terminate by timeout/out-of-arena or dense penalties dominate, success can improve while
reward stays flat or negative.
