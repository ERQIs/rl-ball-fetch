# Embodied Control Branch Scope

## Motivation

`landing_world_model` already shows that the temporal backbone carries useful motion information.
But that line is still centered on supervised landing prediction, not closed-loop control.

This branch is for the more embodied direction:

- keep a longer-running internal state
- fuse visual evidence with car self-motion state
- update memory online for step-by-step control
- support step-by-step action generation for `rl_be`

## Intended Difference From Existing Branches

- `observe_then_act`: delayed action after observation stage
- `landing_world_model`: observation-to-landing supervision
- `embodied_control`: online recurrent control with memory

## Near-Term Questions

- where to inject self state: local encoder input, token fusion, or policy head only
- whether to use only self velocity in v1 or add richer proprioception later
- whether to warm-start from `landing_world_model` local temporal backbone
- how to bridge the temporal latent state into ML-Agents / `rl_be`

## First Practical Milestone

Build a minimal recurrent policy backbone that consumes:

- image
- self planar velocity
- optional privileged supervision during offline warmup only as training targets

and exposes a compact hidden state that can be rolled online.
