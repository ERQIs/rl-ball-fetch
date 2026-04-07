# Embodied Policy V1: Stride4 Window8

## Core Decision

The first embodied-control version will use:

- local visual window length: `8` frames
- local visual window stride / macro-step: `4` frames
- action update period: every `4` frames

This means each local visual token spans two consecutive control intervals while the policy itself acts once per interval.

## Observation Policy

Deployment-time inputs should stay non-privileged:

- image window
- self planar velocity only

Explicitly excluded from policy input:

- ball privileged vector
- car world position
- separate previous action

Rationale:

- in the current Unity environment, action is applied directly as target planar velocity
- so `prev_action` and actual self velocity are nearly the same signal
- actual self velocity is the more grounded embodied variable

## Self-State Timing

For each `8`-frame visual window, use one self-state sample:

- take the self planar velocity at the window end

Because action updates every `4` frames and remains constant inside that interval, the window-end velocity is effectively equivalent to the window's piecewise-constant control state in v1.

## Architecture

Per macro-step `t`:

1. local visual window -> `z_short_t`
2. self velocity -> `z_self_t`
3. memory input fuse(`[z_short_t, z_self_t]`) -> `z_mem_t`
4. recurrent update with `z_mem_t` -> `h_long_t`
5. action head reads `[z_short_t, h_long_t, z_self_t]`
6. value head reads `[z_short_t, h_long_t, z_self_t]`

Design intent:

- `z_short_t`: newest short-horizon visual evidence
- `h_long_t`: long-running trajectory memory
- `z_self_t`: current embodied motion state

## Parameter Reuse Plan

Warm-start from the hierarchical landing branch where possible:

- reuse `encoder`
- reuse `dynamics`
- reuse `token_proj`
- reuse `long_gru` if the fused memory token is projected back to the original token dimension

Do not reuse:

- landing head

## Training Policy

Initial policy for optimization:

- freeze local visual encoder at the start
- allow local dynamics to unfreeze early
- train new self-state encoder, memory fuse, recurrent controller, policy head, and value head from the beginning

## Offline Warmup Policy

Offline warmup may use privileged labels only as targets, never as policy inputs.

Candidate warmup targets:

- landing `xz`
- time-to-land
- future motion summary

The policy input remains:

- image window
- self planar velocity
