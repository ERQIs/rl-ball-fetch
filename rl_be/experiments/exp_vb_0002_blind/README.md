# exp_vb_0002_blind: No-Image Baseline

This experiment creates a strict "agent cannot use image information" baseline.

Method:

- Replace ML-Agents `vis_encode_type: simple` encoder with `BlindVisualEncoder`.
- `BlindVisualEncoder` ignores visual tensors and always outputs zeros.
- By default this experiment also zeros vector observations (`BLIND_ZERO_VECTOR=1`), creating a strict no-perception baseline.

This is isolated to this experiment only; no global package edits.

## Output path

- `rl_be/results/exp_vb_0002_blind/<run-id>/`

## Run in Unity Editor

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\experiments\exp_vb_0002_blind\scripts\run_editor.ps1 `
  -RunId exp_vb_0002_blind_editor_seed1
```

## Run with Unity Build

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\experiments\exp_vb_0002_blind\scripts\run_build.ps1 `
  -EnvPath D:\path\to\YourUnityBuild.exe `
  -RunId exp_vb_0002_blind_build_seed1 `
  -NumEnvs 1
```

Optional:

- `-Resume`: resume training
- `-NoGraphics`: pass `--no-graphics` to ML-Agents
- `-KeepVectorObs`: only blind visual input, keep vector observations

## Interpretation

- If success rate stays close to random with this baseline, your visual information is likely useful.
- If success remains high, policy may rely heavily on non-visual channels (e.g., vector obs/reward shaping leakage).
