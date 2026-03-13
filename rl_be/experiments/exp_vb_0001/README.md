# exp_vb_0001: PPO + Pretrained Visual Backbone

This experiment runs Unity ML-Agents PPO while replacing the default `simple` visual encoder with a pretrained Stage-A encoder checkpoint.

## Isolation

- Code: `rl_be/experiments/exp_vb_0001/`
- Config: `config/ppo_vision_backbone.yaml`
- Launcher: `train_with_backbone.py` (runtime monkey patch only)
- Outputs: `rl_be/results/exp_vb_0001/<run-id>/`

No modifications are made to `rl_be/.venv/site-packages`.

## Required input

- `VBB_CKPT` checkpoint from `vis_backbone/v0.0.1/train.py` (for example `stagea_last.pt`)

## Run in Unity Editor

1. Open Unity project `ball_fetch`, scene `CatchCarScene`, click `Play`.
2. Run:

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\experiments\exp_vb_0001\scripts\run_editor.ps1 `
  -BackboneCkpt d:\projects\rl-ball-fetch\vis_backbone\v0.0.1\artifacts_fg_long\stagea_last.pt `
  -RunId exp_vb_0001_editor_seed1
```

## Run with Unity Build

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\experiments\exp_vb_0001\scripts\run_build.ps1 `
  -BackboneCkpt d:\projects\rl-ball-fetch\vis_backbone\v0.0.1\artifacts_fg_long\stagea_last.pt `
  -EnvPath D:\path\to\YourUnityBuild.exe `
  -RunId exp_vb_0001_build_seed1 `
  -NumEnvs 1 `
  -NoGraphics
```

## Options

- `-Resume`: resume same run-id
- `-UnfreezeBackbone`: allow RL to update backbone weights (default is frozen backbone)
- `-NoGraphics`: pass `--no-graphics` to ML-Agents (headless build run)
- `-Inference`: run inference-only mode (no training updates), useful for visual debugging
- `-TimeScale <value>`: override Unity time scale from CLI (e.g. `1`, `5`, `20`)

Example (visible inference debug in build):

```powershell
.\experiments\exp_vb_0001\scripts\run_build.ps1 `
  -BackboneCkpt d:\projects\rl-ball-fetch\vis_backbone\v0.0.1\artifacts_fg_long\stagea_last.pt `
  -EnvPath D:\path\to\YourUnityBuild.exe `
  -RunId exp_vb_0001_build_seed1 `
  -Resume -Inference -NumEnvs 1 -TimeScale 3
```

## Notes

- The patch maps `vis_encode_type: simple` to the pretrained encoder wrapper.
- Wrapper handles arbitrary visual input channel count using a learnable `1x1` adapter to the grayscale-trained backbone.
