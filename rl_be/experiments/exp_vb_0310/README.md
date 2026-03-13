# exp_vb_0310

Prompt 0310 experiment suite for backbone ablations:

- `E1`: Global-Scratch-E2E
- `E2`: Spatial-Scratch-E2E
- `E3`: Spatial-Pretrain-Frozen
- `E4`: Spatial-Pretrain-Finetune

## Backbone checkpoint facts

From `vis_backbone\v0.0.1\artifacts\stagea_last.pt`:

- input: grayscale `1 x 64 x 64`
- encoder output feature map: `8 x 8 x 8`
- encoder params: `18,240`
- decoder params: `39,153`
- total pretrained model params: `57,393`

## Suite layout

- launcher: `train_variants.py`
- configs: `config\E1_*.yaml` ... `config\E4_*.yaml`
- scripts:
  - `scripts\run_editor.ps1`
  - `scripts\run_build.ps1`
- outputs:
  - `rl_be\results\exp_vb_0310\E1\...`
  - `rl_be\results\exp_vb_0310\E2\...`
  - `rl_be\results\exp_vb_0310\E3\...`
  - `rl_be\results\exp_vb_0310\E4\...`

No global edits to `site-packages`.

## Run in Unity Editor

Open Unity and press Play, then:

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\experiments\exp_vb_0310\scripts\run_editor.ps1 -Experiment E1
.\experiments\exp_vb_0310\scripts\run_editor.ps1 -Experiment E2
.\experiments\exp_vb_0310\scripts\run_editor.ps1 -Experiment E3 -BackboneCkpt d:\projects\rl-ball-fetch\vis_backbone\v0.0.1\artifacts\stagea_last.pt
.\experiments\exp_vb_0310\scripts\run_editor.ps1 -Experiment E4 -BackboneCkpt d:\projects\rl-ball-fetch\vis_backbone\v0.0.1\artifacts\stagea_last.pt
```

## Run with Unity Build

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\experiments\exp_vb_0310\scripts\run_build.ps1 -Experiment E1 -EnvPath D:\path\to\BallFetch.exe
.\experiments\exp_vb_0310\scripts\run_build.ps1 -Experiment E2 -EnvPath D:\path\to\BallFetch.exe
.\experiments\exp_vb_0310\scripts\run_build.ps1 -Experiment E3 -EnvPath D:\path\to\BallFetch.exe -BackboneCkpt d:\projects\rl-ball-fetch\vis_backbone\v0.0.1\artifacts\stagea_last.pt
.\experiments\exp_vb_0310\scripts\run_build.ps1 -Experiment E4 -EnvPath D:\path\to\BallFetch.exe -BackboneCkpt d:\projects\rl-ball-fetch\vis_backbone\v0.0.1\artifacts\stagea_last.pt
```

## Notes

- All four configs use the same PPO hyperparameters by default.
- `E3` freezes the pretrained spatial backbone.
- `E4` loads the same pretrained backbone and allows end-to-end finetuning.
- `E2` uses the same spatial encoder architecture as the pretrained backbone, but random init.
- `E1` uses a global-pooled visual encoder as the global-backbone baseline.
