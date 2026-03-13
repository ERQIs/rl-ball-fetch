# Stage-A Backbone v0.0.1

Minimal implementation based on `vis_backbone/README.md`:

- Input: `I_t`, `I_t1` shape `(1,64,64)` by default (or `(3,64,64)`), `flow_t` shape `(2,64,64)`
- Encoder output: `F_t`, `F_t1` shape `(C,8,8)`
- Decoder output: `I_hat_t`, `I_hat_t1_from_warp`
- Warp: feature-space warp using `grid_sample`
- Loss:
  - `L_rec` reconstruction
  - `L_trans` transport consistency
  - `L_wd` warp-decode consistency
  - optional `L_nb` neighbor smoothness

## Files

- `config.py`: default hyper-parameters
- `model.py`: encoder/decoder + stage forward
- `warp.py`: flow-to-feature warp
- `losses.py`: Stage-A losses
- `data.py`: trajectory frame-pair dataset + zero-flow provider
- `train_step.py`: one-step optimization
- `train.py`: minimal train entry
- `visualize_recon.py`: reconstruction comparison grid export
- `tests/test_warp.py`: warp unit tests

## Quick Start

Use existing env under `rl_be/.venv`:

```powershell
cd vis_backbone\v0.0.1
..\..\rl_be\.venv\Scripts\python.exe -m unittest tests\test_warp.py
..\..\rl_be\.venv\Scripts\python.exe train.py --dry-run --steps 20
```

Train on local capture data (currently uses zero-flow baseline):

```powershell
..\..\rl_be\.venv\Scripts\python.exe train.py `
  --dataset-root ..\datasets\manual_capture\use_this_directory_plz `
  --steps 200 --batch-size 32 --in-channels 1 `
  --fg-weight 6.0 --fg-thresh 0.55 --fg-kernel 5
```

Visualize reconstruction samples from a trained checkpoint:

```powershell
..\..\rl_be\.venv\Scripts\python.exe visualize_recon.py `
  --checkpoint artifacts\stagea_last.pt `
  --dataset-root ..\datasets\manual_capture\use_this_directory_plz `
  --in-channels 1 --num-samples 8 `
  --output artifacts\recon_grid.png
```

## Notes

- `warp.py` assumes `flow_t` is forward flow from `I_t -> I_t1`; sampling uses backward grid (`base - flow`).
- If you have a real dense flow estimator, replace `ZeroFlowProvider` in `data.py`.
- `--in-channels` supports `1` (grayscale) and `3` (RGB), default is `1`.
- `train.py` saves checkpoints under `--save-dir` (default `artifacts`), including `stagea_last.pt`.
- Foreground-weighted reconstruction is enabled by default (`--fg-weight`, `--fg-thresh`, `--fg-kernel`) to reduce small-object collapse.
