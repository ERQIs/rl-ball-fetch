## anouncement

the reason why this project exist is that codex didn't get what I wanted it to do....

# case_reconstruct

Tools to reconstruct and diagnose RL training behavior when performance plateaus.

## Files

- `reconstruct_case.py`: parse TensorBoard events and generate a diagnosis report.
- `MODEL_FLOW.md`: end-to-end training flow reconstruction.

## Quick usage

```powershell
cd d:\projects\rl-ball-fetch
.\rl_be\.venv\Scripts\python.exe .\case_reconstruct\reconstruct_case.py `
  --run-dir .\rl_be\results\exp_vb_0001\exp_vb_0003_build_seed1 `
  --scene .\ball_fetch\Assets\CatchCarScene_basket_0309.unity
```

Output:

- `<run-dir>\case_reconstruct_report.json`

This report summarizes metric trends and flags typical causes for "success up but reward flat/negative".
