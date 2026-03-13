# RL Experiments

Each experiment should be isolated in its own folder:

- own config(s)
- own launcher script(s)
- own notes/readme
- outputs grouped under `rl_be/results/<experiment_name>/`

Current experiments:

- `exp_vb_0001`: PPO with pretrained visual backbone patch.
- `exp_vb_0002_blind`: PPO blind baseline (visual encoder outputs zeros).
- `exp_vb_0310`: Prompt 0310 backbone ablation suite (E1/E2/E3/E4).
