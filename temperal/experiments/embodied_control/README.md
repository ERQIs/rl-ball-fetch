# Embodied Control

This branch isolates the next step beyond `landing_world_model`.

The goal is to move from:

- observation prefix -> latent summary -> landing prediction

toward:

- visual observation + car motion state + action-conditioned memory
- online recurrent control instead of observe-then-act
- a model that can plug back into `rl_be`

Recommended layout:

- `notes/`: design notes and decision records
- `configs/`: experiment-local configs
- `scripts/`: training and evaluation entry points
- `src/models/`: embodied world-model and policy modules
- `src/datasets/`: supervised or offline sequence datasets for warmup stages
- `reports/`: run summaries and comparisons

Current status:

- branch scaffold created on `2026-03-31`
- architecture design discussion is the next step
