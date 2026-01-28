# REMAC-Kinetix

The simulation experiments' source code and implementation of paper [Real-Time Robot Execution with Masked Action Chunking](https://remac-async.github.io/).

## Installation

```bash
# Clone Kinetix submodule
git submodule update --init
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install dependencies
uv sync
```

## Pipeline

1. First follow the implementation of [RTC](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) to produce the base checkpoints:

```
# Train expert policies with RL. Checkpoints, videos, and stats are written to ./logs-expert/<wandb-run-name>
uv run src_lora/train_expert.py 

# Generate data using experts. Data is written back to `./logs-expert/<wandb-run-name>/data/`
uv run src_lora/generate_data.py --config.run-path ./logs-expert/<wandb-run-name>

# Train imitation learning policies
uv run src_lora/train_flow_base.py --config.run-path ./logs-expert/<wandb-run-name>

# Evaluate imitation learning policies if you want
uv run src_lora/eval_flow_no_lora.py --config.run-path ./logs-bc/<wandb-run-name> --output-dir <output-dir>
```

2. Change the `<wandb-run-name>` to `base_model` in `logs-bc`.
Then finetune the trained policies with:
```
bash run_all.sh
```
The above script will train and evaluate the 12 experiments in sequence.

Note: As there are 12 tasks, the number of tasks running should be divisible by the number of GPUs you use.