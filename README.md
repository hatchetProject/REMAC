# REMAC-Kinetix

[![ArXiv](https://img.shields.io/badge/ArXiv-2601.20130-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.20130)
[![Webpage](https://img.shields.io/badge/Webpage-REMAC--Kinetix-2f80ed?style=for-the-badge&logo=googlechrome&logoColor=white)](https://remac-async.github.io/)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-8c1d40?style=for-the-badge&logo=openreview&logoColor=white)](https://openreview.net/forum?id=r0RGJ1j9on)

The simulation experiments' source code and implementation of paper [Real-Time Robot Execution with Masked Action Chunking](https://remac-async.github.io/).

## Update log
- Jan 28, 2026: Initial code upload
- Feb 12, 2026: Reorganized the code. As I merged the base model training and LoRA finetuning code together, feel free to raise issues if there are problems.

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

```bash
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

```bash
bash run_all.sh
```

The above script will run the proposed method, and evaluate on the 12 tasks in sequence.

> Note: As there are 12 tasks, the number of tasks running should be divisible by the number of GPUs you use.


## Acknowledgement

Thanks to these amazing repositories: [RTC](https://github.com/Physical-Intelligence/real-time-chunking-kinetix), [openpi](https://github.com/Physical-Intelligence/openpi) and other inspiring works.

## Citation

If you find this work useful, please consider citing:
```
@misc{wang2026realtimerobotexecutionmasked,
      title={Real-Time Robot Execution with Masked Action Chunking}, 
      author={Haoxuan Wang and Gengyu Zhang and Yan Yan and Yuzhang Shang and Ramana Rao Kompella and Gaowen Liu},
      year={2026},
      eprint={2601.20130},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.20130}, 
}
```
