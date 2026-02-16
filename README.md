# REMAC-Kinetix

[![ArXiv](https://img.shields.io/badge/ArXiv-2601.20130-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.20130)
[![Webpage](https://img.shields.io/badge/Webpage-REMAC--Kinetix-2f80ed?style=for-the-badge&logo=googlechrome&logoColor=white)](https://remac-async.github.io/)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-8c1d40?style=for-the-badge&logo=openreview&logoColor=white)](https://openreview.net/forum?id=r0RGJ1j9on)

## Overview

**REMAC (Real-Time Robot Execution with Masked Action Chunking)** is for achieving real-time robot control through masked action chunking and asynchronous execution. This repository contains the official implementation and simulation experiments for the research paper [Real-Time Robot Execution with Masked Action Chunking](https://remac-async.github.io/).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Stage 1: Train Base Model](#stage-1-train-base-model)
  - [Stage 2: Fine-tune with LoRA](#stage-2-fine-tune-with-lora)
- [Update Log](#update-log)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Prerequisites

Before installation, ensure your system meets the following requirements:

- **GPU**: CUDA-compatible GPU with 12GB+ VRAM (recommended for training)
- **OS**: Test on Ubuntu 22.04+

## Installation

Follow these steps to set up the REMAC-Kinetix environment:

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/yourusername/REMAC-Kinetix.git
cd REMAC-Kinetix

# Alternatively, if already cloned, initialize the Kinetix submodule
git submodule update --init --recursive

# Install uv (fast Python package installer and resolver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all project dependencies
uv sync
```

**Verify Installation:**
```bash
# Check uv installation
uv --version

# Verify Python environment
uv run python --version
```

## Usage

The REMAC training pipeline consists of two main stages: (1) training a base model following the RTC methodology, and (2) fine-tuning with LoRA for improved performance.

### Stage 1: Train Base Model

This stage follows the [Real-Time Chunking (RTC)](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) methodology to produce base model checkpoints.

#### Step 1.1: Train Expert Policies with Reinforcement Learning

```bash
# Train expert policies using RL
# Outputs: checkpoints, videos, and training statistics
# Location: ./logs-expert/<wandb-run-name>
uv run src_lora/train_expert.py
```

#### Step 1.2: Generate Training Data

```bash
# Generate imitation learning data using trained expert policies
# Data will be saved to: ./logs-expert/<wandb-run-name>/data/
uv run src_lora/generate_data.py --config.run-path ./logs-expert/<wandb-run-name>
```

**Note:** Replace `<wandb-run-name>` with the actual run name from your Weights & Biases dashboard.

#### Step 1.3: Train Base Imitation Learning Model

```bash
# Train the base flow-matching policy via behavioral cloning
# Outputs: ./logs-bc/<wandb-run-name>
uv run src_lora/train_flow_base.py --config.run-path ./logs-expert/<wandb-run-name>
```

#### Step 1.4: Evaluate Base Model (Optional)

```bash
# Evaluate the base model performance before fine-tuning
uv run src_lora/eval_flow_no_lora.py \
  --config.run-path ./logs-bc/<wandb-run-name> \
  --output-dir <output-dir>
```

#### Step 1.5: Prepare for Fine-tuning

```bash
# Rename the base model checkpoint directory for the fine-tuning stage
mv ./logs-bc/<wandb-run-name> ./logs-bc/base_model
```

### Stage 2: Fine-tune with LoRA

After preparing the base model, fine-tune it using the REMAC approach with LoRA adaptation:

```bash
# Run the complete fine-tuning and evaluation pipeline
# This script trains and evaluates on all 12 tasks sequentially
bash run_all.sh
```

The `run_all.sh` script will:
1. Fine-tune the base model using LoRA on each task
2. Evaluate the fine-tuned models
3. Generate results and performance metrics

> **Important:** The number of tasks (12) should be divisible by the number of GPUs available. For multi-GPU setups, adjust the parallelization settings in `run_all.sh` accordingly.


## Update Log

- **Feb 12, 2026**: Reorganized codebase structure. Merged base model training and LoRA fine-tuning code into a unified pipeline. Please report any issues via GitHub Issues.
- **Jan 28, 2026**: Initial code release

## Acknowledgements

This project builds upon excellent prior work in robot learning and simulation:

- [**RTC (Real-Time Chunking)**](https://github.com/Physical-Intelligence/real-time-chunking-kinetix): Real-time action chunking methodology
- [**OpenPI**](https://github.com/Physical-Intelligence/openpi): Open-source robot learning framework
- [**Kinetix**](https://kinetix-env.github.io/): High-performance physics simulation platform

We are grateful to the authors and maintainers of these projects for their contributions to the research community.

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
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

## License

This project is released under the MIT License. 