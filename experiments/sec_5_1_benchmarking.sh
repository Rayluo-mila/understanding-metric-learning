#!/usr/bin/env bash

# Experiments for Section 5.1: Benchmarking Methods on Various Noise Settings.

# -------------------- Hyperparameters --------------------

# env: environment type.
#   Options:
#     - dmc_state → state-based DeepMind Control Suite
#     - dmc_pixel → pixel-based DeepMind Control Suite
env=dmc_state

# domain_name: domain within the DeepMind Control Suite.
#   Examples: "walker", "cheetah", "cartpole", "finger", etc.
#   See the paper for the full list.
domain_name=${domain_name}

# task_name: specific task in the selected domain.
#   Examples: "walk", "run", "swingup", "spin", etc.
#   See the paper for the full list.
task_name=${task_name}

# agent.name: RL agent configuration name (must match a key in cfgs/agent_configs.yaml).
#   Options:
#     - sac         → SAC agent
#     - deepmdp     → DeepMDP agent
#     - dbc         → DBC agent
#     - dbcnormed   → DBC-normed agent
#     - mico        → MICo agent
#     - simsr       → SimSR agent
#     - rap         → RAP agent
agent_name=${agent}

# seed: random seed for reproducibility. We show the seeds used in the paper:
#   State-based: 0, 1, 2, 11, 13, 17, 127, 131, 171, 1313, 7311, 13131
#   Pixel-based: 1, 13, 131, 1313, 13131
seed=${seed}

# note: identifier for annotating this run.
#   Examples: "experiment1", "${agent}_${noise_source}_${seed}"
note=${note}

# noise_dim: dimension of IID Gaussian noise.
#   Options: 2, 16, 32, 64, 128
noise_dim=${noise_dim}

# noise_std: standard deviation of IID Gaussian noise.
#   Options: 0.2, 1.0, 2.0, 4.0, 8.0
noise_std=${noise_std}

# wandb_proj_name: name of the Weights & Biases project.
#   Examples: "metric_learning_dmc", "baseline_runs"
wandb_proj_name=${wandb_proj_name}

# wandb_run_name: name of this run in W&B.
#   Suggestion: use agent, img_source, noise_dim, seed for uniqueness.
wandb_run_name=${agent}_${noise_source}_ndim${noise_dim}_${seed}

# save_wandb: whether to log metrics to W&B.
#   Options: true (enable logging), false (disable logging)
save_wandb=true

# noise_source: type of injected noise or background.
#   For env=dmc_state:
#     - none         → clean states
#     - noise        → IID Gaussian noise
#     - random_proj  → random projection of IID noise
#   For env=dmc_pixel:
#     - none         → clean pixels
#     - images       → natural images
#     - images_gray  → grayscale images
#     - video        → natural videos
#     - video_gray   → grayscale videos
#     - noise        → per-pixel IID Gaussian noise
noise_source=${noise_source}

# action_repeat: number of frames to repeat each action.
#   Options:
#     - 4 for most tasks
#     - 8 for cartpole (swingup, swingup_sparse)
#     - 2 for finger spin and walker (walk, run, stand)
action_repeat=${action_repeat}

# work_dir: root directory for logs, checkpoints, and summaries.
#   Examples: "./experiments", "/mnt/data/experiments"
work_dir='./experiments'

# use_vectorized_training_env: whether to use parallelized environments.
#   Options: true (recommended), false
use_vectorized_training_env=true

# replay_ratio: gradient steps per environment step.
#   Example: 0.2 → 1 grad step per 5 env steps
#   If using 10 parallel envs, 2 grad steps per step.
replay_ratio=0.2

# eval_freq: frequency (in episodes) for evaluation.
#   Example: 50 → evaluate every 50 episodes
eval_freq=50

# agent.report_grad_norm: whether to log gradient norms.
#   Options: true (enable), false (disable for speed)
report_grad_norm=true

# -------------------- Run --------------------

python main.py \
    env=${env} \
    domain_name=${domain_name} \
    task_name=${task_name} \
    agent.name=${agent_name} \
    seed=${seed} \
    note=${note} \
    noise_dim=${noise_dim} \
    noise_std=${noise_std} \
    wandb_proj_name=${wandb_proj_name} \
    wandb_run_name=${wandb_run_name} \
    save_wandb=${save_wandb} \
    noise_source=${noise_source} \
    action_repeat=${action_repeat} \
    work_dir=${work_dir} \
    use_vectorized_training_env=${use_vectorized_training_env} \
    replay_ratio=${replay_ratio} \
    eval_freq=${eval_freq} \
    agent.report_grad_norm=${report_grad_norm}