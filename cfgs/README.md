# 📊 About Hyperparameters

## Explanation of Key Hyperparameters for Reproducing

- `env`: Specifies the environment type. Choose from:
    - `dmc_pixel`: Pixel-based observations.
    - `dmc_state`: Low-dimensional proprioceptive state inputs.

- `domain_name`, `task_name`: Define the task from the DeepMind Control Suite (DMC). See the [DMC paper](https://arxiv.org/abs/1801.00690) for available domain-task pairs.

- `agent.name`: Specifies the agent configuration defined in `cfgs/agent_configs.yaml`. Includes both standard agents and isolated metric evaluation variants.

- `action_repeat`: Number of times each action is repeated. Task-specific values are detailed in Appendix Sec. E of the paper.

- `seed`: Random seed for reproducibility.

- `use_vectorized_training_env`: Whether to use vectorized training environments (`true` or `false`).

- `noise_source`: Specifies the noise type applied during training:
    - **State-based DMC**:
        - `noise`: IID Gaussian noise.
        - `random_proj`: IID Gaussian noise followed by random projection.
    - **Pixel-based DMC**:
        - `none`: Clean background.
        - `images_gray`, `images`: Natural image backgrounds (grayscale or RGB).
        - `video_gray`, `video`: Natural video backgrounds (grayscale or RGB).
        - `noise`: Per-pixel IID Gaussian noise.

- `noise_dim`: Dimensionality of the Gaussian noise vector (state-based only).

- `noise_std`: Standard deviation of the Gaussian noise (state-based only).

- `agent.encoder_post_processing`: Controls the encoder normalization scheme:
    - `layer_norm`: Applies LayerNorm after the encoder.
    - `no_layer_norm`: No normalization is applied.
    
    **Defaults**:
    - Pixel-based environments use `layer_norm` by default.
    - State-based environments use `no_layer_norm` by default.
    
    You may override the default to study the effect of normalization on representation learning.


> For full hyperparameter specifications and detailed explanations, see the YAML files in `cfgs/env/`.


## Priority of Hyperparameter Sources

Hyperparameters are defined across multiple configuration files, with the following priority (from highest to lowest):

1. **Agent-specific settings** (`cfgs/agent_configs.yaml`):
   Highest priority. Used to define individual agent configurations. To customize an agent, modify or extend one of the base configurations in this file.

2. **Command-line arguments** (`key=value` format):
   Override most settings *except* those in `agent_configs.yaml`. Note that these arguments do not override the agent-specific hyperparameters. For full agent customization, use `cfgs/agent_configs.yaml`.

3. **Domain-specific settings** (`cfgs/env/*.yaml`):
   Provide domain-specific environmental settings.

4. **High-level defaults** (`cfgs/config.yaml`):
   Global defaults for general experimental settings.

Each source builds upon the lower-priority layers, allowing flexible but controlled overrides.