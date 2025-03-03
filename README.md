# Understanding the Effectiveness of Learning Behavioral Metrics in Deep Reinforcement Learning
The official artifact for the paper "Understanding the Effectiveness of Learning Behavioral Metrics in Deep Reinforcement Learning", RLC 2025.

A framework that affords the comparison of *different behavioral metrics learning methods* in deep reinforcement learning.

## Installation
- Python (3.7.0 or higher recommended, and [Anaconda](https://www.anaconda.com/) recommended to set up a new environment)

- Install packages in `requirements.txt`: `pip install -r requirements.txt`


## To run the experiments
Default hyperparameters are stored in `cfgs/` directory.

To run a minimal experiment, you can use the following command:
### DMC (Pixel-based)
```
python main.py \
        env=dmc_state \
        domain_name=cheetah \
        task_name=run \
        agent.name=sac \
        seed=1 \
        noise_dim=32 \
        noise_std=8.0 \
        img_source=noise \
        use_vectorized_training_env=true
```

### DMC (State-based)Trainer
```
python main.py \
        env=dmc_pixel \
        domain_name=cheetah \
        task_name=run \
        agent.name=sac \
        seed=1 \
        img_source=video \
        use_vectorized_training_env=true
```

## Available Agents
See `cfgs/agent_configs.yaml` for the list of available agents. Pass `agent.name=<agent_name>` to the `main.py` script to use a specific agent.

## Hyperparameters
### Priority of Hyperparameters
- The default *high-level hyperparameters* are stored in `cfgs/config.yaml`.
- The default *domain-specific hyperparameters* are stored in `cfgs/env/*.yaml`.
- The default *agent-specific hyperparameters* are stored in `cfgs/agent_configs.yaml`.
- The priority of hyperparameter overridings is as follows:
    1. agent-specific hyperparameters in `cfgs/agent_configs.yaml`
    2. command line arguments
    3. domain-specific hyperparameters in `cfgs/env/*.yaml`
    4. high-level hyperparameters hyperparmeters specified by `cfgs/config.yaml`

### Explanation of Hyperparameters
See `cfgs/README.md` for the list of hyperparameters and their explanations.

## Code Structure
- `main.py`: The entry to run the experiments.
- `agents/`: Contains the implementations of the agents.
- `cfgs/`: Contains the default hyperparameters for the experiments.
- `environments/`: Contains the environments and their wrappers (based on [dm_control](https://github.com/google-deepmind/dm_control)).
- `trainers/`: Contains the training and the evaluation loop for different domains.
- `utils/`: Contains utility functions.


