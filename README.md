# Understanding the Effectiveness of Learning Behavioral Metrics in Deep Reinforcement Learning
A framework that affords the comparison of *different behavioral metrics learning methods* in deep reinforcement learning.

## Installation
- Python (3.7.0 or higher recommended, and [Anaconda](https://www.anaconda.com/) recommended to set up a new environment)

- Install packages in `requirements.txt`: `pip install -r requirements.txt`


## To run the experiments
Default hyperparameters are stored in `cfgs/` directory.

To run a minimal experiment, you can use the following command:
### DMC (State-based)
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

### DMC (Pixel-based)Trainer
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

## References
DeepMDP: [paper](https://proceedings.mlr.press/v97/gelada19a.html)
Deep Bisimulation Control (DBC): [paper](https://arxiv.org/abs/2006.10742), [code](https://github.com/facebookresearch/deep_bisim4control/)
Matching under Independent Couplings (MICo): [paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/fd06b8ea02fe5b1c2496fe1700e9d16c-Abstract.html), [code](https://github.com/google-research/google-research/blob/bb19948d367f3337c16176232e86069bf36b0bf5/mico)
Robust DBC: [paper](https://arxiv.org/abs/2110.14096), [code](https://github.com/metekemertas/RobustBisimulation)
Simple Distance-based State Representation (SimSR): [paper](https://arxiv.org/abs/2112.15303), [code](https://github.com/bit1029public/SimSR)
Reducing Approximation Gap (RAP): [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/eda9523faa5e7191aee1c2eaff669716-Abstract-Conference.html), [code](https://github.com/jianda-chen/RAP_distance)




