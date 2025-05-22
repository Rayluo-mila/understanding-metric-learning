import numpy as np
import sys
import gymnasium as gym
from multiprocessing import Pipe, Process
from gymnasium.vector.utils import write_to_shared_memory

import environments.dmc2gym as dmc2gym
from environments.gym_utils.wrappers import FrameStack


def worker(connection, env_fn):
    """
    Only support DMC as no reset() after done or truncated,
    as DMC envs always truncate after 1000 // action_repeat steps.
    Call reset() explicitly instead.
    """
    env = env_fn()
    connection.send((env.observation_space, env.action_space))
    while True:
        command, data = connection.recv()
        if command == "step":
            obs, reward, done, truncated, info = env.step(data)
            connection.send((obs, reward, done, truncated, info))
        elif command == "reset":
            obs, info = env.reset()
            connection.send((obs, info))
        elif command == "get_bg_source":
            bg_source = env.unwrapped.get_bg_source()
            connection.send(bg_source)
        else:
            raise NotImplementedError


def worker_shared_memory_no_truncation_reset(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    """
    Adapted from gymnasium's worker function, but without reset() after truncated.
    Call reset() manually instead. Suitable for AsyncVecEnv for DMC.
    """
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, info), True))

            elif command == "step":
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                if terminated:  # Reset after termination, but not trunction
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info["final_observation"] = old_observation
                    info["final_info"] = old_info
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    ((data[0] == observation_space, data[1] == env.action_space), True)
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, env_fns):
        assert len(env_fns) >= 1, "No environment creation functions provided."

        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.locals = []
        self.processes = []

        self.single_observation_space = None
        self.single_action_space = None

        # Start worker processes and obtain observation and action spaces
        for idx, env_fn in enumerate(self.env_fns):
            local, remote = Pipe()
            p = Process(target=worker, args=(remote, env_fn))
            p.daemon = True
            p.start()
            remote.close()
            self.locals.append(local)
            self.processes.append(p)

            # Receive observation and action spaces from the worker
            obs_space, act_space = local.recv()

            # Set observation and action spaces (ensure consistency)
            if self.single_observation_space is None:
                self.single_observation_space = obs_space
            else:
                assert obs_space == self.single_observation_space, (
                    f"Observation spaces differ between environments at index {idx}"
                )

            if self.single_action_space is None:
                self.single_action_space = act_space
            else:
                assert act_space == self.single_action_space, (
                    f"Action spaces differ between environments at index {idx}"
                )

    def sample_actions(self):
        return np.array([self.single_action_space.sample() for _ in range(self.num_envs)])

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [local.recv() for local in self.locals]
        obs, info = zip(*results)
        obs = np.array(obs)
        return obs, info

    def step(self, actions):
        """
        Only support DMC as no reset() after done or truncated,
        as DMC envs always truncate after 1000 // action_repeat steps.
        Call reset() explicitly instead.
        """
        for local, action in zip(self.locals, actions):
            local.send(("step", action))
        results = [local.recv() for local in self.locals]
        obs, reward, done, truncated, info = map(np.array, zip(*results))
        return obs, reward, done, truncated, info

    def _get_bg_source(self, env_idx):
        self.locals[env_idx].send(("get_bg_source", None))
        bg_source = self.locals[env_idx].recv()
        return bg_source
    
    def get_bg_source(self, env_idx=None):
        if env_idx is None:
            return [self._get_bg_source(i) for i in range(self.num_envs)]
        else:
            return self._get_bg_source(env_idx)

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def make_env(cfg, seed, bg_source='unspecified'):
    noise_source = cfg.noise_source if bg_source == 'unspecified' else None
    env = dmc2gym.make(
        domain_name=cfg.domain_name,
        task_name=cfg.task_name,
        resource_files=cfg.resource_files,
        noise_source=noise_source,
        total_frames=cfg.total_frames,
        seed=seed,
        visualize_reward=False,
        from_pixels=True,
        height=cfg.image_size,
        width=cfg.image_size,
        frame_skip=cfg.action_repeat,
    )
    if bg_source != 'unspecified':
        env.unwrapped.set_bg_source(bg_source, cfg.noise_source)
    env = FrameStack(env, k=cfg.frame_stack, stack_clean=False)
    return env


def main():
    cfg_dict = {'domain_name': 'cheetah',
                'task_name': 'run',
                'noise_source': 'video_gray',
                'total_frames': 1000,
                'action_repeat': 4,
                'image_size': 84,
                'num_eval_episodes': 3,
                'frame_stack': 3,
                'resource_files': 'environments/dmc2gym/res/train_video/*.mp4',
                'seed': 1
                }

    cfg = Config(**cfg_dict)
    env_list = [make_env(
        cfg, seed=cfg.seed+cfg.num_eval_episodes+1) for i in range(cfg.num_eval_episodes)]
    for i, env in enumerate(env_list):
        print(f'hash for env {i}: {env.unwrapped.get_bg_source().hash}')
    parallel_env = ParallelEnv(env_list)
    print('ParallelEnv created')
    obs, info = parallel_env.reset()
    for (o, i) in zip(obs, info):
        print(o.shape, i['internal_state'])
    for step in range(500):
        actions = parallel_env.sample_actions()
        obs, reward, done, truncated, info = parallel_env.step(actions)
        if done[0] or truncated[0]:
            obs, info = parallel_env.reset()
        print(f'step: {step}, reward: {reward}, truncated: {truncated}')
