import numpy as np

from gymnasium import Wrapper, spaces
import gym
from skimage.transform import resize
from collections import deque
from environments.gym_utils.noise_utils import random_proj, append_white_noise


class RewardActionWrapper(Wrapper):
    """
    Reduce action space, fix the reward non-Markovian issue, and no truncation
    """
    def __init__(self, env, n_actions=3):
        super(RewardActionWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(n_actions)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward = 1
        terminated = truncated or terminated
        truncated = 0
        return obs, reward, terminated, truncated, info
    

class TransposeWrapper(Wrapper):
    def __init__(self, env):
        super(TransposeWrapper, self).__init__(env)
        self.height = env.observation_space.shape[0]
        self.width = env.observation_space.shape[1]
        self.observation_space = spaces.Box(low=0, high=255, shape=[3, self.height, self.width], dtype=np.uint8)
        self._max_episode_steps = env.spec.max_episode_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._preprocess_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._preprocess_obs(obs)
        return obs, reward, terminated, truncated, info

    def _preprocess_obs(self, obs):
        # obs = resize(obs, (self.height, self.width)) * 255
        obs = np.transpose(obs, [2, 0, 1])
        return obs


class RGBArrayAsObservationWrapper(Wrapper):
    def __init__(self, env, height=84, width=84):
        super(RGBArrayAsObservationWrapper, self).__init__(env)
        self.height = height
        self.width = width
        self.observation_space = spaces.Box(low=0, high=255, shape=[3, height, width], dtype=np.uint8)
        self._max_episode_steps = env.spec.max_episode_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        if len(obs.shape) != 3:
            obs = self.env.render(mode="rgb_array")
            obs = self._preprocess_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if len(obs.shape) != 3:
            obs = self.env.render(mode="rgb_array")
            obs = self._preprocess_obs(obs)
        return obs, reward, terminated, truncated, info

    def _preprocess_obs(self, obs):
        obs = resize(obs, (self.height, self.width)) * 255
        obs = np.transpose(obs, [2, 0, 1])
        return obs
    

class RGBArrayAsObservationWrapperGym(gym.Wrapper):
    def __init__(self, env, height=84, width=84):
        super(RGBArrayAsObservationWrapperGym, self).__init__(env)
        self.height = height
        self.width = width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=[3, height, width], dtype=np.uint8)
        self._max_episode_steps = env.spec.max_episode_steps

    def reset(self, **kwargs):
        obs = self.env.reset()
        if len(obs.shape) != 3:
            obs = self.env.render(mode="rgb_array")
            obs = self._preprocess_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if len(obs.shape) != 3:
            obs = self.env.render(mode="rgb_array")
            obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _preprocess_obs(self, obs):
        obs = resize(obs, (self.height, self.width)) * 255
        obs = np.transpose(obs, [2, 0, 1])
        return obs


class FrameStack(Wrapper):
    def __init__(self, env, k, stack_clean=False):
        super().__init__(env)
        self._k = k
        self._frames = deque([], maxlen=k)
        self._store_clean_frames = False
        if stack_clean and env.unwrapped.img_source is not None and (
            env.unwrapped.img_source in ['noise', 'color'] or \
            env.unwrapped.img_source.startswith('video') or env.unwrapped.img_source.startswith('images')):
            self._store_clean_frames = True
            self._clean_frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self._k):
            self._frames.append(obs)
        if self._store_clean_frames:
            for _ in range(self._k):
                self._clean_frames.append(info['clean_obs'])
        return self._get_obs(), self._get_info(info)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._frames.append(obs)
        if self._store_clean_frames:
            self._clean_frames.append(info['clean_obs'])
        return self._get_obs(), reward, done, truncated, self._get_info(info)

    def _get_obs(self):
        """
        Assuming the frames are channel-first, (C, H, W).
        """
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

    def _get_info(self, info):
        """
        Assuming the frames are channel-first, (C, H, W).
        """
        if self._store_clean_frames:
            assert len(self._clean_frames) == self._k
            clean_frames = np.concatenate(list(self._clean_frames), axis=0)  # channel-first
            # assert np.array_equal(clean_frames[-3:, :, :], info['clean_obs'])
            info['clean_obs'] = clean_frames
        return info


class PhysicalStateWrapper(Wrapper):
    """
    Use the physical state as the observation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.state_observation_space
        self._max_episode_steps = env._max_episode_steps

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info


class GaussianNoiseWrapper(Wrapper):
    def __init__(self, env, noise_dim, noise_std=1., noise_mean=0):
        """
        Given an environment that produces vector observations, this environment wraps it and adds
        noise dimensions that are sampled from an isotropic Gaussian.
        @param env: env to be wrapped.
        @param noise_dim: wrapped observation will have original + noise_dim dimensions.
        @param noise_std: std of the Gaussian noise distribution used for extra dimensions.
        """
        super(GaussianNoiseWrapper, self).__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self.true_obs_dim = env.observation_space.shape[0]
        self.physical_state_dim = env.physical_state_space.shape[0]
        self.noisy_obs_dim = self.true_obs_dim + noise_dim
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.noisy_obs_dim),
            high=np.inf * np.ones(self.noisy_obs_dim))

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = self.add_noise(obs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.add_noise(obs)
        return obs, reward, done, truncated, info

    def add_noise(self, obs):
        noised_obs = append_white_noise(obs, self.noise_dim, self.noise_std, self.noise_mean)
        return noised_obs


class GaussianRandomProjectionWrapper(Wrapper):
    def __init__(self, env, noise_dim, noise_std, shared_proj_matrix, shared_inv_proj_matrix, noise_mean):
        """
        Given an environment that produces vector observations, this environment wraps it and
        projects the observation from R^m to a higher-dimensional space R^(m + noise_dim)
        using a Gaussian random matrix.
        
        @param env: Environment to be wrapped.
        @param noise_dim: Number of extra dimensions; the wrapped observation will have 
                          original observation dimension + noise_dim dimensions.
        @param noise_std: Standard deviation for the Gaussian random matrix entries.
        """
        super(GaussianRandomProjectionWrapper, self).__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self.physical_state_dim = env.physical_state_space.shape[0]
        self.true_obs_dim = env.observation_space.shape[0]
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.noisy_obs_dim = self.true_obs_dim + noise_dim

        self.proj_matrix = shared_proj_matrix
        self.inv_proj_matrix = shared_inv_proj_matrix

        # Update the observation space to reflect the new, higher dimensionality
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(self.noisy_obs_dim, dtype=np.float32),
            high=np.inf * np.ones(self.noisy_obs_dim, dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.add_noise(obs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        info['clean_obs'] = obs
        obs = self.add_noise(obs)
        return obs, reward, done, truncated, info

    def add_noise(self, obs):
        noised_obs = append_white_noise(obs, self.noise_dim, self.noise_std, self.noise_mean)
        noised_obs = random_proj(noised_obs, self.proj_matrix)
        return noised_obs