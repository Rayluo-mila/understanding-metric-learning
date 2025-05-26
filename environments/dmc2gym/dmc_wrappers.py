import os
import glob
import numpy as np
from gymnasium import spaces
from gymnasium.core import Env

try:
    from dm_control import manipulation, suite
except ImportError:
    print("Failed to import dm suite!")

try:
    from dm_env import specs
except ImportError:
    print("Failed to import specs from dm_env")

try:
    from environments.dmc2gym import natural_imgsource
except ImportError:
    print("Failed to import natural_imgsource from dmc2gym")


def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


def add_noise(obs, loc=None, bg_source=None):
    """
    Add noise to an observation in channel-first format (C, H, W).
    Can specify the location of noise background in the noise frame array.
    """
    assert obs.shape[0] == 3
    noised_obs = obs.transpose(1, 2, 0).copy()  # (C, H, W) -> (H, W, C)
    if bg_source is not None:
        mask = np.logical_and(
            (noised_obs[:, :, 2] > noised_obs[:, :, 1]),
            (noised_obs[:, :, 2] > noised_obs[:, :, 0]),
        )  # hardcoded for dmc
        bg = bg_source.get_image(loc=loc)
        noised_obs[mask] = bg[mask]
    noised_obs = noised_obs.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return noised_obs


class DMCWrapper(Env):
    def __init__(
        self,
        domain_name,
        task_name,
        resource_files,
        noise_source,
        total_frames,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
    ):
        assert (
            "random" in task_kwargs
        ), "Please specify a seed for deterministic behavior"
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        seed = task_kwargs.get("random", 1)

        self._initialize_noise_source(noise_source, resource_files, total_frames)

        # Create task
        if (domain_name, task_name) in suite.ALL_TASKS:
            self._env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                visualize_reward=visualize_reward,
                environment_kwargs=environment_kwargs,
            )
        else:
            suffix = "vision" if from_pixels else "features"
            name = f"{domain_name}_{task_name}_{suffix}"
            self._env = manipulation.load(name, seed=seed)

        # Define action and observation spaces
        self._define_spaces(from_pixels)

        # Set seed
        self._seed(seed=seed)

    def _initialize_noise_source(self, noise_source, resource_files, total_frames):
        self._noise_source = None if noise_source == "none" else noise_source
        if self._noise_source is None:
            self._bg_source = None
            return

        grayscale = "gray" in noise_source
        random_frame = "random" in noise_source
        shape2d = (self._height, self._width)

        if noise_source == "color":
            self._bg_source = natural_imgsource.RandomColorSource(shape2d)
        elif noise_source == "noise":
            self._bg_source = natural_imgsource.NoiseSource(shape2d)
        elif "images" in noise_source:
            files = glob.glob(os.path.expanduser(resource_files))
            assert files, f"Pattern {resource_files} does not match any files"
            # self._bg_source = natural_imgsource.RandomImageSource(shape2d, files, grayscale=grayscale, total_frames=total_frames)
            self._bg_source = natural_imgsource.RandomVideoSourceBgFixed(
                shape2d, files, grayscale=grayscale, total_frames=total_frames
            )
        elif "video" in noise_source:
            files = glob.glob(os.path.expanduser(resource_files))
            assert files, f"Pattern {resource_files} does not match any files"
            self._bg_source = natural_imgsource.RandomVideoSource(
                shape2d,
                files,
                grayscale=grayscale,
                random_frame=random_frame,
                total_frames=total_frames,
            )
        else:
            raise ValueError(f"noise_source {noise_source} not defined.")

    def _define_spaces(self, from_pixels):
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=self._true_action_space.shape,
            dtype=np.float32,
        )

        if from_pixels:
            self._observation_space = spaces.Box(
                low=0, high=255, shape=[3, self._height, self._width], dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        self._state_observation_space = _spec_to_box(
            self._env.observation_spec().values()
        )
        self._physical_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._env.physics.get_state().shape,
            dtype=np.float32,
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height, width=self._width, camera_id=self._camera_id
            ).transpose(
                2, 0, 1
            )  # channel-last obs
            return self.add_noise(obs.copy()), obs.copy()
        obs = _flatten_obs(time_step.observation)
        # obs = self.get_physical_state()
        # print('time_step.observation: ', time_step.observation)
        # print('physical state: ', self.get_physical_state())
        return obs, obs.copy()

    def _get_info(self, time_step, clean_obs):
        info = {"internal_state": self.get_physical_state()}
        info["discount"] = time_step.discount
        info["clean_obs"] = clean_obs
        if self.noise_source is not None and (
            self.noise_source.startswith("video")
            or self.noise_source.startswith("images")
        ):
            info["noise_vid_loc"] = self._bg_source.loc
        return info

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_observation_space(self):
        return self._state_observation_space

    @property
    def physical_state_space(self):
        return self._physical_state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def noise_source(self):
        if self._noise_source == "none":
            return None
        return self._noise_source

    def _seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def add_noise(self, obs, loc=None):
        return add_noise(obs, loc=loc, bg_source=self._bg_source)

    def step(self, action):
        if not self._norm_action_space.contains(action):
            print(
                f"action {action}, type {type(action)} NOT in norm action space {self._norm_action_space}"
            )
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        if not self._true_action_space.contains(action):
            print(f"action {action} not in true action space {self._true_action_space}")
        assert self._true_action_space.contains(action)
        reward = 0

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            truncated = time_step.last()
            if truncated:
                break
        obs, clean_obs = self._get_obs(time_step)
        return obs, reward, False, truncated, self._get_info(time_step, clean_obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_bg_source()
        time_step = self._env.reset()
        obs, clean_obs = self._get_obs(time_step)
        return obs, self._get_info(time_step, clean_obs)

    def get_physical_state(self):
        return self._env.physics.get_state().copy()

    def get_bg_source(self):
        return self._bg_source

    def reset_bg_source(self):
        if self._bg_source is not None:
            self._bg_source.reset()

    def set_bg_source(self, bg_source="unspecified", noise_source="unspecified"):
        if bg_source != "unspecified":
            self._bg_source = bg_source
            self._noise_source = noise_source

    def render(self, mode="rgb_array", height=None, width=None, camera_id=0):
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
