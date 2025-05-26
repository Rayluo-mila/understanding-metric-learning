import numpy as np
import torch
import os
import time
import gc
import functools
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
from pathlib import Path
from gymnasium.vector import AsyncVectorEnv

import environments.dmc2gym as dmc2gym
from environments.dmc2gym.dmc_wrappers import add_noise
import utils.rl_utils as rl_utils
from environments.gym_utils.wrappers import FrameStack
from trainers.base_trainer import BaseTrainer, make_agent
from environments.gym_utils.penv import worker_shared_memory_no_truncation_reset


def make_env(cfg, seed, bg_source="unspecified", is_eval=True):
    noise_source = cfg.noise_source if bg_source == "unspecified" else None
    env_ = dmc2gym.make(
        domain_name=cfg.domain_name,
        task_name=cfg.task_name,
        resource_files=cfg.eval_resource_files if is_eval else cfg.resource_files,
        noise_source=noise_source,
        total_frames=cfg.eval_total_frames if is_eval else cfg.total_frames,
        seed=seed,
        visualize_reward=False,
        from_pixels=(cfg.agent.encoder_type == "pixel"),
        height=cfg.image_size,
        width=cfg.image_size,
        frame_skip=cfg.action_repeat,
    )
    if bg_source != "unspecified":
        env_.unwrapped.set_bg_source(bg_source, cfg.noise_source)
    stack_clean = True if is_eval else False
    env_ = FrameStack(env_, k=cfg.frame_stack, stack_clean=stack_clean)
    return env_


def make_env_fn(cfg, seed, bg_source="unspecified", is_eval=True):
    return functools.partial(make_env, cfg, seed, bg_source, is_eval)


class DMCPixelTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init(self, cfg):
        print(
            f"Trainer: {cfg.env_name}, Domain: {cfg.domain_name}, Task: {cfg.task_name}"
        )

        # Miscellaneous init
        os.environ["MUJOCO_GL"] = cfg.MUJOCO_GL
        torch.backends.cuda.matmul.allow_tf32 = True

        # Create envs
        if not cfg.agent.encoder_type.startswith("pixel"):
            raise NotImplementedError

        print("Init training envs...")
        if not cfg.use_vectorized_training_env:
            self.env = make_env(cfg, cfg.seed, is_eval=False)
            train_bg_source = deepcopy(self.env.unwrapped.get_bg_source())
        else:
            if cfg.enforce_same_states:  # Same seed, same physical state when resetting
                self.env = AsyncVectorEnv(
                    [
                        make_env_fn(cfg, seed=cfg.seed, is_eval=False)
                        for _ in range(cfg.num_parallel_envs)
                    ],
                    worker=worker_shared_memory_no_truncation_reset,
                )
                train_bg_source = deepcopy(self.env.call(name="get_bg_source")[0])
                self.cfg.training_env_homogeneous_level = 0
            elif (
                cfg.training_env_homogeneous_level == 0
            ):  # Heterogeneous noise sources for each environment
                self.env = AsyncVectorEnv(
                    [
                        make_env_fn(cfg, seed=cfg.seed + i, is_eval=False)
                        for i in range(cfg.num_parallel_envs)
                    ],
                    worker=worker_shared_memory_no_truncation_reset,
                )
                train_bg_source = deepcopy(self.env.call(name="get_bg_source")[0])
            elif (
                cfg.training_env_homogeneous_level == 1
            ):  # Uniform noise sources across all parallel environments
                self.individual_homo_train_env = make_env(
                    cfg, seed=cfg.seed, is_eval=False
                )
                train_bg_source = deepcopy(
                    self.individual_homo_train_env.unwrapped.get_bg_source()
                )
                self.env = AsyncVectorEnv(
                    [
                        make_env_fn(
                            cfg,
                            seed=cfg.seed + i,
                            is_eval=False,
                            bg_source=deepcopy(train_bg_source),
                        )
                        for i in range(cfg.num_parallel_envs)
                    ],
                    worker=worker_shared_memory_no_truncation_reset,
                )
                self.env.call(name="reset_bg_source")
                train_bg_source = deepcopy(self.env.call(name="get_bg_source")[0])
            elif (
                cfg.training_env_homogeneous_level == 2
            ):  # Synchronized noise backgrounds for all parallel environments
                self.individual_homo_train_env = make_env(
                    cfg, seed=cfg.seed, is_eval=False
                )
                train_bg_source = deepcopy(
                    self.individual_homo_train_env.unwrapped.get_bg_source()
                )
                self.env = AsyncVectorEnv(
                    [
                        make_env_fn(
                            cfg,
                            seed=cfg.seed + i,
                            is_eval=False,
                            bg_source=deepcopy(train_bg_source),
                        )
                        for i in range(cfg.num_parallel_envs)
                    ],
                    worker=worker_shared_memory_no_truncation_reset,
                )

        print("Init eval envs...")
        self.eval_env = AsyncVectorEnv(
            [
                make_env_fn(cfg, seed=cfg.seed + cfg.num_parallel_envs + 1 + i)
                for i in range(cfg.num_eval_episodes)
            ],
            worker=worker_shared_memory_no_truncation_reset,
        )
        self.eval_env.call(name="reset_bg_source")
        self.eval_env_noise_srcs = self.eval_env.call(name="get_bg_source")
        print("Init homogeneous (Level 1) eval envs...")
        self.individual_homo_eval_env = make_env(
            cfg, seed=cfg.seed + cfg.num_parallel_envs + 1
        )
        homo_eval_bg_source = self.individual_homo_eval_env.unwrapped.get_bg_source()
        self.homo_eval_env = AsyncVectorEnv(
            [
                make_env_fn(
                    cfg,
                    seed=cfg.seed + cfg.num_parallel_envs + 1 + i,
                    bg_source=deepcopy(homo_eval_bg_source),
                )
                for i in range(cfg.num_eval_episodes)
            ],
            worker=worker_shared_memory_no_truncation_reset,
        )
        self.eval_train_env = AsyncVectorEnv(
            [
                make_env_fn(
                    cfg,
                    seed=cfg.seed + cfg.num_parallel_envs + 1 + i,
                    bg_source=deepcopy(train_bg_source),
                )
                for i in range(cfg.num_eval_episodes)
            ],
            worker=worker_shared_memory_no_truncation_reset,
        )
        self.eval_train_env_noise_srcs = self.eval_train_env.call(name="get_bg_source")
        self.homo_eval_env.call(name="reset_bg_source")
        self.eval_train_env.call(name="reset_bg_source")

        obs_space = (
            self.env.single_observation_space
            if cfg.use_vectorized_training_env
            else self.env.observation_space
        )
        act_space = (
            self.env.single_action_space
            if cfg.use_vectorized_training_env
            else self.env.action_space
        )
        assert abs(act_space.low.min()) == act_space.high.max()

        # Create replay buffer
        self.max_episode_steps = self.eval_env.get_attr("_max_episode_steps")[0]
        self.replay_buffer = rl_utils.MemSaveReplayBuffer(
            obs_shape=obs_space.shape,
            action_shape=act_space.shape,
            capacity=cfg.replay_buffer_capacity,
            batch_size=cfg.agent.batch_size,
            device=self.device,
            nprocs=cfg.num_parallel_envs,
        )
        self.eval_obs_mem = rl_utils.ObsMemory(
            obs_shape=obs_space.shape,
            capacity=cfg.num_eval_episodes * self.max_episode_steps,
            batch_size=cfg.pos_neg_batch_size,
            device=self.device,
            nprocs=cfg.num_eval_episodes,
            clean_obs_shape=obs_space.shape,
        )

        # Create agent
        self.agent = make_agent(
            obs_shape=obs_space.shape,
            action_shape=act_space.shape,
            actor_action_max=act_space.high.max(),
            cfg=cfg.agent,
            L=self.L,
            max_reward=self.cfg.action_repeat,
            min_reward=0,
        )

        # Print info
        print(f"Distractor: {cfg.noise_source}")
        self.agent.print_model_stats()

        # Load encoder
        if cfg.agent.load_encoder:
            model_dict = self.agent.actor.encoder.state_dict()
            encoder_dict = torch.load(cfg.load_encoder)
            encoder_dict = {
                k[8:]: v for k, v in encoder_dict.items() if "encoder." in k
            }
            self.agent.actor.encoder.load_state_dict(encoder_dict)
            self.agent.critic.encoder.load_state_dict(encoder_dict)

    def train(self):
        episode_reward, truncated = 0, True
        start_time = time.time()
        self.timer.start("all")
        for self._global_step in range(
            0, self.cfg.num_train_steps, self.cfg.num_parallel_envs
        ):
            if truncated:
                if self.cfg.agent.decoder_type == "inverse":
                    for i in range(
                        1, self.cfg.k
                    ):  # fill k_obs with 0s if episode is truncated
                        self.replay_buffer.k_obses[self.replay_buffer.idx - i] = 0

                if self.global_step > 0:
                    time_elapsed = time.time() - start_time
                    self.log("train/episode_duration", time_elapsed)
                    fps = self.cfg.action_repeat * episode_step / time_elapsed
                    self.log("train/fps", fps)
                    start_time = time.time()
                    self.log("train/episode_reward", episode_reward)
                    self.train_eps_rew_queue.append(episode_reward)
                    self.L.dump(self.global_step)

                # evaluate agent periodically
                if self.global_episode % self.cfg.eval_freq == 0 and (
                    self.cfg.eval_at_start or self.global_episode != 0
                ):
                    # Log global time elapsed
                    global_duration = self.timer.get_global_elapsed_time() / 3600
                    print(f"--- Global time elapsed: {global_duration} hrs.")
                    self.log("eval/global_duration", global_duration)

                    start_time = time.time()
                    with self.timer.time("eval"):
                        self.eval()

                    self.log("eval/episode", self.global_episode, log_on_wandb=False)
                    eval_duration = time.time() - start_time
                    self.log("eval_debug/duration", eval_duration)
                    self.L.dump(self.global_step)
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                    start_time = time.time()  # don't count eval time in fps

                    torch.cuda.empty_cache()
                    gc.collect()

                obs, _ = self.env.reset()
                if self.cfg.save_bg_debug:
                    self.save_dmc_obs(obs, "reset")
                truncated = False
                episode_reward = 0
                episode_step = 0
                self._global_episode += (
                    self.cfg.num_parallel_envs
                    if self.cfg.use_vectorized_training_env
                    else 1
                )
                reward = (
                    np.zeros(self.cfg.num_parallel_envs)
                    if self.cfg.use_vectorized_training_env
                    else 0
                )

                self.log("train/episode", self.global_episode, log_on_wandb=False)

            # Sample action for data collection
            with self.timer.time("inference"):
                if self.global_step < self.cfg.init_steps:
                    action = self.env.action_space.sample()
                else:
                    with rl_utils.eval_mode(self.agent):
                        action = self.agent.sample_action(
                            obs, multiproc=self.cfg.use_vectorized_training_env
                        )
                if self.cfg.enforce_same_states:
                    action = np.tile(
                        action[0], (self.cfg.num_parallel_envs, 1)
                    )  # shape = (num_parallel_envs, 1*action_dim)

            # Record fps
            if (
                self.global_step % self.cfg.profiling_steps == 0
                and self.global_step > 0
            ):
                self.timer.stop("all")
                time_elapsed = self.timer.get_elapsed_time("all")
                fps = self.cfg.action_repeat * self.cfg.profiling_steps / time_elapsed
                if self.cfg.profiling:
                    print(f"FPS: {fps:.4f}")
                    print(f"{self.timer}")
                self.timer.reset()
                self.timer.start("all")

            # Simulation
            curr_reward = reward
            with self.timer.time("simulation"):
                next_obs, reward, _, truncated, info = self.env.step(action)
                if not self.cfg.use_vectorized_training_env and info["discount"] != 1.0:
                    print(
                        f"Warning: discount is not 1.0, but {info['discount']} at eps step {episode_step}"
                    )

            if self.cfg.use_vectorized_training_env:
                episode_reward += reward.mean()
            else:
                episode_reward += reward

            # Add transition to buffer
            with self.timer.time("rb_add"):
                self.replay_buffer.add(
                    obs, action, curr_reward, reward, next_obs, truncated
                )
                if self.cfg.agent.decoder_type == "inverse":
                    np.copyto(
                        self.replay_buffer.k_obses[self.replay_buffer.idx - self.cfg.k],
                        next_obs,
                    )

            # Run training update
            if self.global_step >= self.cfg.init_steps:
                num_updates = self.calculate_num_updates()
                for _ in range(num_updates):
                    with self.timer.time("rb_sample"):
                        if self.cfg.agent.decoder_type == "inverse":
                            batch_transition = self.replay_buffer.sample(k=True)
                        else:
                            batch_transition = self.replay_buffer.sample()
                        if self.cfg.agent.on_policy_metric_update:
                            assert self.cfg.use_vectorized_training_env
                            on_policy_batch_transition = (obs, action, reward, next_obs)
                        else:
                            on_policy_batch_transition = None

                    with self.timer.time("update"):
                        self.agent.update(
                            batch_transition,
                            self.global_step,
                            on_policy_transition=on_policy_batch_transition,
                        )

            obs = next_obs
            if self.cfg.save_bg_debug:
                self.save_dmc_obs(obs, "next")
            truncated = (
                truncated[0] if self.cfg.use_vectorized_training_env else truncated
            )
            episode_step += self.cfg.num_parallel_envs

        self.terminate()

    def eval(self, embed_viz_dir=None):
        eval_bism_states = (
            self.cfg.eval_bism_states
            and self.cfg.noise_source is not None
            and (
                self.cfg.noise_source in ["noise", "color"]
                or self.cfg.noise_source.startswith("video")
                or self.cfg.noise_source.startswith("images")
            )
        )

        # Initialize lists for embedding visualization
        obses_for_emb = []
        values = []
        embeddings = []

        # Initialize lists for bisimilar state evaluation
        eval_multiproc = self.cfg.num_eval_episodes > 1

        # Reset environments
        obs, info = self.eval_env.reset()
        if self.homo_eval_env:
            obs_homo, _ = self.homo_eval_env.reset()
        if self.eval_train_env:
            obs_train, _ = self.eval_train_env.reset()
        if self.cfg.save_bg_debug:
            self.save_dmc_obs(obs, "eval_reset")

        # Initialize variables for episode tracking
        truncated = False
        episode_reward = np.zeros(self.cfg.num_eval_episodes)
        if self.homo_eval_env:
            episode_reward_homo = np.zeros(self.cfg.num_eval_episodes)
        if self.eval_train_env:
            episode_reward_train = np.zeros(self.cfg.num_eval_episodes)
        episode_step = 0
        while not truncated:
            episode_step += 1
            with rl_utils.eval_mode(self.agent):
                action = self.agent.select_action(obs, multiproc=eval_multiproc)
                if self.homo_eval_env:
                    action_homo = self.agent.select_action(
                        obs_homo, multiproc=eval_multiproc
                    )
                if self.eval_train_env:
                    action_train = self.agent.select_action(
                        obs_train, multiproc=eval_multiproc
                    )

            if embed_viz_dir:
                self._record_embeddings(obs, action, obses_for_emb, values, embeddings)

            obs, reward, _, truncated, info = self.eval_env.step(action)
            if self.cfg.save_bg_debug:
                self.save_dmc_obs(obs, f"eval_next_st{episode_step}")
            if self.homo_eval_env:
                obs_homo, reward_homo, _, _, _ = self.homo_eval_env.step(action_homo)
                if self.cfg.save_bg_debug:
                    self.save_dmc_obs(obs_homo, f"eval_homo_next_st{episode_step}")
            if self.eval_train_env:
                obs_train, reward_train, _, _, _ = self.eval_train_env.step(
                    action_train
                )
                if self.cfg.save_bg_debug:
                    self.save_dmc_obs(obs_train, f"eval_train_next_st{episode_step}")
            truncated = (
                truncated[0] if not isinstance(truncated, bool) else truncated
            )  # Assume all envs truncated simultaneously
            # Make sure the types match as Asyncvecenv will not convert info[xxx] to np.uint8
            clean_obs = np.stack(info["clean_obs"]).astype(np.uint8)
            if eval_bism_states:
                self.eval_obs_mem.add(obs, clean_obs)

            episode_reward += reward
            if self.homo_eval_env:
                episode_reward_homo += reward_homo
            if self.eval_train_env:
                episode_reward_train += reward_train

        # Compute and log mean episode rewards
        mean_eps_reward = episode_reward.mean()
        if self.homo_eval_env:
            mean_eps_reward_homo = episode_reward_homo.mean()
        if self.eval_train_env:
            mean_eps_reward_train = episode_reward_train.mean()
        self.log("eval/episode_reward", mean_eps_reward)
        if self.homo_eval_env:
            self.log("eval_debug/episode_reward_homogeneous", mean_eps_reward_homo)
        if self.eval_train_env:
            self.log("eval_debug/episode_reward_training_env", mean_eps_reward_train)

        # Log generalization reward losses
        if len(self.train_eps_rew_queue) > 0:
            # Compute the sliding window average of episode reward in training (stochastic policy)
            train_eps_rew = sum(self.train_eps_rew_queue) / len(
                self.train_eps_rew_queue
            )
        else:
            train_eps_rew = 0
        self.log(
            "eval_debug/train_eval_episode_reward_gap", train_eps_rew - mean_eps_reward
        )
        if self.eval_train_env:
            # Using the training env for evaluation (deterministic policy) - Using the eval env for evaluation
            self.log(
                "eval/generalization_reward_gap",
                mean_eps_reward_train - mean_eps_reward,
            )

        # Eval denoising
        if eval_bism_states:
            eval_encoder_result = self.eval_encoder()
            self.eval_obs_mem.clear()
        else:
            eval_encoder_result = None

        # Reporting result for creating performance table
        if self.homo_eval_env is None:
            mean_eps_reward_homo = None
        self._record_and_report_performance(
            mean_eps_reward, mean_eps_reward_homo, eval_encoder_result
        )

        if embed_viz_dir:
            dataset = {"obs": obses_for_emb, "values": values, "embeddings": embeddings}
            torch.save(
                dataset,
                os.path.join(
                    embed_viz_dir, "train_dataset_{}.pt".format(self.global_step)
                ),
            )

    def eval_encoder(self):
        buffer = self.eval_obs_mem
        encoder = (
            self.agent.encoder_metric
            if "isolated-metric" in self.cfg.agent.name
            else self.agent.critic.encoder
        )
        anchor_batch_size = self.cfg.anchor_batch_size
        pos_neg_batch_size = self.cfg.pos_neg_batch_size

        batch_count = 0
        pos_score_sum, neg_score_sum, pos_score_L2_sum, neg_score_L2_sum = 0, 0, 0, 0
        pos_score_train_sum, pos_score_train_L2_sum = 0, 0
        n_batch = 0
        anchor_obses = []
        positive_obses = []
        positive_train_obses = []
        negative_obses = []

        # Ensure buffer has enough samples
        total_samples = (buffer.idx + buffer.capacity - 1) % buffer.capacity + 1
        if total_samples < anchor_batch_size:
            print("Warning: Not enough samples in buffer to evaluate.")
            return

        for i in range(total_samples):
            # Get anchor state
            obs = buffer.obses[i]
            anchor_obses.append(obs)
            clean_obs = buffer.clean_obses[i]

            # Generate positive observations in a vectorized manner
            _positive_obses = []
            _positive_train_obses = []
            for _ in range(pos_neg_batch_size):
                rnd_idx = np.random.randint(0, self.cfg.num_eval_episodes)
                pos_obs = self._add_noise_to_obs(
                    clean_obs, self.eval_env_noise_srcs[rnd_idx]
                )
                pos_train_obs = self._add_noise_to_obs(
                    clean_obs, self.eval_train_env_noise_srcs[rnd_idx]
                )
                _positive_obses.append(pos_obs)
                _positive_train_obses.append(pos_train_obs)
            positive_obses.append(_positive_obses)
            positive_train_obses.append(_positive_train_obses)

            # Sample negative observations
            _negative_obses = buffer.sample(pos_neg_batch_size)
            negative_obses.append(_negative_obses)

            batch_count += 1
            if batch_count == anchor_batch_size:
                batch_count = 0
                n_batch += 1
                with rl_utils.eval_mode(self.agent):
                    pos_score, neg_score, pos_score_L2, neg_score_L2 = (
                        self.eval_bisimilar_states(
                            anchor_obses,
                            positive_obses,
                            negative_obses,
                            self.agent.metric_func,
                            encoder,
                        )
                    )
                    pos_score_train, _, pos_score_L2_train, _ = (
                        self.eval_bisimilar_states(
                            anchor_obses,
                            positive_train_obses,
                            negative_obses,
                            self.agent.metric_func,
                            encoder,
                        )
                    )
                pos_score_sum += pos_score
                neg_score_sum += neg_score
                pos_score_L2_sum += pos_score_L2
                neg_score_L2_sum += neg_score_L2
                pos_score_train_sum += pos_score_train
                pos_score_train_L2_sum += pos_score_L2_train

                anchor_obses.clear()
                positive_obses.clear()
                positive_train_obses.clear()
                negative_obses.clear()

        # Log the denoising metrics
        if n_batch > 0:
            result = self._log_eval_bism(
                pos_score_sum,
                neg_score_sum,
                pos_score_L2_sum,
                neg_score_L2_sum,
                n_batch,
                identifier="_ood",
            )
            result_id = self._log_eval_bism(
                pos_score_train_sum,
                neg_score_sum,
                pos_score_train_L2_sum,
                neg_score_L2_sum,
                n_batch,
                identifier="_id",
            )
            result = {**result, **result_id}
        else:
            result = None
            print("Warning: No batches were processed for evaluation.")
        return result

    def _add_noise_to_obs(self, clean_obs, noise_source, noise_vid_loc=None):
        """
        If noise_vid_loc is specified, relocate the noise backgrounds to avoid using the same ones.
        Otherwise, randomly sample some consecutive noise backgrounds.
        """
        # Precompute slices
        slices = [
            (self.cfg.frame_stack * j, self.cfg.frame_stack * (j + 1))
            for j in range(self.cfg.frame_stack)
        ]
        if noise_vid_loc:
            new_loc = (
                noise_vid_loc
                + np.random.randint(self.cfg.frame_stack, self.cfg.eval_total_frames)
            ) % self.cfg.eval_total_frames
        else:
            new_loc = (
                np.random.randint(self.cfg.frame_stack, self.cfg.eval_total_frames)
            ) % self.cfg.eval_total_frames
        noise_offsets = [
            new_loc - self.cfg.frame_stack + 1 + j for j in range(self.cfg.frame_stack)
        ]

        # Generate another_noised_obs using list comprehension and vectorized operations
        noised_obs = np.concatenate(
            [
                add_noise(clean_obs[st:ed, :, :], offset, noise_source)
                for (st, ed), offset in zip(slices, noise_offsets)
            ],
            axis=0,
        )
        return noised_obs

    def save_dmc_obs(self, obs, filename_suffix=""):
        if obs.ndim == 4:
            pass  # Already in the expected shape
        elif obs.ndim == 3:
            obs = obs[np.newaxis, ...]  # Unsqueeze to [1, *other dims]
        else:
            raise ValueError(f"Invalid observation shape: {obs.shape}")
        self._save_dmc_obs(obs, filename_suffix)

    def _save_dmc_obs(self, obs, filename_suffix=""):
        """
        Save the observations as a grid image for debugging.
        Input obs for DMC: (n_paralleled_envs, frame_stack * 3, 84, 84)
        The grid will have frames along the y-axis and environment indices along the x-axis.
        """
        dir = Path(self.cfg.save_bg_dir)
        dir.mkdir(parents=True, exist_ok=True)
        n_envs = obs.shape[0]
        frame_stack = self.cfg.frame_stack

        # Collect all images into a list of lists: images[frame_idx][env_idx]
        images = []
        for frame_idx in range(frame_stack):
            row_images = []
            for env_idx in range(n_envs):
                # Extract the image for the current frame and environment
                img_array = obs[env_idx, frame_idx * 3 : frame_idx * 3 + 3]
                img_array = img_array.transpose(1, 2, 0)
                img = Image.fromarray(img_array.astype(np.uint8))
                row_images.append(img)
            images.append(row_images)

        # Get image dimensions
        img_width, img_height = images[0][0].size

        # Define margins for labels
        left_margin = 50
        top_margin = 50
        h_spacing = 10  # Horizontal spacing between images
        v_spacing = 10  # Vertical spacing between images

        # Calculate the total size of the grid image including margins and spacing
        total_width = left_margin + n_envs * img_width + n_envs * h_spacing
        total_height = top_margin + frame_stack * img_height + frame_stack * v_spacing

        # Create a new image with space for all images and labels
        grid_img = Image.new("RGB", (total_width, total_height), "white")

        # Prepare to draw labels
        draw = ImageDraw.Draw(grid_img)
        try:
            font = ImageFont.truetype("Times New Roman.ttf", size=14)
        except IOError:
            # If Times New Roman is not available, fall back to default font
            font = ImageFont.load_default()

        # Add episode and step number at the top-left corner
        ep_step_text = f"Ep{self.global_episode} St{self.global_step} {filename_suffix}"
        text_width, text_height = draw.textsize(ep_step_text, font=font)
        x = 5  # Small padding from the left edge
        y = 5  # Small padding from the top edge
        draw.text((x, y), ep_step_text, fill="black", font=font)

        # Draw column labels (environment indices)
        for col_idx in range(n_envs):
            x = left_margin + col_idx * (img_width + h_spacing) + img_width // 2
            y = top_margin - 20
            label = f"Env {col_idx}"
            text_width, text_height = draw.textsize(label, font=font)
            draw.text((x - text_width // 2, y), label, fill="black", font=font)

        # Draw row labels (frame indices)
        for row_idx in range(frame_stack):
            x = 0
            y = top_margin + row_idx * (img_height + v_spacing) + img_height // 2
            label = f"Frame {row_idx+1}"
            text_width, text_height = draw.textsize(label, font=font)
            draw.text((x + 5, y - text_height // 2), label, fill="black", font=font)

        # Paste images into the grid
        for row_idx, row_images in enumerate(images):
            for col_idx, img in enumerate(row_images):
                x = left_margin + col_idx * (img_width + h_spacing)
                y = top_margin + row_idx * (img_height + v_spacing)
                grid_img.paste(img, (x, y))

        # Save the final image
        grid_img.save(
            dir / f"ep{self.global_episode}_st{self.global_step}_{filename_suffix}.png"
        )
        print(
            f'Saved observation grid image to {dir / f"ep{self.global_episode}_st{self.global_step}_{filename_suffix}.png"}'
        )
