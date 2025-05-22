import numpy as np
import torch
import os
import time
import gc
import functools
from copy import deepcopy
from gymnasium.vector import AsyncVectorEnv

import environments.dmc2gym as dmc2gym
import utils.rl_utils as rl_utils
from environments.gym_utils.wrappers import PhysicalStateWrapper, GaussianNoiseWrapper, GaussianRandomProjectionWrapper
from trainers.base_trainer import BaseTrainer, make_agent
from environments.gym_utils.penv import worker_shared_memory_no_truncation_reset
from environments.gym_utils.noise_utils import recover_obs, random_proj, append_white_noise, generate_full_rank_matrix, get_inverse_matrix


def make_env(cfg, seed, bg_source='unspecified', is_eval=True, proj_matrix=None, inv_proj_matrix=None):
    env_ = dmc2gym.make(
        domain_name=cfg.domain_name,
        task_name=cfg.task_name,
        resource_files=cfg.eval_resource_files if is_eval else cfg.resource_files,
        noise_source='none',
        total_frames=cfg.eval_total_frames if is_eval else cfg.total_frames,
        seed=seed,
        visualize_reward=False,
        from_pixels=(cfg.agent.encoder_type == 'pixel'),
        height=cfg.image_size,
        width=cfg.image_size,
        frame_skip=cfg.action_repeat,
    )
    if bg_source != 'unspecified':
        env_.unwrapped.set_bg_source(bg_source, cfg.noise_source)
    env_ = PhysicalStateWrapper(env_)
    if cfg.noise_source == 'noise':
        env_ = GaussianNoiseWrapper(env_, cfg.noise_dim, cfg.noise_std)
    elif cfg.noise_source == 'random_proj':
        noise_mean = cfg.eval_noise_mean if is_eval else 0.0
        env_ = GaussianRandomProjectionWrapper(env_, cfg.noise_dim, cfg.noise_std, proj_matrix, inv_proj_matrix, noise_mean)
    return env_


def make_env_fn(cfg, seed, bg_source='unspecified', is_eval=True, proj_matrix=None, inv_proj_matrix=None):
    return functools.partial(make_env, cfg, seed, bg_source, is_eval, proj_matrix, inv_proj_matrix)


class DMCStateTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init(self, cfg):
        print(f'Trainer: {cfg.env_name}, Domain: {cfg.domain_name}, Task: {cfg.task_name}')

        # Miscellaneous init
        os.environ["MUJOCO_GL"] = cfg.MUJOCO_GL
        torch.backends.cuda.matmul.allow_tf32 = True

        if cfg.noise_source == 'random_proj':
            # To get the true_obs_dim, we need to create a dummy env
            dummy_env = make_env(cfg, seed=cfg.seed, is_eval=False)
            matrix_m = dummy_env.true_obs_dim + cfg.noise_dim
            matrix_n = matrix_m if cfg.random_proj_dim == 'identical' else cfg.random_proj_dim
            print(f'Generating random projection matrix with shape {matrix_n} * {matrix_m}...')
            self.proj_matrix = generate_full_rank_matrix(matrix_n, matrix_m, cfg.random_proj_std)
            self.inv_proj_matrix = get_inverse_matrix(self.proj_matrix)
            print(np.round(self.proj_matrix, 2))
            del dummy_env
        else:
            self.proj_matrix = None
            self.inv_proj_matrix = None

        # Create envs
        if cfg.use_vectorized_training_env:
            self.env = AsyncVectorEnv([make_env_fn(
                cfg, seed=cfg.seed, is_eval=False, proj_matrix=self.proj_matrix, inv_proj_matrix=self.inv_proj_matrix
                ) for _ in range(cfg.num_parallel_envs)], worker=worker_shared_memory_no_truncation_reset)
            self.true_obs_dim = self.env.get_attr('true_obs_dim')[0]
            self.physical_state_dim = self.env.get_attr('physical_state_dim')[0]
            train_bg_source = deepcopy(self.env.call(name='get_bg_source')[0])
        else:
            self.cfg.num_parallel_envs = 1
            self.env = make_env(cfg, seed=cfg.seed, is_eval=False,
                                proj_matrix=self.proj_matrix, inv_proj_matrix=self.inv_proj_matrix)
            self.true_obs_dim = self.env.true_obs_dim
            self.physical_state_dim = self.env.physical_state_dim
            train_bg_source = self.env.get_bg_source()

        if self.cfg.num_eval_episodes == 1:
            self.eval_env = make_env(cfg, seed=cfg.seed, is_eval=False,
                                     proj_matrix=self.proj_matrix, inv_proj_matrix=self.inv_proj_matrix)
            self.max_episode_steps = self.eval_env._max_episode_steps
        else:
            self.eval_env = AsyncVectorEnv([make_env_fn(
            cfg, seed=cfg.seed+cfg.num_parallel_envs+1+i, proj_matrix=self.proj_matrix, inv_proj_matrix=self.inv_proj_matrix
            ) for i in range(cfg.num_eval_episodes)], worker=worker_shared_memory_no_truncation_reset)
            self.max_episode_steps = self.eval_env.get_attr('_max_episode_steps')[0]

        self.homo_eval_env = None
        self.eval_train_env = AsyncVectorEnv([make_env_fn(
            cfg, seed=cfg.seed+cfg.num_parallel_envs+1+i, bg_source=deepcopy(train_bg_source),
            proj_matrix=self.proj_matrix, inv_proj_matrix=self.inv_proj_matrix
            ) for i in range(cfg.num_eval_episodes)], worker=worker_shared_memory_no_truncation_reset)

        if cfg.agent.encoder_type.startswith('pixel'):
            raise NotImplementedError

        obs_space = self.env.single_observation_space if cfg.use_vectorized_training_env else self.env.observation_space
        act_space = self.env.single_action_space if cfg.use_vectorized_training_env else self.env.action_space
        assert abs(act_space.low.min()) == act_space.high.max()

        # Print info
        print(f'Obs dim: {self.true_obs_dim}, Physical state dim: {self.physical_state_dim}')
        print(f'Distractor: {cfg.noise_source}, Noise dim: {cfg.noise_dim}, std: {cfg.noise_std}')

        # Create replay buffer
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
            clean_obs_shape=(self.true_obs_dim,),
            )

        # Create agent
        self.agent = make_agent(
            obs_shape=obs_space.shape,
            action_shape=act_space.shape,
            actor_action_max=act_space.high.max(),
            cfg=cfg.agent,
            L=self.L,
            max_reward=self.cfg.action_repeat,
            min_reward=0
        )
        self.agent.print_model_stats()

        # Load encoder
        if cfg.agent.load_encoder:
            model_dict = self.agent.actor.encoder.state_dict()
            encoder_dict = torch.load(cfg.load_encoder) 
            encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}
            self.agent.actor.encoder.load_state_dict(encoder_dict)
            self.agent.critic.encoder.load_state_dict(encoder_dict)

    def train(self):
        episode_reward, truncated = 0, True
        start_time = time.time()
        self.timer.start('all')
        for self._global_step in range(0, self.cfg.num_train_steps, self.cfg.num_parallel_envs):
            if truncated:
                if self.cfg.agent.decoder_type == 'inverse':
                    for i in range(1, self.cfg.k):  # fill k_obs with 0s if episode is truncated
                        self.replay_buffer.k_obses[self.replay_buffer.idx - i] = 0

                if self.global_step > 0:
                    time_elapsed = time.time() - start_time
                    self.log('train/episode_duration', time_elapsed)
                    fps = self.cfg.action_repeat * episode_step / time_elapsed
                    self.log('train/fps', fps)
                    start_time = time.time()
                    self.log('train/episode_reward', episode_reward)
                    self.train_eps_rew_queue.append(episode_reward)
                    self.L.dump(self.global_step)

                # evaluate agent periodically
                if self.global_episode % self.cfg.eval_freq == 0 and \
                    (self.cfg.eval_at_start or self.global_episode != 0):
                    # Log global time elapsed
                    global_duration = self.timer.get_global_elapsed_time() / 3600
                    print(f"--- Global time elapsed: {global_duration} hrs.")
                    self.log('eval/global_duration', global_duration)

                    start_time = time.time()
                    with self.timer.time('eval'):
                        self.eval()

                    self.log('eval/episode', self.global_episode, log_on_wandb=False)
                    eval_duration = time.time() - start_time
                    self.log('eval_debug/duration', eval_duration)
                    self.L.dump(self.global_step)
                    if self.cfg.save_snapshot: self.save_snapshot()
                    start_time = time.time()  # don't count eval time in fps

                    torch.cuda.empty_cache()
                    gc.collect()

                obs, _ = self.env.reset()
                truncated = False
                episode_reward = 0
                episode_step = 0
                self._global_episode += self.cfg.num_parallel_envs if self.cfg.use_vectorized_training_env else 1
                reward = np.zeros(self.cfg.num_parallel_envs) if self.cfg.use_vectorized_training_env else 0

                self.log('train/episode', self.global_episode, log_on_wandb=False)

            # Sample action for data collection
            with self.timer.time('inference'):
                if self.global_step < self.cfg.init_steps:
                    action = self.env.action_space.sample()
                else:
                    with rl_utils.eval_mode(self.agent):
                        action = self.agent.sample_action(obs, multiproc=self.cfg.use_vectorized_training_env)
                if self.cfg.enforce_same_states:
                    action = np.tile(action[0], (self.cfg.num_parallel_envs, 1))  # shape = (num_parallel_envs, 1*action_dim)

            # Record fps
            if self.global_step % self.cfg.profiling_steps == 0 and self.global_step > 0:
                self.timer.stop('all')
                time_elapsed = self.timer.get_elapsed_time('all')
                fps = self.cfg.action_repeat * self.cfg.profiling_steps / time_elapsed
                if self.cfg.profiling:
                    print(f'FPS: {fps:.4f}')
                    print(f'{self.timer}')
                self.timer.reset()
                self.timer.start('all')

            # Simulation
            curr_reward = reward
            with self.timer.time('simulation'):
                next_obs, reward, _, truncated, info = self.env.step(action)
                if not self.cfg.use_vectorized_training_env and info['discount'] != 1.0:
                    print(f"Warning: discount is not 1.0, but {info['discount']} at eps step {episode_step}")

            if self.cfg.use_vectorized_training_env:
                episode_reward += reward.mean()
            else: episode_reward += reward

            # Add transition to buffer
            with self.timer.time('rb_add'):
                self.replay_buffer.add(obs, action, curr_reward, reward, next_obs, truncated)
                if self.cfg.agent.decoder_type == 'inverse':
                    np.copyto(self.replay_buffer.k_obses[self.replay_buffer.idx - self.cfg.k], next_obs)

            # Run training update
            if self.global_step >= self.cfg.init_steps:
                num_updates = self.calculate_num_updates()
                for _ in range(num_updates):
                    with self.timer.time('rb_sample'):
                        if self.cfg.agent.decoder_type == 'inverse':
                            batch_transition = self.replay_buffer.sample(k=True)
                        else:
                            batch_transition = self.replay_buffer.sample()
                        if self.cfg.agent.on_policy_metric_update:
                            assert self.cfg.use_vectorized_training_env
                            on_policy_batch_transition = (obs, action, reward, next_obs)
                        else: on_policy_batch_transition = None

                    with self.timer.time('update'):
                        self.agent.update(batch_transition, self.global_step, on_policy_transition=on_policy_batch_transition)

            obs = next_obs
            truncated = truncated[0] if self.cfg.use_vectorized_training_env else truncated
            episode_step += self.cfg.num_parallel_envs

        self.terminate()

    def eval(self, embed_viz_dir=None):
        eval_bism_states = self.cfg.eval_bism_states and self.cfg.noise_source is not None and (
            self.cfg.noise_source in ['noise', 'color', 'random_proj'])

        # Initialize lists for embedding visualization
        obses_for_emb = []
        values = []
        embeddings = []

        # Initialize lists for bisimilar state evaluation
        eval_multiproc = self.cfg.num_eval_episodes > 1

        # Reset environments
        obs, _ = self.eval_env.reset()
        if self.homo_eval_env: obs_homo, _ = self.homo_eval_env.reset()
        if self.eval_train_env: obs_train, _ = self.eval_train_env.reset()

        # Initialize variables for episode tracking
        truncated = False
        episode_reward = np.zeros(self.cfg.num_eval_episodes)
        if self.homo_eval_env: episode_reward_homo = np.zeros(self.cfg.num_eval_episodes)
        if self.eval_train_env: episode_reward_train = np.zeros(self.cfg.num_eval_episodes)
        episode_step = 0
        while not truncated:
            episode_step += 1
            with rl_utils.eval_mode(self.agent):
                action = self.agent.select_action(obs, multiproc=eval_multiproc)
                if self.homo_eval_env: action_homo = self.agent.select_action(obs_homo, multiproc=eval_multiproc)
                if self.eval_train_env: action_train = self.agent.select_action(obs_train, multiproc=eval_multiproc)

            if embed_viz_dir:
                self._record_embeddings(obs, action, obses_for_emb, values, embeddings)

            obs, reward, _, truncated, info = self.eval_env.step(action)
            if self.homo_eval_env:
                obs_homo, reward_homo, _, _, _ = self.homo_eval_env.step(action_homo)
            if self.eval_train_env: 
                obs_train, reward_train, _, _, _ = self.eval_train_env.step(action_train)
            truncated = truncated[0] if not isinstance(truncated, bool) else truncated  # Assume all envs truncated simultaneously
            # Make sure the types match as Asyncvecenv will not convert info[xxx] to np.float32
            clean_obs = np.stack(info['clean_obs']).astype(np.float32)
            if eval_bism_states: self.eval_obs_mem.add(obs, clean_obs)

            episode_reward += reward
            if self.homo_eval_env: episode_reward_homo += reward_homo
            if self.eval_train_env: episode_reward_train += reward_train

        # Compute and log mean episode rewards
        mean_eps_reward = episode_reward.mean()
        if self.homo_eval_env: mean_eps_reward_homo = episode_reward_homo.mean()
        if self.eval_train_env: mean_eps_reward_train = episode_reward_train.mean()
        self.log('eval/episode_reward', mean_eps_reward)
        if self.homo_eval_env: self.log('eval_debug/episode_reward_homogeneous', mean_eps_reward_homo)
        if self.eval_train_env: self.log('eval_debug/episode_reward_training_env', mean_eps_reward_train)

        # Log generalization reward losses
        if len(self.train_eps_rew_queue) > 0:
            # Compute the sliding window average of episode reward in training (stochastic policy)
            train_eps_rew = sum(self.train_eps_rew_queue) / len(self.train_eps_rew_queue)
        else: train_eps_rew = 0
        self.log('eval_debug/train_eval_episode_reward_gap', train_eps_rew - mean_eps_reward)
        if self.eval_train_env:
            # Using the training env for evaluation (deterministic policy) - Using the eval env for evaluation
            self.log('eval/generalization_reward_gap', mean_eps_reward_train - mean_eps_reward)

        # Eval denoising
        if eval_bism_states:
            eval_encoder_result = self.eval_encoder()
            self.eval_obs_mem.clear()
        else:
            eval_encoder_result = None

        # Reporting result for creating performance table
        if self.homo_eval_env is None: mean_eps_reward_homo = None
        self._record_and_report_performance(mean_eps_reward, mean_eps_reward_homo, eval_encoder_result)

        if embed_viz_dir:
            dataset = {'obs': obses_for_emb, 'values': values, 'embeddings': embeddings}
            torch.save(dataset, os.path.join(embed_viz_dir,
                    'train_dataset_{}.pt'.format(self.global_step)))

    def eval_encoder(self):
        buffer = self.eval_obs_mem
        encoder = self.agent.encoder_metric if 'isolated-metric' in self.cfg.agent.name else self.agent.critic.encoder
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
            _positive_obses = [self._add_noise_to_obs(clean_obs) for _ in range(pos_neg_batch_size)]
            _positive_train_obses = [self._add_noise_to_obs(clean_obs, is_ood=False) for _ in range(pos_neg_batch_size)]
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
                    pos_score, neg_score, pos_score_L2, neg_score_L2 = self.eval_bisimilar_states(
                        anchor_obses, positive_obses, negative_obses, self.agent.metric_func,
                        encoder)  # TODO: action distance
                    pos_score_train, _, pos_score_L2_train, _ = self.eval_bisimilar_states(
                        anchor_obses, positive_train_obses, negative_obses, self.agent.metric_func,
                        encoder)
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
            result = self._log_eval_bism(pos_score_sum, neg_score_sum,
                                         pos_score_L2_sum, neg_score_L2_sum, n_batch, identifier='_ood')
            result_id = self._log_eval_bism(pos_score_train_sum, neg_score_sum,
                                    pos_score_train_L2_sum, neg_score_L2_sum, n_batch, identifier='_id')
            result = {**result, **result_id}
        else:
            result = None
            print("Warning: No batches were processed for evaluation.")
        return result

    def _get_clean_obs(self, obs):
        if self.cfg.noise_source == 'noise':
            return obs[:self.true_obs_dim]
        elif self.cfg.noise_source == 'random_proj':
            inversed_obs = recover_obs(obs, self.inv_proj_matrix)
            clean_obs = inversed_obs[:self.true_obs_dim]
            return clean_obs
        return obs

    def _add_noise_to_obs(self, clean_obs, is_ood=True):
        if self.cfg.noise_dim == 0:
            return clean_obs
        if self.cfg.noise_source == 'noise':
            if not is_ood:
                noised_obs = append_white_noise(clean_obs, self.cfg.noise_dim, self.cfg.noise_std)
            else:
                noised_obs = append_white_noise(clean_obs, self.cfg.noise_dim, self.cfg.noise_std, self.cfg.eval_noise_mean)
        elif self.cfg.noise_source == 'random_proj':
            if not is_ood:
                noised_obs = append_white_noise(clean_obs, self.cfg.noise_dim, self.cfg.noise_std)
            else:
                noised_obs = append_white_noise(clean_obs, self.cfg.noise_dim, self.cfg.noise_std, self.cfg.eval_noise_mean)
            noised_obs = random_proj(noised_obs, self.proj_matrix)
        return noised_obs