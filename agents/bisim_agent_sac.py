import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.rl_utils as rl_utils
from agents.base_agent import SACAgent
from agents.transition_model import make_transition_model


class BisimAgent(SACAgent):
    """Bisimulation-based agent."""
    def __init__(self, obs_shape, action_shape, actor_action_max, cfg, L):
        super().__init__(obs_shape, action_shape, actor_action_max, cfg, L)
        self.transition_model_type = cfg.transition_model_type

        # stats
        self.mu_rd = 0
        self.mu_bd = 0
        self.mu_nzd = 0
        self.mu_zd = 0

        self.transition_model = make_transition_model(
            cfg.transition_model_type, cfg.encoder_feature_dim, action_shape,
            layer_width=cfg.transition_model_hidden_size,
            encoder_max_norm=self.encoder_max_norm, encoder_max_norm_ord=self.encoder_max_norm_ord
        ).to(cfg.device)

        if self.cfg.rp_zprime:
            self.reward_decoder = nn.Sequential(
            nn.Linear(cfg.encoder_feature_dim, cfg.reward_decoder_hidden_size),
            nn.LayerNorm(cfg.reward_decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.reward_decoder_hidden_size, 1)).to(cfg.device)
        else:
            self.reward_decoder = nn.Sequential(
                nn.Linear(cfg.encoder_feature_dim + action_shape[0], cfg.reward_decoder_hidden_size),
                nn.LayerNorm(cfg.reward_decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(cfg.reward_decoder_hidden_size, 1)).to(cfg.device)

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=cfg.decoder_lr,
            weight_decay=cfg.decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=cfg.encoder_lr
        )

        # Precompute indices for generating samples for metric learning
        if self.cfg.on_policy_metric_update:
            idx = torch.arange(cfg.num_parallel_envs)
            idx1, idx2 = torch.meshgrid(idx, idx)
            idx1 = idx1.reshape(-1)
            idx2 = idx2.reshape(-1)
            mask = idx1 < idx2
            self.idx1 = idx1[mask]
            self.idx2 = idx2[mask]

        self.train()

    def print_model_stats(self):
        super().print_model_stats()
        print(f'Reward decoder params count: {rl_utils.count_parameters(self.reward_decoder)}')
        print(f'Transition model params count: {rl_utils.count_parameters(self.transition_model)}')

    def log_metric_stats(self, bisimilarity, r_dist, z_dist, transition_dist, metric_loss, step, r_dist_l1=None):
        with torch.no_grad():
            self.mu_bd = 0.95 * self.mu_bd + 0.05 * bisimilarity.mean().item()
            self.mu_rd = 0.95 * self.mu_rd + 0.05 * r_dist.mean().item()
            self.mu_nzd = 0.95 * self.mu_nzd + 0.05 * transition_dist.mean().item()
            self.mu_zd = 0.95 * self.mu_zd + 0.05 * z_dist.mean().item()

            self.log('train_metric/mu_rd', self.mu_rd, step)
            self.log('train_metric/mu_bd', self.mu_bd, step)
            self.log('train_metric/mu_nzd', self.mu_nzd, step)
            self.log('train_metric/mu_zd', self.mu_zd, step)
            self.log('train_metric/mu_rd_div_bd', self.mu_rd / (self.mu_bd + 1e-9), step)
            self.log('train_metric/metric_loss', metric_loss, step)

    def generate_bisim_samples(self, z, z_target, reward, mu, sigma, r_var=None):
        """
        Generates pairs of embeddings, rewards, and next latents for metric learning.

        Parameters:
        - z: Tensor of shape (batch_size, num_features), embeddings from the encoder.
        - z_target: Tensor of shape (batch_size, num_features), embeddings from the target encoder.
        - reward: Tensor of shape (batch_size, 1), rewards associated with the transitions.
        - mu: Tensor of shape (batch_size, num_latents) or (ensemble_size, batch_size, num_latents).
        - sigma: Tensor of shape matching mu, or None.

        Returns:
        - z1, z2: Pairwise embeddings.
        - reward1, reward2: Pairwise rewards.
        - mu1, mu2: Pairwise next latent means.
        - sigma1, sigma2: Pairwise next latent variances (if applicable).
        """
        batch_size = z.size(0)
        
        if not self.cfg.on_policy_metric_update:
            # Random pairing without excluding self-pairs
            perm = np.random.permutation(batch_size)
            idx1, idx2 = torch.arange(batch_size), perm
        else:
            # Precomputed indices for all pairs excluding self-pairs
            idx1, idx2 = self.idx1, self.idx2

        # Pair embeddings and rewards
        z1, z2 = z[idx1], z_target[idx2]
        reward1, reward2 = reward[idx1], reward[idx2]
        if r_var is not None:
            r_var1, r_var2 = r_var[idx1], r_var[idx2]

        # Pair next latent means and variances
        if mu.ndim == 2:  # Shape (B, Z), no ensemble
            mu1, mu2 = mu[idx1], mu[idx2]
            sigma1, sigma2 = (sigma[idx1], sigma[idx2]) if sigma is not None else (None, None)
        elif mu.ndim == 3:  # Shape (E, B, Z), using an ensemble
            mu1, mu2 = mu[:, idx1], mu[:, idx2]
            sigma1, sigma2 = (sigma[:, idx1], sigma[:, idx2]) if sigma is not None else (None, None)
        else:
            raise NotImplementedError("Unsupported dimensionality for mu.")

        if r_var is None:
            return z1, z2, reward1, reward2, mu1, mu2, sigma1, sigma2
        else:
            return z1, z2, reward1, reward2, mu1, mu2, sigma1, sigma2, r_var1, r_var2

    def update_encoder(self, obs, next_obs, action, reward, step):
        """
        Update the metrics (encoders) using sampled rewards and transition models.
        """
        z = self.critic.encoder(obs)
        if self.cfg.use_target_encoder:
            z_target = self.critic_target.encoder(obs).detach()
        else:
            z_target = z

        # Get next latent states and rewards
        with torch.no_grad():
            if self.cfg.model_based_on_policy_metric_update:
                action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)

            if self.transition_model_type in ['', 'deterministic', 'ensemble_det']:
                pred_z_prime_sigma = None
                pred_z_prime_mu = self.transition_model.sample_prediction(torch.cat([z_target, action], dim=1))
            else:
                pred_z_prime_mu, pred_z_prime_sigma = self.transition_model(torch.cat([z_target, action], dim=1))

            if self.cfg.model_based_on_policy_metric_update:
                if self.cfg.rp_zprime:
                    reward_on_policy = self.reward_decoder(pred_z_prime_mu)
                else:
                    reward_on_policy = self.reward_decoder(torch.cat([z, action], dim=1))
                self.log('train_metric/reward_buffer_on_policy_gap', (reward - reward_on_policy).abs().mean(), step)
                reward = reward_on_policy

        z, z2, reward, reward2, pred_z_prime_mu, pred_z_prime_mu2, \
        pred_z_prime_sigma, pred_z_prime_sigma2 = self.generate_bisim_samples(
            z, z_target, reward, pred_z_prime_mu, pred_z_prime_sigma)

        # Get z_dist, r_dist, and z'_dist
        z_dist = self.metric_func(z, z2)
        r_dist = self.reward_dist(reward, reward2)
        if 'det' in self.transition_model_type or self.transition_model_type == '':
            transition_dist = self.metric_func(pred_z_prime_mu, pred_z_prime_mu2)
        else:
            transition_dist = self.prob_metric_func(
                pred_z_prime_mu, pred_z_prime_mu2, pred_z_prime_sigma, pred_z_prime_sigma2)
            if step % 500 == 0:
                self.log('train_metric/z_prime_mu_l2norm', pred_z_prime_mu.norm(p=2, dim=-1).mean(), step)
                self.log('train_metric/z_prime_sigma_l2norm', pred_z_prime_sigma.norm(p=2, dim=-1).mean(), step)

        # Compute metric loss
        bisimilarity = self.cfg.c_R * r_dist + self.cfg.c_T * transition_dist
        if self.cfg.metric_loss_type == 'huber':
            loss = F.smooth_l1_loss(z_dist, bisimilarity.detach(), reduction='mean')
        else:
            loss = (z_dist - bisimilarity.detach()).pow(2).mean()

        # Stats
        r_dist_l1 = torch.abs(reward - reward2)
        self.log_metric_stats(bisimilarity, r_dist, z_dist, transition_dist, loss, step, r_dist_l1)
        return loss

    def update_transition_reward_model(self, obs, action, next_obs, reward, step):
        """
        Implementing (detachable) RP + ZP losses.
        """
        transition_loss = 0.
        if self.cfg.zp:
            z_zp = self.critic.encoder(obs)
            if self.cfg.zp_detach_encoder: z_zp = z_zp.detach()
            pred_z_prime_mu, pred_z_prime_sigma = self.transition_model(torch.cat([z_zp, action], dim=1))
            if pred_z_prime_sigma is None:
                pred_z_prime_sigma = torch.ones_like(pred_z_prime_mu)

            z_prime = self.critic.encoder(next_obs)
            diff = (pred_z_prime_mu - z_prime.detach()) / pred_z_prime_sigma
            transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_z_prime_sigma))
            self.log('train_ae/transition_loss', transition_loss.item(), step)

        reward_loss = 0.
        if self.cfg.rp:
            z_rp = self.critic.encoder(obs)
            if self.cfg.rp_detach_encoder: z_rp = z_rp.detach()
            if self.cfg.rp_zprime:
                pred_z_prime = self.transition_model.sample_prediction(torch.cat([z_rp, action], dim=1))
                pred_next_reward = self.reward_decoder(pred_z_prime)
            else:
                pred_next_reward = self.reward_decoder(torch.cat([z_rp, action], dim=1))
            reward_loss = F.mse_loss(pred_next_reward, reward)
            self.log('train_ae/reward_loss', reward_loss.item(), step)
        total_loss = transition_loss + reward_loss
        return total_loss

    def update(self, batch_transition, step, **kwargs):
        obs, action, _, reward, next_obs, not_done = batch_transition
        if 'on_policy_transition' in kwargs and kwargs['on_policy_transition'] is not None:
            obs_op, action_op, reward_op, next_obs_op = kwargs['on_policy_transition']
            obs_op = torch.as_tensor(obs_op, device=self.device).float()
            action_op = torch.as_tensor(action_op, device=self.device).float()
            reward_op = torch.as_tensor(reward_op, device=self.device).float()
            next_obs_op = torch.as_tensor(next_obs_op, device=self.device).float()

        self.log('train/batch_reward', reward.mean(), step)

        bisim_coef = self.cfg.bisim_coef if step > self.cfg.bisim_start_step else 0

        self.update_critic(obs, action, reward, next_obs, not_done, step)
        transition_reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, step) if self.cfg.rzp_coef > 0 else 0
        if self.cfg.bisim_coef <= 0:
            encoder_loss = 0
        elif 'on_policy_transition' in kwargs and kwargs['on_policy_transition'] is not None:
            encoder_loss = self.update_encoder(obs_op, next_obs_op, action_op, reward_op, step)
        else:
            encoder_loss = self.update_encoder(obs, next_obs, action, reward, step)
        total_loss = bisim_coef * encoder_loss + self.cfg.rzp_coef * transition_reward_loss
        if self.cfg.report_grad_norm:
            encoder_parameters = [p for p in self.critic.encoder.parameters() if p.requires_grad]
            self.grad_norm_analysis(bisim_coef * encoder_loss, encoder_parameters, step, label='metric_loss')
            self.grad_norm_analysis(self.cfg.rzp_coef * transition_reward_loss, encoder_parameters, step, label='transition_reward_loss')
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)

        self.soft_update_params(step)