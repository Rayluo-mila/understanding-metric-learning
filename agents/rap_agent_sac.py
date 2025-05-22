import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.bisim_agent_sac import BisimAgent


class StateRewardDecoder(nn.Module):
    def __init__(self, encoder_feature_dim, max_sigma=1e0, min_sigma=1e-4):
        super().__init__()
        self.trunck = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 2))

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

    def forward(self, x):
        y = self.trunck(x)
        sigma = y[..., 1:2]
        mu = y[..., 0:1]
        sigma = torch.sigmoid(sigma)  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def loss(self, mu, sigma, r, reduce='mean'):
        diff = (mu - r.detach()) / sigma
        if reduce == 'none':
            loss = 0.5 * (0.5 * diff.pow(2) + torch.log(sigma))
        elif  reduce =='mean':
            loss = 0.5 * torch.mean(0.5 * diff.pow(2) + torch.log(sigma))
        else:
            raise NotImplementedError
        return loss


class RAPAgent(BisimAgent):
    """
    RAP agent
    https://github.com/jianda-chen/RAP_distance
    """
    def __init__(self, obs_shape, action_shape, actor_action_max, cfg, L):
        super().__init__(obs_shape, action_shape, actor_action_max, cfg, L)
        self.state_reward_decoder = StateRewardDecoder(
            cfg.encoder_feature_dim).to(self.device)
        
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters())
            + list(self.state_reward_decoder.parameters()),
            lr=cfg.decoder_lr,
            weight_decay=cfg.decoder_weight_lambda
        )

    def update_encoder(self, obs, next_obs, action, reward, step):
        """
        Update the metrics (encoders) using sampled rewards and transition models.
        """
        z = self.critic.encoder(obs)
        if self.cfg.use_target_encoder:
            z_target = self.critic_target.encoder(obs).detach()
        else:
            z_target = z

        # Get next latent states
        with torch.no_grad():
            if self.transition_model_type in ['', 'deterministic', 'ensemble_det']:
                pred_z_prime_sigma = None
                pred_z_prime_mu = self.transition_model.sample_prediction(torch.cat([z_target, action], dim=1))
            else:
                pred_z_prime_mu, pred_z_prime_sigma = self.transition_model(torch.cat([z_target, action], dim=1))

        z, z2, reward, reward2, pred_z_prime_mu, pred_z_prime_mu2, pred_z_prime_sigma, pred_z_prime_sigma2 = self.generate_bisim_samples(
            z, z_target, reward, pred_z_prime_mu, pred_z_prime_sigma)

        # Predict state-dependent reward
        if self.cfg.state_rp_detach_encoder:
            reward_mu, reward_sigma = self.state_reward_decoder(z.detach())
        else: reward_mu, reward_sigma = self.state_reward_decoder(z)
        state_reward_decoder_loss = self.state_reward_decoder.loss(
            reward_mu, reward_sigma, reward)
        r_var = reward_sigma.detach().pow(2.)
        if self.cfg.bisim_coef <= 0: return 0, state_reward_decoder_loss

        z, z2, reward, reward2, pred_z_prime_mu, pred_z_prime_mu2, \
        pred_z_prime_sigma, pred_z_prime_sigma2, r_var, r_var2 = self.generate_bisim_samples(
                z, z_target, reward, pred_z_prime_mu, pred_z_prime_sigma, r_var=r_var)

        # Get z_dist, r_dist, and z'_dist
        z_dist = self.metric_func(z, z2)
        r_dist = self.reward_dist(reward, reward2, (r_var, r_var2))
        if 'det' in self.transition_model_type or self.transition_model_type == '':
            transition_dist = self.metric_func(pred_z_prime_mu, pred_z_prime_mu2)
        else:
            transition_dist = self.prob_metric_func(
                pred_z_prime_mu, pred_z_prime_mu2, pred_z_prime_sigma, pred_z_prime_sigma2)

        # Compute metric loss
        diff_square = (z_dist - self.cfg.c_T * transition_dist).pow(2.)
        loss = F.smooth_l1_loss(diff_square, self.cfg.c_R * r_dist, reduction='mean')

        # Stats
        if self.cfg.r_dist == 'rap':
            bisimilarity = self.cfg.c_R * torch.sqrt(torch.relu(r_dist)) + self.cfg.c_T * transition_dist
        else: bisimilarity = self.cfg.c_R * r_dist + self.cfg.c_T * transition_dist
        r_dist_l1 = torch.abs(reward - reward2)
        self.log_metric_stats(bisimilarity, r_dist, z_dist, transition_dist, loss, step, r_dist_l1)
        return loss, state_reward_decoder_loss

    def update(self, batch_transition, step, **kwargs):
        obs, action, _, reward, next_obs, not_done = batch_transition
        if 'on_policy_transition' in kwargs and kwargs['on_policy_transition'] is not None:
            obs_op, action_op, reward_op, next_obs_op = kwargs['on_policy_transition']
            obs_op = torch.as_tensor(obs_op, device=self.device).float()
            action_op = torch.as_tensor(action_op, device=self.device).float()
            reward_op = torch.as_tensor(reward_op, device=self.device).float()
            next_obs_op = torch.as_tensor(next_obs_op, device=self.device).float()

        self.L.log('train/batch_reward', reward.mean(), step)

        bisim_coef = self.cfg.bisim_coef if step > self.cfg.bisim_start_step else 0

        self.update_critic(obs, action, reward, next_obs, not_done, step)
        transition_reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, step) if self.cfg.rzp_coef > 0 else 0
        if 'on_policy_transition' in kwargs and kwargs['on_policy_transition'] is not None:
            encoder_loss, state_reward_decoder_loss = self.update_encoder(obs_op, next_obs_op, action_op, reward_op, step)
        else:
            encoder_loss, state_reward_decoder_loss = self.update_encoder(obs, next_obs, action, reward, step)
        total_loss = bisim_coef * encoder_loss + self.cfg.rzp_coef * transition_reward_loss \
            + self.cfg.state_reward_decoder_loss_coef * state_reward_decoder_loss

        if self.cfg.report_grad_norm:
            encoder_parameters = [p for p in self.critic.encoder.parameters() if p.requires_grad]
            self.grad_norm_analysis(bisim_coef * encoder_loss, encoder_parameters, step, label='metric_loss')
            self.grad_norm_analysis(self.cfg.rzp_coef * transition_reward_loss, encoder_parameters, step, label='transition_reward_loss')
            self.grad_norm_analysis(self.cfg.state_reward_decoder_loss_coef * state_reward_decoder_loss, encoder_parameters, step, label='state_reward_decoder_loss')

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)

        self.soft_update_params(step)