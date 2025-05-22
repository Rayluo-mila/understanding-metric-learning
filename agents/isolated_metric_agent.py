import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

import utils.rl_utils as rl_utils
from agents.bisim_agent_sac import BisimAgent
from agents.base_agent import get_max_norm
from agents.encoder import make_encoder
from agents.transition_model import make_transition_model


class IsolatedMetricSACAgent(BisimAgent):
    """SAC agent with a learned metric, but not using the metric in RL."""
    def __init__(self, obs_shape, action_shape, actor_action_max, cfg, L):
        super().__init__(obs_shape, action_shape, actor_action_max, cfg, L)
        self.need_zp = 'mico' not in cfg.bisim_dist and 'simsr' not in cfg.bisim_dist
        self.need_rp = 'mico' not in cfg.bisim_dist and 'simsr' not in cfg.bisim_dist

        encoder_max_norm, encoder_max_norm_ord = get_max_norm(self.metric_ub, cfg, metric_encoder=True)
        self.encoder_metric = make_encoder(
            cfg.encoder_type_metric, obs_shape, cfg.metric_encoder_feature_dim, cfg.num_layers,
            cfg.num_filters, cfg.encoder_stride,
            max_norm=encoder_max_norm, max_norm_ord=encoder_max_norm_ord
        ).to(cfg.device)
        self.encoder_metric_target = make_encoder(
            cfg.encoder_type_metric, obs_shape, cfg.metric_encoder_feature_dim, cfg.num_layers,
            cfg.num_filters, cfg.encoder_stride,
            max_norm=encoder_max_norm, max_norm_ord=encoder_max_norm_ord
        ).to(cfg.device)
        self.encoder_metric_target.load_state_dict(self.encoder_metric.state_dict())

        # Transition model and reward model
        self.transition_model = make_transition_model(
            cfg.transition_model_type, cfg.metric_encoder_feature_dim, action_shape,
            layer_width=cfg.transition_model_hidden_size,
            encoder_max_norm=self.encoder_max_norm, encoder_max_norm_ord=self.encoder_max_norm_ord
        ).to(cfg.device)
        if self.cfg.rp_zprime:
            self.reward_decoder = nn.Sequential(
            nn.Linear(cfg.metric_encoder_feature_dim, cfg.reward_decoder_hidden_size),
            nn.LayerNorm(cfg.reward_decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.reward_decoder_hidden_size, 1)).to(cfg.device)
        else:
            self.reward_decoder = nn.Sequential(
                nn.Linear(cfg.metric_encoder_feature_dim + action_shape[0], cfg.reward_decoder_hidden_size),
                nn.LayerNorm(cfg.reward_decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(cfg.reward_decoder_hidden_size, 1)).to(cfg.device)

        self.metric_optimizer = torch.optim.Adam(
            self.encoder_metric.parameters(),
            lr=cfg.metric_encoder_lr
        )
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=cfg.decoder_lr,
            weight_decay=cfg.decoder_weight_lambda
        )

        self.train()

    def print_model_stats(self):
        super().print_model_stats()
        print(f'Metric Encoder params count: {rl_utils.count_parameters(self.encoder_metric)}')

    def update_encoder(self, obs, next_obs, action, reward, step):
        """
        Update the metrics (encoders) using sampled rewards and transition models.
        """
        z = self.encoder_metric(obs)
        if self.cfg.use_target_encoder:
            z_target = self.encoder_metric_target(obs).detach()
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
        Implementing (detachable) RP+ZP losses.
        """
        transition_loss = 0.
        if self.need_zp:
            z_zp = self.encoder_metric(obs)
            if self.cfg.metric_zp_detach_encoder: z_zp = z_zp.detach()
            pred_z_prime_mu, pred_z_prime_sigma = self.transition_model(torch.cat([z_zp, action], dim=1))
            if pred_z_prime_sigma is None:
                pred_z_prime_sigma = torch.ones_like(pred_z_prime_mu)

            z_prime = self.encoder_metric(next_obs).detach()
            diff = (pred_z_prime_mu - z_prime) / pred_z_prime_sigma
            transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_z_prime_sigma))
            self.log('train_ae/transition_loss', transition_loss.item(), step)

        reward_loss = 0.
        if self.need_rp:
            z_rp = self.encoder_metric(obs)
            if self.cfg.metric_rp_detach_encoder: z_rp = z_rp.detach()
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

        # Base agent
        self.update_critic(obs, action, reward, next_obs, not_done, step)
        transition_reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, step) if self.cfg.rzp_coef > 0 else 0
        transition_reward_loss = self.cfg.rzp_coef * transition_reward_loss
        
        # Metric update
        if self.cfg.bisim_coef <= 0:
            encoder_loss = 0
        elif 'on_policy_transition' in kwargs and kwargs['on_policy_transition'] is not None:
            encoder_loss = self.update_encoder(obs_op, next_obs_op, action_op, reward_op, step)
        else:
            encoder_loss = self.update_encoder(obs, next_obs, action, reward, step)
        encoder_loss = bisim_coef * encoder_loss

        if self.cfg.report_grad_norm:
            metric_parameters = [p for p in self.encoder_metric.parameters() if p.requires_grad]
            self.grad_norm_analysis(encoder_loss, metric_parameters, step, label='metric_loss')
            encoder_parameters = [p for p in self.encoder_metric.parameters() if p.requires_grad]
            self.grad_norm_analysis(transition_reward_loss, encoder_parameters, step, label='transition_reward_loss')

        self.decoder_optimizer.zero_grad()
        self.metric_optimizer.zero_grad()
        if transition_reward_loss != 0: transition_reward_loss.backward()
        if encoder_loss != 0: encoder_loss.backward()
        self.decoder_optimizer.step()
        self.metric_optimizer.step()

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)

        self.soft_update_params(step)

    def soft_update_params(self, step):
        super().soft_update_params(step)
        if step % self.critic_target_update_freq == 0:
            rl_utils.soft_update_params(
                self.encoder_metric, self.encoder_metric_target,
                self.encoder_tau
            )

    def save(self, model_dir, step):
        super().save(model_dir, step)
        torch.save(
            self.encoder_metric.state_dict(), '%s/encoder_metric_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.encoder_metric_target.state_dict(), '%s/encoder_metric_target_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        super().load(model_dir, step)
        self.encoder_metric.load_state_dict(
            torch.load('%s/encoder_metric_%s.pt' % (model_dir, step))
        )
        self.encoder_metric.load_state_dict(
            torch.load('%s/encoder_metric_target_%s.pt' % (model_dir, step))
        )