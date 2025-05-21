import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

import utils.rl_utils as rl_utils
from agents.models import  Actor, Critic
from agents.distance_functions import metric_func, prob_metric_func, reward_dist


def get_max_norm(metric_ub, cfg, metric_encoder=False):
    encoder_max_norm = None
    encoder_max_norm_ord = None
    if (cfg.encoder_max_norm and not metric_encoder) or \
            (cfg.encoder_max_norm_metric and metric_encoder):
        encoder_max_norm = 0.5 * metric_ub

        # Override the max_norm if specified
        if (cfg.encoder_max_norm_override > 0 and not metric_encoder) or \
            (cfg.encoder_max_norm_override_metric > 0 and metric_encoder):
            encoder_max_norm = cfg.encoder_max_norm_override
        
        if 'L1' in cfg.bisim_dist: encoder_max_norm_ord = 1
        elif 'huber' in cfg.bisim_dist: encoder_max_norm_ord = 1
        elif 'L2' in cfg.bisim_dist: encoder_max_norm_ord = 2
        else: encoder_max_norm_ord = 2
    return encoder_max_norm, encoder_max_norm_ord


class SACAgent(object):
    def __init__(self, obs_shape, action_shape, actor_action_max, cfg, L):
        self.cfg = cfg
        self.L = L
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.discount = cfg.discount
        self.critic_tau = cfg.critic_tau
        self.encoder_tau = cfg.encoder_tau
        self.actor_update_freq = cfg.actor_update_freq
        self.critic_target_update_freq = cfg.critic_target_update_freq
        self.decoder_update_freq = cfg.decoder_update_freq
        self.decoder_type = cfg.decoder_type
        self.sigma = 0.5

        # Metric related
        self.max_reward = cfg.max_reward
        self.min_reward = cfg.min_reward
        self.metric_ub = cfg.c_R * (self.max_reward - self.min_reward) / (1 - cfg.c_T)
        self.encoder_max_norm, self.encoder_max_norm_ord = get_max_norm(self.metric_ub, cfg)

        # For tracking
        self.max_r_dist_l1 = 0
        self.max_z_dist = 0
        self.max_zprime_dist = 0
        self.max_bd = 0
        self.collected_r_dist = []
        self.current_step = 0

        self.actor = Actor(
            obs_shape, action_shape, cfg.hidden_dim, cfg.encoder_type,
            cfg.encoder_feature_dim, actor_action_max, cfg.actor_log_std_min, cfg.actor_log_std_max,
            cfg.num_layers, cfg.num_filters, cfg.encoder_stride,
            encoder_max_norm=self.encoder_max_norm, encoder_max_norm_ord=self.encoder_max_norm_ord
        ).to(cfg.device)

        self.critic = Critic(
            obs_shape, action_shape, cfg.hidden_dim, cfg.encoder_type,
            cfg.encoder_feature_dim, cfg.num_layers, cfg.num_filters, cfg.encoder_stride,
            encoder_max_norm=self.encoder_max_norm, encoder_max_norm_ord=self.encoder_max_norm_ord
        ).to(cfg.device)

        self.critic_target = Critic(
            obs_shape, action_shape, cfg.hidden_dim, cfg.encoder_type,
            cfg.encoder_feature_dim, cfg.num_layers, cfg.num_filters, cfg.encoder_stride,
            encoder_max_norm=self.encoder_max_norm, encoder_max_norm_ord=self.encoder_max_norm_ord
        ).to(cfg.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(cfg.init_temperature)).to(cfg.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr, betas=(cfg.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr, betas=(cfg.critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=cfg.alpha_lr, betas=(cfg.alpha_beta, 0.999)
        )

        self.decoder = None
        self.reward_decoder = None
        self.transition_model = None
        self.encoder_metric = None

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)
        if self.decoder is not None: self.decoder.train(training)
        if self.reward_decoder is not None: self.reward_decoder.train(training)
        if self.transition_model is not None:
            if 'ensemble' in self.cfg.transition_model_type:
                for i in range(len(self.transition_model.models)):
                    self.transition_model.models[i].train(training)
            else:
                self.transition_model.train(training)
        if self.encoder_metric is not None:
            self.encoder_metric.train(training)
            self.encoder_metric_target.train(training)

    def print_model_stats(self):
        print(f'Actor params count: {rl_utils.count_parameters(self.actor)}')
        print(f'Critic params count: {rl_utils.count_parameters(self.critic)}')
        print(f'Encoder params count: {rl_utils.count_parameters(self.critic.encoder)}')

    def log(self, label, value, step):
        self.L.log(label, value, step)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, multiproc=False, **kwargs):  # TODO: torch_mode
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0) if not multiproc else obs  # 4-dim: vectorized
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            if not multiproc:
                return mu.cpu().data.numpy().flatten()
            else:
                return mu.cpu().data.numpy()

    def sample_action(self, obs, multiproc=False, **kwargs):
        with torch.no_grad():
            if multiproc:
                obs = torch.FloatTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            if not multiproc:
                return pi.cpu().data.numpy().flatten()
            else:
                return pi.cpu().data.numpy()

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            if self.cfg.use_done_signal:
                target_Q = reward + (not_done * self.discount * target_V)
            else:
                target_Q = reward + (self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2, h_norm = self.critic(obs, action,
                                                     detach_encoder=self.cfg.no_phi_q_star,
                                                     require_hnorm=True)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        self.L.log('train_critic/loss', critic_loss, step)
        self.L.log('train_critic/batch_q1', current_Q1.mean().item(), step)
        self.L.log('train_critic/batch_q2', current_Q2.mean().item(), step)
        for k, v in h_norm.items():
            self.L.log(f'train_metric/{k}', v, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.cfg.report_grad_norm:
            encoder_parameters = [p for p in self.critic.encoder.parameters() if p.requires_grad]
            critic_parameters = [p for p in self.critic.parameters() if p.requires_grad]
            self.grad_norm_analysis(critic_loss, encoder_parameters, step, label='critic_encoder_loss', post_backward=True)
            self.grad_norm_analysis(critic_loss, critic_parameters, step, label='critic_all_loss', post_backward=True)
        if self.cfg.clip_grad_norm_critic > 0:
            nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), self.cfg.clip_grad_norm_critic)
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        self.L.log('train_actor/loss', actor_loss, step)
        self.L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        self.L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.cfg.report_grad_norm:
            actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
            self.grad_norm_analysis(actor_loss, actor_parameters, step, label='actor_loss', post_backward=True)
        if self.cfg.clip_grad_norm_actor > 0:
            nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), self.cfg.clip_grad_norm_actor)
        self.actor_optimizer.step()

        self.actor.log(self.L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.L.log('train_alpha/loss', alpha_loss, step)
        self.L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, batch_transition, step, **kwargs):
        obs, action, _, reward, next_obs, not_done = batch_transition

        self.L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)
        
        self.soft_update_params(step)

    def soft_update_params(self, step):
        if step % self.critic_target_update_freq == 0:
            rl_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            rl_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            rl_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

    def grad_norm_analysis(self, loss, parameters, step, label='', post_backward=False):
        if step % 100 != 0: return
        if not post_backward:
            if (isinstance(loss, float) or isinstance(loss, int)) and loss == 0:  # loss is zero
                return
            grads = torch.autograd.grad(
                outputs=loss, inputs=parameters, create_graph=False, retain_graph=True, allow_unused=True
            )
            grads_norms = [torch.linalg.norm(g.detach().data).item() ** 2 for g in grads if g is not None]
        else:
            grads_norms = [torch.linalg.norm(p.grad.detach().data).item() ** 2 for p in parameters if p.grad is not None]
        grads_norm_sum = sum(grads_norms)
        encoder_metric_norm = grads_norm_sum ** 0.5
        self.log(f'train_metric/{label}_grad_norm', encoder_metric_norm, step)

    def reward_dist(self, reward, reward2, r_vars=None):
        r_dist = reward_dist(reward, reward2, self.cfg, r_vars)  # (batch_size)
        return r_dist

    def collect_and_save_r_dist_plot(self, step, num_bins=20):
        """Combine collected r_dist and save distribution plot."""
        file_dir = Path(self.cfg.plot_r_dist_save_dir)
        file_dir.mkdir(parents=True, exist_ok=True)
        # Combine all batches into one tensor
        collected_r_dist = torch.cat(self.collected_r_dist).cpu().numpy()
        y_max = len(collected_r_dist)
        collected_r_dist = collected_r_dist[collected_r_dist != 0]

        # Create histogram bins
        hist, bin_edges = np.histogram(collected_r_dist, bins=num_bins, range=(self.min_reward, self.max_reward))

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(bin_edges[:-1], hist, width=(self.max_reward - self.min_reward) / num_bins, edgecolor='k', align='edge')
        plt.title(f"Distribution of r_dist, step={step}")
        plt.xlabel("r_dist values")
        plt.ylabel("Frequency")
        plt.xticks(bin_edges, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.xlim(self.min_reward, self.max_reward)
        plt.ylim(0, y_max)

        # Save the plot with filename based on the total number of samples
        filename = file_dir / f"st{step}_r_dist_distribution.png"
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        plt.close()

    def metric_func(self, x, y, step=None):
        z_dist = metric_func(x, y, self.cfg, self.metric_ub, step, self.L)  # (batch_size)
        return z_dist

    def prob_metric_func(self, mu, mu2, sigma, sigma2):
        zprime_dist = prob_metric_func(mu, mu2, sigma, sigma2, self.cfg, metric_ub=self.metric_ub)  # (batch_size)
        return zprime_dist

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )
        if self.reward_decoder is not None:
            torch.save(
                self.reward_decoder.state_dict(),
                '%s/reward_decoder_%s.pt' % (model_dir, step)
            )
        if self.transition_model is not None:
            torch.save(
                self.transition_model.state_dict(),
                '%s/transition_model_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_target_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
        if self.reward_decoder is not None:
            self.reward_decoder.load_state_dict(
                torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
            )
        if self.transition_model is not None:
            self.transition_model.load_state_dict(
                torch.load('%s/transition_model_%s.pt' % (model_dir, step))
            )