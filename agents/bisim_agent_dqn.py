import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.rl_utils as rl_utils
from memory_profiler import profile
from agents.models import QNetwork, Clamp
from agents.transition_model import make_transition_model


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def _sqrt(x, tol=0.):
    tol = torch.zeros_like(x)
    return torch.sqrt(torch.maximum(x, tol))


def cosine_distance(x, y):
    EPSILON = 1e-9
    numerator = torch.sum(x * y, dim=-1)
    # print("numerator", numerator.shape, numerator)
    denominator = torch.sqrt(
        torch.sum(x.pow(2.), dim=-1)) * torch.sqrt(torch.sum(y.pow(2.), dim=-1))
    cos_similarity = numerator / (denominator + EPSILON)

    return torch.atan2(_sqrt(1. - cos_similarity.pow(2.)), cos_similarity)


class BisimAgentDQN(object):
    """Bisimulation metric algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        transition_model_type,
        batch_size=64,
        env=None,
        hidden_dim=256,
        discount=0.99,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        q_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        encoder_max_norm=None,
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        bisim_coef=0.5,
        bisim_dist='huber',
        c_R=1.,
        c_T=0.99,
        max_reward=1,
        min_reward=0,
        reward_decoder_hidden_size=64,
        reward_decoder_type='dqn',
        use_target_encoder=False,
        phi_q_star=True,
        repr_upd_phase_length=1,
        q_upd_phase_length=1,
        on_policy_metric_update=False,
        metric_type='pbsm',
        rp_zp=True,
        cheat_reward_transition=False,
        fixed_policy_after_n_steps=0,
        r_dist='L1',
        metric_loss_type='L2',
        use_uniformly_sampled_states=False,
        encoder_norm_scale=256,
    ):
        self.device = device
        self.debug = False
        self.env = env
        self.batch_size = batch_size
        self.discount = discount
        self.action_dim = action_shape[0]
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.q_target_update_freq = q_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.reward_decoder_type = reward_decoder_type
        self.obs_shape = obs_shape
        self.max_reward = max_reward
        self.min_reward = min_reward

        # cheating
        if cheat_reward_transition:
            self.P = torch.as_tensor(self.env.get_P(), device=self.device)  # [ns, na, ns]
            self.R = torch.as_tensor(self.env.get_R(), device=self.device)  # [ns, na]

        # metric related
        self.transition_model_type = transition_model_type
        self.bisim_coef = bisim_coef
        self.bisim_dist = bisim_dist
        self.encoder_feature_dim = encoder_feature_dim
        self.use_target_encoder = use_target_encoder
        self.phi_q_star = phi_q_star
        self.repr_upd_phase_length = repr_upd_phase_length
        self.q_upd_phase_length = q_upd_phase_length
        self.on_policy_metric_update = on_policy_metric_update
        self.metric_type = metric_type
        self.cheat_reward_transition = cheat_reward_transition
        self.fixed_policy_after_n_steps = fixed_policy_after_n_steps
        self.use_uniformly_sampled_states = use_uniformly_sampled_states
        self.is_policy_fixed = False
        self.rp_zp = rp_zp
        self.c_R = c_R
        self.c_T = c_T
        self.r_dist = r_dist
        self.metric_loss_type = metric_loss_type
        self.encoder_norm_scale = encoder_norm_scale
        if encoder_max_norm:
            encoder_max_norm = 0.5 * c_R * \
                (max_reward - min_reward) / (1 - c_T)
            if bisim_dist == 'huber':
                encoder_max_norm += 0.5  # beta in huber loss
            if bisim_dist == 'L2':
                encoder_max_norm *= math.sqrt(encoder_feature_dim)
        
            # Prevent the representation to be numerically too small
            encoder_max_norm *= encoder_norm_scale

        # stats
        self.mu_rd = 0
        self.mu_bd = 0
        self.mu_nzd = 0
        self.mu_zd = 0

        ###
        self.q_net = QNetwork(
            obs_shape, self.action_dim, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, encoder_max_norm=encoder_max_norm
        ).to(device)
        self.target_net = QNetwork(
            obs_shape, self.action_dim, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, encoder_max_norm=encoder_max_norm
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # a fixed Q network
        self.q_net_fixed = QNetwork(
            obs_shape, self.action_dim, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, encoder_max_norm=encoder_max_norm
        ).to(device)
        for param in self.q_net_fixed.parameters():
            param.requires_grad = False
        self.q_net_fixed.eval()

        # metric net
        if bisim_dist == 'metric_net':
            self.metric_decoder = nn.Sequential(
                nn.Linear(encoder_feature_dim * 2,
                          reward_decoder_hidden_size),
                nn.LayerNorm(reward_decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(reward_decoder_hidden_size, 1)).to(device)
            
            self.metric_decoder_target = nn.Sequential(
                nn.Linear(encoder_feature_dim * 2,
                          reward_decoder_hidden_size),
                nn.LayerNorm(reward_decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(reward_decoder_hidden_size, 1)).to(device)
            self.metric_decoder_target.load_state_dict(self.metric_decoder.state_dict())

            self.metric_decoder_optimizer = torch.optim.Adam(
                self.metric_decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )
            self.metric_decoder_target.train()

        zp_rp_input_dim = encoder_feature_dim
        self.transition_model = make_transition_model(
            transition_model_type, zp_rp_input_dim, action_shape, encoder_max_norm=encoder_max_norm
        ).to(device)

        if reward_decoder_type == 'dqn':
            self.reward_decoder = nn.Sequential(
                nn.Linear(zp_rp_input_dim, reward_decoder_hidden_size),
                nn.LayerNorm(reward_decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(reward_decoder_hidden_size, self.action_dim),
                Clamp(min_value=self.min_reward, max_value=self.max_reward)).to(device)
        else:
            self.reward_decoder = nn.Sequential(
                nn.Linear(zp_rp_input_dim + self.action_dim,
                          reward_decoder_hidden_size),
                nn.LayerNorm(reward_decoder_hidden_size),
                nn.ReLU(),
                nn.Linear(reward_decoder_hidden_size, 1),
                Clamp(min_value=self.min_reward, max_value=self.max_reward)).to(device)

        # optimizers
        if self.phi_q_star:
            self.q_optimizer = torch.optim.Adam(
                self.q_net.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
            )
        else:
            self.q_optimizer = torch.optim.Adam(
                self.q_net.q.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
            )

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) +
            list(self.transition_model.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for q_net encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.q_net.encoder.parameters(), lr=encoder_lr
        )

        self.train()
        self.target_net.train()

    def train(self, training=True):
        self.training = training
        self.q_net.train(training)

    def get_q_nets(self):
        if self.is_policy_fixed:
            return self.q_net_fixed
        else:
            return self.q_net

    def select_action(self, obs, torch_mode=False, output_value=False):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs, device=self.device, dtype=torch.float32)
            if not torch_mode:
                obs_tensor = obs_tensor.unsqueeze(0)
            q_values = self.get_q_nets()(obs_tensor)
            if output_value:
                if torch_mode:
                    q_val, action_idx = torch.max(q_values, dim=-1)
                else:
                    q_val, action_idx = torch.max(q_values, dim=-1).cpu()
                    q_val = q_val.numpy()
                    action_idx = action_idx.numpy()
                return action_idx, q_val
            else:
                action_idx = torch.argmax(q_values, dim=-1)
                action_idx = action_idx if torch_mode else action_idx[0].cpu().numpy()
                return action_idx

    def update_q_nets(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            target_max, _ = self.target_net(next_obs).max(dim=1)
            td_target = reward.flatten() + self.discount * target_max * not_done.flatten()

        # Get current Q estimates
        old_val = self.q_net(obs, detach_encoder=not self.phi_q_star).gather(
            1, action).squeeze()
        q_loss = F.mse_loss(td_target, old_val)

        # Optimize the Q net
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        L.log('train_q/loss', q_loss, step)
        _not_done = not_done.squeeze()
        Q_vals_at_termination = old_val[_not_done == 0].mean()
        Q_vals_elsewhere = old_val[_not_done == 1].mean()
        if not torch.isnan(Q_vals_at_termination):
            L.log('train_q/max_q_value_termination', Q_vals_at_termination.item(), step)
        if not torch.isnan(Q_vals_elsewhere):
            L.log('train_q/max_q_value_elsewhere', Q_vals_elsewhere.item(), step)
        self.q_net.log(L, step)

    def metric_func(self, x, y, target=False):
        if self.bisim_dist == 'L1':
            dist = torch.linalg.norm(x-y, ord=1, dim=-1)
            dist = dist / self.encoder_norm_scale
        elif self.bisim_dist == 'L2':
            dist = torch.linalg.norm(x-y, ord=2, dim=-1)
            dist = dist / self.encoder_norm_scale
        elif self.bisim_dist == 'huber':
            dist = F.smooth_l1_loss(x, y, reduction='none')
            dist = dist.sum(dim=-1)
            dist = dist / self.encoder_norm_scale
        elif self.bisim_dist == 'metric_net':
            if not target:
                dist = self.metric_decoder(torch.cat((x, y), dim=-1))
            else:
                dist = self.metric_decoder_target(torch.cat((x, y), dim=-1))
        elif self.bisim_dist == 'mico':
            beta = 1e-6  # try 1e-5 or 0.1
            base_distances = cosine_distance(x, y)
            # print("base_distances", base_distances)
            norm_average = x.pow(2.).sum(dim=-1) + y.pow(2.).sum(dim=-1)
            dist = norm_average + beta * base_distances
        elif self.bisim_dist == 'x^2+y^2-xy':  # RAP feature
            # beta = 1.0  # 0 < beta < 2
            k = 0.1  # 0 < k < 2
            base_distances = (x * y).sum(dim=-1)
            # print("base_distances", base_distances)
            norm_average = (x.pow(2.).sum(dim=-1)
                            + y.pow(2.).sum(dim=-1))
            # dist = norm_average - (2. - beta) * base_distances
            dist = norm_average - k * base_distances
            # dist = dist.sqrt()
        else:
            raise NotImplementedError
        return dist

    def reward_dist(self, reward, reward2):
        if self.r_dist == 'L1':
            r_dist = (reward - reward2).abs()  # shape: (batch_size, action_dim)
        elif self.r_dist == 'L2':
            r_dist = (reward - reward2).pow(2)
        elif self.r_dist == 'huber':
            r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
        return r_dist

    def get_reward_next_state_from_env(self, obs, action=None):
        """
        Only for deterministic Garnet env!
        """
        with torch.no_grad():
            obs_idx = torch.argmax(obs, dim=1)
            if action is None:
                next_states = self.P[obs_idx].float()  # shape: (batch_size, na, ns)
                rewards = self.R[obs_idx]  # shape: (batch_size, na)
            else:  # get all actions' next states and rewards
                # shape: (batch_size, ns), i.e., P[obs_idx[i], action[i], :]
                next_states = self.P[obs_idx, action].float()
                rewards = self.R[obs_idx, action]  # shape: (batch_size)
        return rewards, next_states

    def update_encoder(self, obs, action, reward, L, step):
        h = self.q_net.encoder(obs)

        if self.on_policy_metric_update:
            action, q_vals = self.select_action(
                obs, torch_mode=True, output_value=True)
        else:
            _, q_vals = self.select_action(
                obs, torch_mode=True, output_value=True)

        # Sample random states across episodes at random
        perm = torch.randperm(self.batch_size, device=self.device)
        h2 = h[perm]
        q_vals2 = q_vals[perm]
        obs2 = obs[perm]

        with torch.no_grad():
            action_onehot = F.one_hot(action.long(), self.action_dim).float().squeeze()  # discrete
            if self.on_policy_metric_update and self.cheat_reward_transition:
                reward, pred_next_latent_mu1_ = self.get_reward_next_state_from_env(obs, action)
                with torch.no_grad():
                    if self.use_target_encoder:
                        pred_next_latent_mu1 = self.target_net.encoder(pred_next_latent_mu1_)
                    else:
                        pred_next_latent_mu1 = self.q_net.encoder(pred_next_latent_mu1_)
                pred_next_latent_sigma1 = None
            else:
                if self.on_policy_metric_update:
                    if self.reward_decoder_type == 'dqn':
                        reward = self.reward_decoder(h).gather(1, action.unsqueeze(dim=1))
                    elif self.reward_decoder_type == 'transition':
                        h_act = torch.cat([h, action_onehot], dim=1)
                        reward = self.reward_decoder(h_act)
                if self.use_target_encoder:
                    h_target = self.target_net.encoder(obs)
                    pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(
                        torch.cat([h_target, action_onehot], dim=1))
                else:
                    pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(
                        torch.cat([h, action_onehot], dim=1))
                # reward = self.reward_decoder(pred_next_latent_mu1)
            reward2 = reward[perm]

        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError

        z_dist = self.metric_func(h, h2)
        r_dist = self.reward_dist(reward, reward2)
        L.log('train/norm', h.norm(p=1, dim=-1).mean(), step)

        if self.transition_model_type in ['', 'deterministic']:
            with torch.no_grad():
                transition_dist = self.metric_func(pred_next_latent_mu1, pred_next_latent_mu2, target=True)
        else:
            raise NotImplementedError

        r_dist = r_dist.squeeze()
        bisimilarity = self.c_R * r_dist + self.c_T * transition_dist

        # Used only for tracking purposes
        with torch.no_grad():
            self.mu_bd = 0.95 * self.mu_bd + 0.05 * bisimilarity.mean().item()
            self.mu_rd = 0.95 * self.mu_rd + 0.05 * r_dist.mean().item()
            self.mu_nzd = 0.95 * self.mu_nzd + 0.05 * transition_dist.mean().item()
            self.mu_zd = 0.95 * self.mu_zd + 0.05 * z_dist.mean().item()

            L.log('train/mu_rd', self.mu_rd, step)
            L.log('train/mu_bd', self.mu_bd, step)
            L.log('train/mu_nzd', self.mu_nzd, step)
            L.log('train/mu_zd', self.mu_zd, step)
            L.log('train/mu_rd_div_bd', self.mu_rd / self.mu_bd, step)

            if self.debug:
                non_zero_indices = torch.nonzero(z_dist.detach(), as_tuple=True)[0]
                q_val_diff = torch.abs(q_vals - q_vals2)
                v_d_lips_vec = torch.div(q_val_diff[non_zero_indices], z_dist[non_zero_indices])
                v_d_lipschitz_cnt = (v_d_lips_vec > (1.0 / self.c_R)).float().mean().item()
                v_d_lipschitz = v_d_lips_vec.max().item()
                del q_val_diff
                del v_d_lips_vec
                td_d_lips_vec = torch.div(transition_dist[non_zero_indices], z_dist[non_zero_indices])
                td_d_lipschitz = td_d_lips_vec.max().item()
                td_d_rev_lips = td_d_lips_vec.min().item()
                del td_d_lips_vec
                raw_obs_dist = (obs - obs2).norm(p=1, dim=-1)
                d_raw_lipschitz = torch.div(z_dist[non_zero_indices], raw_obs_dist[non_zero_indices]).max().item()
                del raw_obs_dist
                del non_zero_indices

                L.log('train/v_d_lipschitz', v_d_lipschitz, step)
                L.log('train/v_d_lipschitz_cnt', v_d_lipschitz_cnt, step)
                L.log('train/td_d_lipschitz', td_d_lipschitz, step)
                L.log('train/td_d_rev_lips', td_d_rev_lips, step)
                L.log('train/d_raw_lipschitz', d_raw_lipschitz, step)
        
        # loss = ((z_dist - bisimilarity).abs() - 0.1).clip(min=0).mean()
        if self.metric_loss_type == 'huber':
            loss = F.smooth_l1_loss(z_dist, bisimilarity, reduction='mean')
        else:
            loss = (z_dist - bisimilarity).pow(2).mean()
        L.log('train/metric_loss', loss, step)
        return loss

    def get_all_action_next_latent(self, h):
        # pred_next_latent_mu1 = torch.zeros(self.batch_size, self.action_dim, self.encoder_feature_dim)
            # for i in range(self.action_dim):
            #     action_onehot = F.one_hot(i.long(), self.action_dim).float().squeeze()  # discrete
            #     mu1, _ = self.transition_model(
            #         torch.cat([h, action_onehot], dim=1))
            #     pred_next_latent_mu1[:, i, :] = mu1

        # Create a batch of one-hot encoded actions
        actions_onehot = torch.eye(self.action_dim, device=self.device).repeat(
            self.batch_size, 1, 1)  # shape: (batch_size, action_dim, action_dim)

        # Repeat the hidden states (h) to match the actions_onehot dimensions
        # shape: (batch_size, action_dim, hidden_dim)
        h_repeated = h.unsqueeze(1).repeat(1, self.action_dim, 1)

        # Concatenate h and actions_onehot along the last dimension
        # shape: (batch_size, action_dim, hidden_dim + action_dim)
        h_actions_combined = torch.cat(
            (h_repeated, actions_onehot), dim=2)

        # Reshape for the transition model input
        # shape: (batch_size * action_dim, hidden_dim + action_dim)
        h_actions_combined = h_actions_combined.view(
            -1, h_actions_combined.size(2))

        # Pass through the transition model
        mu1, _ = self.transition_model(h_actions_combined)

        # Reshape the output to the desired shape
        # shape: (batch_size, action_dim, encoder_feature_dim)
        pred_next_latent_mu1 = mu1.view(
            self.batch_size, self.action_dim, self.encoder_feature_dim)
        
        return pred_next_latent_mu1

    def generate_random_state_pairs(self):
        # Randomly select indices for the one-hot encoded vectors
        state_dim = self.obs_shape[0]
        indices1 = torch.randint(0, state_dim, (self.batch_size,), device=self.device)
        indices2 = torch.randint(0, state_dim, (self.batch_size,), device=self.device)

        # Create one-hot encoded vectors using F.one_hot
        states1 = F.one_hot(indices1, num_classes=state_dim).float()
        states2 = F.one_hot(indices2, num_classes=state_dim).float()

        return states1, states2

    def update_encoder_bism(self, obs, L, step, obs2=None):
        """
        Only works for RP+ZP and reward_model_type =='dqn'
        """
        h = self.q_net.encoder(obs)

        _, q_vals = self.select_action(
            obs, torch_mode=True, output_value=True)

        # Sample random states across episodes at random
        perm = torch.randperm(self.batch_size, device=self.device)
        h2 = h[perm]
        q_vals2 = q_vals[perm]
        if obs2 is None:
            obs2 = obs[perm]

        with torch.no_grad():
            if self.use_target_encoder:
                h_target = self.target_net.encoder(obs)
            if self.cheat_reward_transition:
                reward, pred_next_latent_mu1_ = self.get_reward_next_state_from_env(obs)
                with torch.no_grad():
                    if self.use_target_encoder:
                        pred_next_latent_mu1 = self.target_net.encoder(pred_next_latent_mu1_)
                    else:
                        pred_next_latent_mu1 = self.q_net.encoder(pred_next_latent_mu1_)
            else:
                if self.reward_decoder_type == 'dqn':
                    reward = self.reward_decoder(h)  # shape: (batch_size, na)
                    if self.use_target_encoder:
                        reward_target = self.reward_decoder(h_target)
                else:
                    raise NotImplementedError
                
                pred_next_latent_mu1 = self.get_all_action_next_latent(h)
                
                if self.use_target_encoder:
                    pred_next_latent_mu1_target = self.get_all_action_next_latent(h_target)

                # reward = self.reward_decoder(pred_next_latent_mu1)

            reward2 = reward[perm]
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            if self.use_target_encoder:
                reward2_target = reward_target[perm]
                pred_next_latent_mu2_target = pred_next_latent_mu1_target[perm]
        
        z_dist = self.metric_func(h, h2)  # shape: (batch_size)
        r_dist = self.reward_dist(reward, reward2).squeeze()  # shape: (batch_size, action_dim)
        if self.use_target_encoder:
            r_dist_target = self.reward_dist(reward_target, reward2_target).squeeze()
        L.log('train/norm', h.norm(p=1, dim=-1).mean(), step)

        if self.transition_model_type in ['', 'deterministic']:
            with torch.no_grad():
                # shape: (batch_size, action_dim)
                transition_dist = self.metric_func(pred_next_latent_mu1, pred_next_latent_mu2, target=True)
                if self.use_target_encoder:
                    transition_dist_target = self.metric_func(pred_next_latent_mu1_target, pred_next_latent_mu2_target, target=True)
        else:
            raise NotImplementedError

        with torch.no_grad():
            if self.use_target_encoder:
                # shape: (batch_size, action_dim)
                bisimilarity = self.c_R * r_dist + self.c_T * transition_dist
                bism_target = self.c_R * r_dist_target + self.c_T * transition_dist_target
                _, max_idx = torch.max(bisimilarity, dim=-1)  # shape: (batch_size)
                batch_indices = torch.arange(self.batch_size, device=self.device)
                max_bism = bism_target[batch_indices, max_idx]  # shape: (batch_size)
            else:
                bisimilarity = self.c_R * r_dist + self.c_T * transition_dist
                max_bism, _ = torch.max(bisimilarity, dim=-1)

        # Used only for tracking purposes
        with torch.no_grad():
            self.mu_bd = 0.95 * self.mu_bd + 0.05 * max_bism.mean().item()
            self.mu_rd = 0.95 * self.mu_rd + 0.05 * r_dist.mean().item()
            self.mu_nzd = 0.95 * self.mu_nzd + 0.05 * transition_dist.mean().item()
            self.mu_zd = 0.95 * self.mu_zd + 0.05 * z_dist.mean().item()

            L.log('train/mu_rd', self.mu_rd, step)
            L.log('train/mu_bd', self.mu_bd, step)
            L.log('train/mu_nzd', self.mu_nzd, step)
            L.log('train/mu_zd', self.mu_zd, step)
            L.log('train/mu_rd_div_bd', self.mu_rd / self.mu_bd, step)

            if self.debug:
                non_zero_indices = torch.nonzero(z_dist.detach(), as_tuple=True)[0]
                q_val_diff = torch.abs(q_vals - q_vals2)
                v_d_lips_vec = torch.div(q_val_diff[non_zero_indices], z_dist[non_zero_indices])
                v_d_lipschitz_cnt = (v_d_lips_vec > (1.0 / self.c_R)).float().mean().item()
                v_d_lipschitz = v_d_lips_vec.max().item()
                del q_val_diff
                del v_d_lips_vec
                raw_obs_dist = (obs - obs2).norm(p=1, dim=-1)
                d_raw_lipschitz = torch.div(z_dist[non_zero_indices], raw_obs_dist[non_zero_indices]).max().item()
                del raw_obs_dist
                del non_zero_indices

                L.log('train/v_d_lipschitz', v_d_lipschitz, step)
                L.log('train/v_d_lipschitz_cnt', v_d_lipschitz_cnt, step)
                L.log('train/d_raw_lipschitz', d_raw_lipschitz, step)
        
        # loss = ((z_dist - max_bism).abs() - 0.1).clip(min=0).mean()
        if self.metric_loss_type == 'huber':
            loss = F.smooth_l1_loss(z_dist, max_bism, reduction='mean')
        else:
            loss = (z_dist - max_bism).pow(2).mean()
        L.log('train/metric_loss', loss, step)
        return loss

    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h = self.q_net.encoder(obs)
        action_onehot = F.one_hot(action.long(), self.action_dim).float().squeeze()  # discrete
        h_act = torch.cat([h, action_onehot], dim=1)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(h_act)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.q_net.encoder(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train/transition_loss', loss, step)

        # pred_next_latent = self.transition_model.sample_prediction(h_act)
        if self.reward_decoder_type == 'dqn':
            pred_next_reward = self.reward_decoder(h).gather(1, action)
        elif self.reward_decoder_type == 'transition':
            pred_next_reward = self.reward_decoder(h_act)
        if self.reward_decoder_type != 'none':
            reward_loss = F.mse_loss(pred_next_reward, reward)
            L.log('train/reward_loss', reward_loss, step)
            with torch.no_grad():  # TODO: may delete later for other envs
                max_rew_indices = torch.nonzero(reward == self.max_reward, as_tuple=True)[0]
                reward_loss_at_max = F.mse_loss(pred_next_reward[max_rew_indices], reward[max_rew_indices])
                if not torch.isnan(reward_loss_at_max):
                    L.log('train/reward_loss_at_max', reward_loss_at_max.item(), step)
                del max_rew_indices
        else:
            reward_loss = 0

        total_loss = loss + reward_loss
        return total_loss

    def fix_policy(self):
        self.q_net_fixed.load_state_dict(self.q_net.state_dict())
        self.is_policy_fixed = True

    def update(self, replay_buffer, L, step):
        # if step > 100000:  # verify if RP+ZP helps metric learning
        #     self.rp_zp = True

        if self.fixed_policy_after_n_steps > 0 and \
            step > self.fixed_policy_after_n_steps and not self.is_policy_fixed:
            self.fix_policy()

        # Update representation
        for i in range(self.repr_upd_phase_length):
            # obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
            data = replay_buffer.sample(self.batch_size)
            obs, action, reward, next_obs, not_done = \
                data.observations.float(), data.actions, data.rewards, \
                data.next_observations.float(), 1 - data.dones
            if i == 0:
                L.log('train/batch_reward', reward.mean(), step)

            if self.rp_zp:
                transition_reward_loss = self.update_transition_reward_model(
                    obs, action, next_obs, reward, L, step)
            else:
                transition_reward_loss = 0
            
            if self.bisim_coef == 0:
                encoder_loss = 0
            elif self.metric_type == 'pbsm':
                encoder_loss = self.update_encoder(obs, action, reward, L, step)
            elif self.metric_type == 'bsm':
                if self.use_uniformly_sampled_states:
                    obs_, obs2 = self.generate_random_state_pairs()
                    encoder_loss = self.update_encoder_bism(obs_, L, step, obs2)
                else:
                    encoder_loss = self.update_encoder_bism(obs, L, step)
            else:
                raise NotImplementedError
            
            total_loss = self.bisim_coef * encoder_loss + transition_reward_loss

            if total_loss != 0:
                self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            if self.bisim_dist == 'metric_net':
                self.metric_decoder_optimizer.zero_grad()
            if total_loss != 0:
                total_loss.backward()
                self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            if self.bisim_dist == 'metric_net':
                self.metric_decoder_optimizer.step()
                if step % self.q_target_update_freq == 0:
                    rl_utils.soft_update_params(
                        self.metric_decoder, self.metric_decoder_target,
                        self.encoder_tau
                    )

            if step % self.q_target_update_freq == 0:
                rl_utils.soft_update_params(
                    self.q_net.encoder, self.target_net.encoder,
                    self.encoder_tau
                )

        # Update q params using the last sampled batch
        if not self.is_policy_fixed:
            for i in range(self.q_upd_phase_length):
                self.update_q_nets(obs, action, reward, next_obs, not_done, L, step)
                if step % self.q_target_update_freq == 0:
                    rl_utils.soft_update_params(
                        self.q_net.q, self.target_net.q, self.critic_tau
                    )
                    # utils.soft_update_params(
                    #     self.q_net.q2, self.target_net.q2, self.critic_tau
                    # )

        del transition_reward_loss, encoder_loss, total_loss

    def save(self, model_dir, step):
        torch.save(
            self.q_net.state_dict(), '%s/q_net_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.reward_decoder.state_dict(),
            '%s/reward_decoder_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.q_net.load_state_dict(
            torch.load('%s/q_net_%s.pt' % (model_dir, step))
        )
        self.reward_decoder.load_state_dict(
            torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
        )