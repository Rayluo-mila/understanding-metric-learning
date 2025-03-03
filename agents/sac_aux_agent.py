import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.rl_utils as rl_utils
from agents.models import weight_init
from agents.transition_model import make_transition_model
from agents.decoder import make_decoder
from agents.base_agent import SACAgent


class SACAuxAgent(SACAgent):
    """SAC with transition model and various decoder types."""
    def __init__(self, obs_shape, action_shape, actor_action_max, cfg, L):
        super().__init__(obs_shape, action_shape, actor_action_max, cfg, L)
        self.hinge = 1.

        self.transition_model = make_transition_model(
            cfg.transition_model_type, cfg.encoder_feature_dim, action_shape,
            layer_width=cfg.transition_model_hidden_size
        ).to(cfg.device)

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            self.transition_model.parameters(),
            lr=cfg.decoder_lr,
            weight_decay=cfg.decoder_weight_lambda
        )

        self.decoder = None
        encoder_params = list(self.critic.encoder.parameters()) + list(self.transition_model.parameters())
        if cfg.decoder_type == 'pixel':
            # create decoder
            self.decoder = make_decoder(
                cfg.decoder_type, obs_shape, cfg.encoder_feature_dim, cfg.num_layers,
                cfg.num_filters
            ).to(cfg.device)
            self.decoder.apply(weight_init)
        elif cfg.decoder_type == 'inverse':
            self.inverse_model = nn.Sequential(
                nn.Linear(cfg.encoder_feature_dim * 2, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, action_shape[0])).to(cfg.device)
            encoder_params += list(self.inverse_model.parameters())
        if cfg.decoder_type != 'identity':
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=cfg.encoder_lr)
        if cfg.decoder_type == 'pixel':  # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=cfg.decoder_lr,
                weight_decay=cfg.decoder_weight_lambda
            )
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=cfg.encoder_lr
            )

        self.train()

    def print_model_stats(self):
        super().print_model_stats()
        if self.decoder is not None:
            print(f'Decoder params count: {rl_utils.count_parameters(self.decoder)}')
            print(f'Decoder: {self.decoder}')

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
            normalization = 0.
        else:
            pred_trans_mu, pred_trans_sigma = self.transition_model(torch.cat([state, action], dim=1))
            if pred_trans_sigma is None:
                pred_trans_sigma = torch.Tensor([1.]).to(self.device)
            if isinstance(pred_trans_mu, list):  # i.e. comes from an ensemble
                raise NotImplementedError  # TODO: handle the additional ensemble dimension (0) in this case
            diff = (state + pred_trans_mu - next_state) / pred_trans_sigma
            normalization = torch.log(pred_trans_sigma)
        return norm * (diff.pow(2) + normalization).sum(1)

    def contrastive_loss(self, state, action, next_state):
        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]

        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state, no_trans=True)).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def update_decoder(self, obs, action, target_obs, step):  #  uses transition model
        # image might be stacked, just grab the first 3 (rgb)!
        assert target_obs.dim() == 4, target_obs.shape
        target_obs = target_obs[:, :3, :, :]

        h = self.critic.encoder(obs)
        next_h = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = rl_utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(next_h)
        loss = F.mse_loss(target_obs, rec_obs)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.L.log('train_ae/ae_loss', loss, step)

    def update_contrastive(self, obs, action, next_obs, step):
        latent = self.critic.encoder(obs)
        next_latent = self.critic.encoder(next_obs)
        loss = self.contrastive_loss(latent, action, next_latent)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.L.log('train_ae/contrastive_loss', loss, step)

    def update_inverse(self, obs, action, next_obs, step):
        non_final_mask = torch.tensor(tuple(map(lambda s: not (s == 0).all(), next_obs)), device=self.device).long()  # hack
        latent = self.critic.encoder(obs[non_final_mask])
        next_latent = self.critic.encoder(next_obs[non_final_mask].to(self.device).float())
        # pred_next_latent = self.transition_model(torch.cat([latent, action], dim=1))
        # fpred_action = self.inverse_model(latent, pred_next_latent)
        pred_action = self.inverse_model(torch.cat([latent, next_latent], dim=1))
        loss = F.mse_loss(pred_action, action[non_final_mask])  # + F.mse_loss(fpred_action, action)
        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.L.log('train_ae/inverse_loss', loss, step)

    def update(self, batch_transition, step, **kwargs):
        if self.decoder_type == 'inverse':
            obs, action, reward, next_obs, not_done, k_obs = batch_transition
        else:
            obs, action, _, reward, next_obs, not_done = batch_transition

        self.L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)

        self.soft_update_params(step)

        if self.decoder is not None and step % self.decoder_update_freq == 0:  # decoder_type is pixel
            self.update_decoder(obs, action, next_obs, step)

        if self.decoder_type == 'contrastive':
            self.update_contrastive(obs, action, next_obs, step)
        elif self.decoder_type == 'inverse':
            self.update_inverse(obs, action, k_obs, step)