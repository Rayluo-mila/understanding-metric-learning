import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.rl_utils as rl_utils
from agents.models import weight_init
from agents.transition_model import make_transition_model
from agents.decoder import make_decoder
from agents.base_agent import SACAgent


class DeepMDPAgent(SACAgent):
    """
    SAC with a transition model, reward model,
    and pixel decoder (for reconstruction or next obs reconstruction).
    """

    def __init__(self, obs_shape, action_shape, actor_action_max, cfg, L):
        super().__init__(obs_shape, action_shape, actor_action_max, cfg, L)
        self.reconstruction = False
        if cfg.decoder_type == "reconstruction":
            cfg.decoder_type = "pixel"
            self.reconstruction = True

        self.transition_model = make_transition_model(
            cfg.transition_model_type,
            cfg.encoder_feature_dim,
            action_shape,
            layer_width=cfg.transition_model_hidden_size,
        ).to(cfg.device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(
                cfg.encoder_feature_dim + action_shape[0],
                cfg.reward_decoder_hidden_size,
            ),
            nn.LayerNorm(cfg.reward_decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.reward_decoder_hidden_size, 1),
        ).to(cfg.device)

        decoder_params = list(self.transition_model.parameters()) + list(
            self.reward_decoder.parameters()
        )

        self.decoder = None
        if cfg.decoder_type == "pixel":
            # create decoder
            self.decoder = make_decoder(
                cfg.decoder_type,
                obs_shape,
                cfg.encoder_feature_dim,
                cfg.num_layers,
                cfg.num_filters,
            ).to(cfg.device)
            self.decoder.apply(weight_init)
            decoder_params += list(self.decoder.parameters())

        self.decoder_optimizer = torch.optim.Adam(
            decoder_params, lr=cfg.decoder_lr, weight_decay=cfg.decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=cfg.encoder_lr
        )

        self.train()

    def print_model_stats(self):
        super().print_model_stats()
        if self.decoder is not None:
            print(f"Decoder params count: {rl_utils.count_parameters(self.decoder)}")
            print(f"Decoder: {self.decoder}")
        print(
            f"Reward decoder params count: {rl_utils.count_parameters(self.reward_decoder)}"
        )
        print(
            f"Transition model params count: {rl_utils.count_parameters(self.transition_model)}"
        )

    def update_transition_reward_model(self, obs, action, next_obs, reward, step):
        """
        Implementing RP + ZP losses.
        """
        h = self.critic.encoder(obs)
        transition_loss = 0
        if self.cfg.zp:
            pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(
                torch.cat([h, action], dim=1)
            )
            if pred_next_latent_sigma is None:
                pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

            next_h = self.critic.encoder(next_obs)
            diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
            transition_loss = torch.mean(
                0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma)
            )
            self.L.log("train_ae/transition_loss", transition_loss.item(), step)

        reward_loss = 0
        if self.cfg.rp:
            pred_next_reward = self.reward_decoder(torch.cat([h, action], dim=1))
            reward_loss = F.mse_loss(pred_next_reward, reward)
            self.L.log("train_ae/reward_loss", reward_loss.item(), step)
        total_loss = self.cfg.rzp_coef * (transition_loss + reward_loss)

        if self.cfg.report_grad_norm:
            encoder_parameters = [
                p for p in self.critic.encoder.parameters() if p.requires_grad
            ]
            self.grad_norm_analysis(
                total_loss, encoder_parameters, step, label="transition_reward_loss"
            )

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_decoder(self, obs, action, target_obs, step):  # uses transition model
        # image might be stacked, just grab the first 3 (rgb)!
        assert target_obs.dim() == 4, target_obs.shape
        target_obs = target_obs[:, :3, :, :]

        h = self.critic.encoder(obs)
        if not self.reconstruction:
            next_h = self.transition_model.sample_prediction(
                torch.cat([h, action], dim=1)
            )
            if target_obs.dim() == 4:
                # preprocess images to be in [-0.5, 0.5] range
                target_obs = rl_utils.preprocess_obs(target_obs)
            rec_obs = self.decoder(next_h)
            loss = F.mse_loss(target_obs, rec_obs)
        else:
            rec_obs = self.decoder(h)
            loss = F.mse_loss(obs, rec_obs)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.L.log("train_ae/ae_loss", loss, step)

    def update(self, batch_transition, step, **kwargs):
        obs, action, _, reward, next_obs, not_done = batch_transition

        self.L.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, step)
        self.update_transition_reward_model(obs, action, next_obs, reward, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, step)

        self.soft_update_params(step)

        if (
            self.decoder is not None and step % self.decoder_update_freq == 0
        ):  # decoder_type is pixel
            self.update_decoder(obs, action, next_obs, step)
