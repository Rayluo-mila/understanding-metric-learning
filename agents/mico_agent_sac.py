import torch
import torch.nn.functional as F

from agents.bisim_agent_sac import BisimAgent


class MICoAgent(BisimAgent):
    """
    MICo-based agent
    https://github.com/google-research/google-research/blob/bb19948d367f3337c16176232e86069bf36b0bf5/mico
    https://github.com/bit1029public/SimSR
    """

    def __init__(self, obs_shape, action_shape, actor_action_max, cfg, L):
        super().__init__(obs_shape, action_shape, actor_action_max, cfg, L)
        self.sample_next_state_from_buffer = False
        if self.cfg.bisim_dist in ["simsr_basic", "mico"]:
            self.sample_next_state_from_buffer = True

    def update_encoder(self, obs, next_obs, action, reward, step):
        """
        Update the metrics (encoders) using sampled rewards and transition samples.
        """
        z = self.critic.encoder(obs)
        if self.cfg.use_target_encoder:
            z_target = self.critic_target.encoder(obs).detach()
        else:
            z_target = z

        with torch.no_grad():
            # Target encoder for computing z' from x' is used in both SimSR-basic and MICo
            if self.sample_next_state_from_buffer:
                pred_z_prime = self.critic_target.encoder(next_obs)
            else:
                pred_z_prime = self.transition_model.sample_prediction(
                    torch.cat([z_target, action], dim=1)
                )  # shape (B, Z)

        z, z2, reward, reward2, pred_z_prime, pred_z_prime2, _, _ = (
            self.generate_bisim_samples(z, z_target, reward, pred_z_prime, None)
        )

        r_dist = self.reward_dist(reward, reward2)
        transition_dist = self.metric_func(pred_z_prime, pred_z_prime2)
        z_dist = self.metric_func(z, z2)

        bisimilarity = self.cfg.c_R * r_dist + self.cfg.c_T * transition_dist
        if self.cfg.metric_loss_type == "huber":
            loss = F.smooth_l1_loss(z_dist, bisimilarity.detach(), reduction="mean")
        else:
            loss = (z_dist - bisimilarity.detach()).pow(2).mean()

        # Stats
        r_dist_l1 = torch.abs(reward - reward2)
        self.log_metric_stats(
            bisimilarity, r_dist, z_dist, transition_dist, loss, step, r_dist_l1
        )
        return loss
