import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.encoder import make_encoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi, action_max=1.):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = action_max * torch.tanh(mu)
    if pi is not None:
        pi = action_max * torch.tanh(pi)
        if log_pi is not None:
            log_pi -= torch.log(F.relu(action_max - (1. / action_max) * pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def vector_norm(x, ord=2, dim=-1):
    # Check the PyTorch version
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    
    # If PyTorch version is 1.9.0 or newer, use torch.linalg.vector_norm
    if torch_version >= (1, 9):
        return torch.linalg.vector_norm(x, ord=ord, dim=dim)
    # For versions older than 1.9.0, use torch.norm
    else:
        return torch.norm(x, p=ord, dim=dim)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, action_max, log_std_min, log_std_max, num_layers, num_filters, stride,
        encoder_max_norm=None, encoder_max_norm_ord=None
    ):
        super().__init__()

        # print('obs shape', obs_shape)

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, stride,
            max_norm=encoder_max_norm, max_norm_ord=encoder_max_norm_ord
        )
        self.action_max = action_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False, z=None
    ):
        # print('obs shape', obs.shape) # Carla 1 x 9 x 84 x 420 = 317520
        # sys.exit(0)
        z = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(z).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi, action_max=self.action_max)
        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        pass


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, stride,
        encoder_max_norm=None, encoder_max_norm_ord=None, encoder=None
    ):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = make_encoder(
                encoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters, stride,
                max_norm=encoder_max_norm, max_norm_ord=encoder_max_norm_ord
            )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False, require_hnorm=False, cnn_output=None):
        # detach_encoder allows to stop gradient propogation to encoder
        z = self.encoder(obs, detach=detach_encoder)
        if require_hnorm:
            with torch.no_grad():
                z_norm = {}
                z_norm['L1_norm'] = vector_norm(z, ord=1, dim=-1).mean().item()
                z_norm['L2_norm'] = vector_norm(z, ord=2, dim=-1).mean().item()
                z_norm['L_inf_norm'] = vector_norm(z, ord=float('inf'), dim=-1).mean().item()

        q1 = self.Q1(z, action)
        q2 = self.Q2(z, action)

        if require_hnorm: return q1, q2, z_norm
        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        pass


class Clamp(nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, min=self.min_value, max=self.max_value)