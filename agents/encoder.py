import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    # adapted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/modules/dynamics_module.py#L18
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)


def AvgL1Norm(x, eps=1e-8):
    # adapted from https://github.com/sfujim/TD7/blob/main/TD7.py
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class BasePixelEncoder(nn.Module):
    """Base convolutional encoder for pixel observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=stride)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = self.compute_output_dim(obs_shape[1], obs_shape[2])
        
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.max_norm = kwargs.get("max_norm")
        self.max_norm_ord = kwargs.get("max_norm_ord")

    def compute_output_dim(self, h, w):
        for conv in self.convs:
            kernel_size_h, kernel_size_w = conv.kernel_size
            stride_h, stride_w = conv.stride

            h = (h - kernel_size_h) // stride_h + 1  # ceil((h - kernel_size + 2 * padding) / stride)
            w = (w - kernel_size_w) // stride_w + 1

        return h  # Assuming h and w are equal after convolution

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        raise NotImplementedError("This method should be implemented in the subclasses.")

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        pass

class BasePixelEncoder(nn.Module):
    """Base convolutional encoder for pixel observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=stride)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = self.compute_output_dim(obs_shape[1], obs_shape[2])
        
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.max_norm = kwargs.get("max_norm")
        self.max_norm_ord = kwargs.get("max_norm_ord")

    def compute_output_dim(self, h, w):
        for conv in self.convs:
            kernel_size_h, kernel_size_w = conv.kernel_size
            stride_h, stride_w = conv.stride

            h = (h - kernel_size_h) // stride_h + 1  # ceil((h - kernel_size + 2 * padding) / stride)
            w = (w - kernel_size_w) // stride_w + 1

        return h  # Assuming h and w are equal after convolution

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        conv = torch.relu(self.convs[0](obs))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        raise NotImplementedError("This method should be implemented in the subclasses.")

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        pass


class PixelEncoder(BasePixelEncoder):
    """Convolutional encoder of pixels observations with LayerNorm and optional max norm constraint."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__(obs_shape, feature_dim, num_layers, num_filters, stride, **kwargs)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        out = self.ln(h_fc)

        if self.max_norm:
            norm_to_max = (torch.linalg.norm(out, ord=self.max_norm_ord, dim=-1) / self.max_norm).clamp(min=1).unsqueeze(-1)
            out = out / norm_to_max

        return out


class PixelEncoderNoLayerNorm(BasePixelEncoder):
    """Convolutional encoder of pixels observations with LayerNorm and optional max norm constraint."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__(obs_shape, feature_dim, num_layers, num_filters, stride, **kwargs)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.fc(h)

        if self.max_norm:
            norm_to_max = (torch.linalg.norm(out, ord=self.max_norm_ord, dim=-1) / self.max_norm).clamp(min=1).unsqueeze(-1)
            out = out / norm_to_max

        return out


class PixelEncoderL2Normed(BasePixelEncoder):
    """Convolutional encoder of pixels observations with L2 normalization."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__(obs_shape, feature_dim, num_layers, num_filters, stride, **kwargs)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        h_norm = F.normalize(h_fc, dim=1, p=2)

        return h_norm


class PixelEncoderRMSNormed(BasePixelEncoder):
    """Convolutional encoder of pixels observations with RMS normalization."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__(obs_shape, feature_dim, num_layers, num_filters, stride, **kwargs)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        h_norm = F.normalize(h_fc, dim=1, p=2) * (h_fc.size(1) ** 0.5)

        return h_norm


class PixelEncoderAvgL1Normed(BasePixelEncoder):
    """Convolutional encoder of pixels observations with RMS normalization."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__(obs_shape, feature_dim, num_layers, num_filters, stride, **kwargs)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        h_norm = AvgL1Norm(h_fc)

        return h_norm


class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()
        print('Building PE-Carla096')
        print(f'\tNlayers = {num_layers}, Nfilters = {num_filters}, dim(feats) = {feature_dim}')
        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        #out_dims = 100  # if defaults change, adjust this as needed
        #out_dims = int( (84 * (84*5)) / 4 )
        # 35 = 84 / 2 - 3 - 2 - 2, 203 = 420 / 2 - 3 - 2 - 2
        out_dims = 35 * 203
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()
        print('Building PE-Carla098')

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        #out_dims = 56  # 3 cameras
        out_dims = 100  # 5 cameras
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args, **kwargs):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class MLPEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args, **kwargs):
        super().__init__()
        assert len(obs_shape) == 1
        obs_shape = obs_shape[0]
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, feature_dim)
        )
        self.feature_dim = feature_dim
        self.max_norm = kwargs.get("max_norm")
        self.max_norm_ord = kwargs.get("max_norm_ord")

    def forward(self, obs, detach=False, normalize=True):
        x = self.model(obs)
        if self.max_norm and normalize:
            x = self.normalize(x)
        if detach:
            x = x.detach()
        return x

    def normalize(self, x):
        if self.max_norm:
            norms = torch.linalg.norm(x, ord=self.max_norm_ord, dim=-1)
            norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
            x = x / norm_to_max
        return x

    def copy_conv_weights_from(self, source):
        source_layers = [m for m in source.modules() if isinstance(m, nn.Linear)]
        self_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        assert (len(self_layers) == len(source_layers))
        for self_layer, source_layer in zip(self_layers, source_layers):
            tie_weights(src=source_layer, trg=self_layer)

    def log(self, L, step, log_freq):
        pass


class MLPEncoderLayerNormed(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args, **kwargs):
        super().__init__()
        assert len(obs_shape) == 1
        obs_shape = obs_shape[0]
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, feature_dim)
        )
        self.feature_dim = feature_dim
        self.max_norm = kwargs.get("max_norm")
        self.max_norm_ord = kwargs.get("max_norm_ord")
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward(self, obs, detach=False, normalize=True):
        x = self.model(obs)
        x = self.ln(x)
        if self.max_norm and normalize:
            x = self.normalize(x)
        if detach:
            x = x.detach()
        return x

    def normalize(self, x):
        if self.max_norm:
            norms = torch.linalg.norm(x, ord=self.max_norm_ord, dim=-1)
            norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
            x = x / norm_to_max
        return x

    def copy_conv_weights_from(self, source):
        source_layers = [m for m in source.modules() if isinstance(m, nn.Linear)]
        self_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        assert (len(self_layers) == len(source_layers))
        for self_layer, source_layer in zip(self_layers, source_layers):
            tie_weights(src=source_layer, trg=self_layer)

    def log(self, L, step, log_freq):
        pass


class MLPEncoderL2Normed(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args, **kwargs):
        super().__init__()
        assert len(obs_shape) == 1
        obs_shape = obs_shape[0]
        self.model = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, feature_dim)
        )
        self.feature_dim = feature_dim

    def forward(self, obs, detach=False, normalize=True):
        x = self.model(obs)
        if normalize:
            x = self.normalize(x)
        if detach:
            x = x.detach()
        return x

    def normalize(self, x):
        x = F.normalize(x, dim=1, p=2)
        return x

    def copy_conv_weights_from(self, source):
        source_layers = [m for m in source.modules() if isinstance(m, nn.Linear)]
        self_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        assert (len(self_layers) == len(source_layers))
        for self_layer, source_layer in zip(self_layers, source_layers):
            tie_weights(src=source_layer, trg=self_layer)

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixel_nolayernormed': PixelEncoderNoLayerNorm,
                       'pixel_layernormed': PixelEncoder,
                       'pixel_l2normed': PixelEncoderL2Normed,
                       'pixel_avg_l1normed': PixelEncoderAvgL1Normed,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'pixel_rmsnormed': PixelEncoderRMSNormed,
                       'identity': IdentityEncoder,
                       'mlp': MLPEncoder,
                       'mlp_layernormed': MLPEncoderLayerNormed,
                       'mlp_l2normed': MLPEncoderL2Normed,
                       }


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride,
    max_norm=None, max_norm_ord=None
):
    print(f"Encoder_name: {encoder_type}")
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, stride,
        max_norm=max_norm, max_norm_ord=max_norm_ord
    )
