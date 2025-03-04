import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.init_height = 4
        self.init_width = 4
        num_out_channels = 3  # rgb
        kernel = 3

        self.fc = nn.Linear(
            feature_dim, num_filters * self.init_height * self.init_width
        )

        self.deconvs = nn.ModuleList()

        pads = [0, 1, 0]
        for i in range(self.num_layers - 1):
            output_padding = pads[i]
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, kernel, stride=2, output_padding=output_padding)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, num_out_channels, kernel, stride=2, output_padding=1
            )
        )

    def forward(self, h):
        h = torch.relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.init_height, self.init_width)
        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
        obs = self.deconvs[-1](deconv)
        return obs

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_DECODERS = {'pixel': PixelDecoder}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
