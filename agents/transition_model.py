import random
import torch
import torch.nn as nn


class DeterministicTransitionModel(nn.Module):
    def __init__(
        self,
        encoder_feature_dim,
        action_shape,
        layer_width,
        encoder_max_norm=None,
        encoder_max_norm_ord=None,
    ):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.max_norm = encoder_max_norm
        self.max_norm_ord = encoder_max_norm_ord
        print("Deterministic transition model chosen.")

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        if self.max_norm:
            mu = self.normalize(mu)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x):
        mu, _ = self(x)
        return mu

    def normalize(self, x):
        norms = torch.linalg.norm(x, ord=self.max_norm_ord, dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max
        return x


class ProbabilisticTransitionModel(nn.Module):
    def __init__(
        self,
        encoder_feature_dim,
        action_shape,
        layer_width,
        announce=True,
        max_sigma=1e1,
        min_sigma=1e-4,
        encoder_max_norm=None,
        encoder_max_norm_ord=None,
    ):
        super().__init__()
        self.fc = nn.Linear(encoder_feature_dim + action_shape[0], layer_width)
        self.ln = nn.LayerNorm(layer_width)
        self.fc_mu = nn.Linear(layer_width, encoder_feature_dim)
        self.fc_sigma = nn.Linear(layer_width, encoder_feature_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert self.max_sigma >= self.min_sigma
        if announce:
            print("Probabilistic transition model chosen.")

        self.max_norm = encoder_max_norm
        self.max_norm_ord = encoder_max_norm_ord

    def forward(self, x, normalize=True):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)

        mu = self.fc_mu(x)
        if self.max_norm and normalize:
            mu = self.normalize(mu)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = (
            self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        )  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x, normalize=False)
        eps = torch.randn_like(sigma)
        ret = mu + sigma * eps
        if self.max_norm:
            ret = self.normalize(ret)
            # WARNING: not adjusting for non-linear change in distribution.
        return ret

    def normalize(self, x):
        norms = torch.linalg.norm(x, ord=self.max_norm_ord, dim=-1)
        norm_to_max = (norms / self.max_norm).clamp(min=1).unsqueeze(-1)
        x = x / norm_to_max
        return x


class BaseEnsembleTransitionModel(object):
    def __init__(
        self,
        model_class,
        encoder_feature_dim,
        action_shape,
        layer_width,
        ensemble_size=5,
        announce=False,
        encoder_max_norm=None,
        encoder_max_norm_ord=None,
    ):
        self.models = [
            model_class(
                encoder_feature_dim,
                action_shape,
                layer_width,
                announce=announce,
                encoder_max_norm=encoder_max_norm,
                encoder_max_norm_ord=encoder_max_norm_ord,
            )
            for _ in range(ensemble_size)
        ]
        print(f"Ensemble of {model_class.__name__} transition models chosen.")

    def __call__(self, x):
        mu_sigma_list = [model.forward(x) for model in self.models]
        mus, sigmas = zip(*mu_sigma_list)
        mus, sigmas = torch.stack(mus), torch.stack(sigmas)
        return mus, sigmas

    def sample_prediction(self, x):
        model = random.choice(self.models)
        return model.sample_prediction(x)

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def parameters(self, recurse=True):
        assert recurse
        list_of_parameters = [list(model.parameters()) for model in self.models]
        parameters = [p for ps in list_of_parameters for p in ps]
        return parameters


class EnsembleOfProbabilisticTransitionModels(BaseEnsembleTransitionModel):
    def __init__(
        self,
        encoder_feature_dim,
        action_shape,
        layer_width,
        ensemble_size=5,
        encoder_max_norm=None,
        encoder_max_norm_ord=None,
    ):
        super().__init__(
            ProbabilisticTransitionModel,
            encoder_feature_dim,
            action_shape,
            layer_width,
            ensemble_size,
            encoder_max_norm,
            encoder_max_norm_ord,
        )


class EnsembleOfDeterministicTransitionModels(BaseEnsembleTransitionModel):
    def __init__(
        self,
        encoder_feature_dim,
        action_shape,
        layer_width,
        ensemble_size=5,
        encoder_max_norm=None,
        encoder_max_norm_ord=None,
    ):
        super().__init__(
            DeterministicTransitionModel,
            encoder_feature_dim,
            action_shape,
            layer_width,
            ensemble_size,
            encoder_max_norm,
            encoder_max_norm_ord,
        )


_AVAILABLE_TRANSITION_MODELS = {
    "": DeterministicTransitionModel,
    "deterministic": DeterministicTransitionModel,
    "probabilistic": ProbabilisticTransitionModel,
    "ensemble": EnsembleOfProbabilisticTransitionModels,
    "ensemble_det": EnsembleOfDeterministicTransitionModels,
}


def make_transition_model(
    transition_model_type,
    encoder_feature_dim,
    action_shape,
    layer_width=512,
    encoder_max_norm=None,
    encoder_max_norm_ord=None,
):
    assert transition_model_type in _AVAILABLE_TRANSITION_MODELS
    return _AVAILABLE_TRANSITION_MODELS[transition_model_type](
        encoder_feature_dim,
        action_shape,
        layer_width,
        encoder_max_norm=encoder_max_norm,
        encoder_max_norm_ord=encoder_max_norm_ord,
    )
