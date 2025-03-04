import torch
import torch.nn.functional as F
import math

def _sqrt(x, tol=1e-9):
    tol_vec = torch.full_like(x, tol)
    return torch.sqrt(torch.maximum(x, tol_vec))

def metric_func(x, y, cfg, metric_ub=None, step=None, L=None, override=None):
    bisim_dist_type = cfg.bisim_dist if override is None else override
    if bisim_dist_type == 'L1':
        dist = torch.linalg.norm(x-y, ord=1, dim=-1)
    elif bisim_dist_type == 'L1_mean':
        dist = torch.abs(x-y).mean(dim=-1)
    elif bisim_dist_type == 'L2':
        dist = torch.linalg.norm(x-y, ord=2, dim=-1)
    elif bisim_dist_type == 'L2_mean':
        dist = torch.linalg.norm(x-y, ord=2, dim=-1)
        num_features = x.size(-1)
        dist = dist / num_features
    elif bisim_dist_type == 'huber':
        dist = F.smooth_l1_loss(x, y, reduction='none').sum(dim=-1)
    elif bisim_dist_type == 'huber_mean':
        dist = F.smooth_l1_loss(x, y, reduction='none').mean(dim=-1)
    elif bisim_dist_type == 'mico':
        beta = cfg.mico_beta
        cos_similarity = F.cosine_similarity(x, y, dim=-1, eps=1e-9)
        base_distances = torch.atan2(_sqrt(1. - cos_similarity.pow(2.)), cos_similarity)
        norm_average = x.pow(2.).sum(dim=-1) + y.pow(2.).sum(dim=-1)
        dist = norm_average + beta * base_distances
        if step is not None and L is not None:
            L.log('train_metric/angular_dist', (beta * base_distances).mean(), step)
            L.log('train_metric/norm_average', norm_average.mean(), step)
    elif 'simsr' in bisim_dist_type:
        cosine_similarity = F.cosine_similarity(x, y, dim=-1, eps=1e-9)
        dist = 1 - cosine_similarity
    else:
        raise NotImplementedError
    return dist

def prob_metric_func(mu, mu2, sigma, sigma2, cfg, metric_ub=None):
    if cfg.prob_dist == '':
        if 'mean' in cfg.bisim_dist:
            dist = torch.sqrt(
                (mu - mu2).pow(2) +
                (sigma - sigma2).pow(2)
            ).mean(dim=-1)
        else:
            mu_dist = torch.linalg.norm(mu - mu2, ord=2, dim=-1)
            sigma_dist = torch.linalg.norm(sigma - sigma2, ord=2, dim=-1)
            dist = torch.sqrt(mu_dist.pow(2) + sigma_dist.pow(2))
    else:
        if 'L2_rms' in cfg.prob_dist:
            mu_dist = torch.linalg.norm(mu - mu2, ord=2, dim=-1)  # [0, 2sqrt(n)]
            sigma_dist = torch.linalg.norm(sigma - sigma2, ord=2, dim=-1)  # [0, around sqrt(n)] as max_sigma=1
            if 'nostd' in cfg.prob_dist:
                dist = mu_dist * metric_ub / (2 * math.sqrt(mu.size(-1)))
            else:
                denom = math.sqrt(5 * mu.size(-1))
                dist = torch.sqrt(mu_dist.pow(2) + sigma_dist.pow(2)) * metric_ub / denom
        elif 'L2_mean' in cfg.prob_dist:
            if 'nostd' in cfg.prob_dist:
                dist = (mu - mu2).abs().mean(dim=-1)
            else:
                dist = torch.sqrt(
                    (mu - mu2).pow(2) +
                    (sigma - sigma2).pow(2)
                ).mean(dim=-1)
        elif 'L2' in cfg.prob_dist:
            mu_dist = torch.linalg.norm(mu - mu2, ord=2, dim=-1)
            sigma_dist = torch.linalg.norm(sigma - sigma2, ord=2, dim=-1)
            if 'nostd' in cfg.prob_dist:
                dist = mu_dist
            else:
                dist = torch.sqrt(mu_dist.pow(2) + sigma_dist.pow(2))
        elif 'L1' in cfg.prob_dist:
            mu_dist = torch.linalg.norm(mu - mu2, ord=1, dim=-1)
            sigma_dist = torch.linalg.norm(sigma - sigma2, ord=1, dim=-1)
            if 'nostd' in cfg.prob_dist:
                dist = mu_dist
            else:
                dist = torch.sqrt(mu_dist.pow(2) + sigma_dist.pow(2))
        elif 'huber_mean' in cfg.prob_dist:
            mu_dist = F.smooth_l1_loss(mu, mu2, reduction='none').mean(-1)
            if 'nostd' in cfg.prob_dist:
                sigma_dist = torch.tensor(0., device=mu.device)
            else:
                sigma_dist = F.smooth_l1_loss(sigma, sigma2, reduction='none').mean(-1)
            dist = mu_dist + sigma_dist
        elif 'huber' in cfg.prob_dist:
            mu_dist = F.smooth_l1_loss(mu, mu2, reduction='none').sum(-1)
            if 'nostd' in cfg.prob_dist:
                sigma_dist = torch.tensor(0., device=mu.device)
            else:
                sigma_dist = F.smooth_l1_loss(sigma, sigma2, reduction='none').sum(-1)
            dist = mu_dist + sigma_dist
        elif cfg.prob_dist == 'zero':
            dist = torch.tensor(0., device=mu.device)
        else:
            raise NotImplementedError
    if metric_ub is not None:
        dist = torch.clamp(dist, max=metric_ub, min=0.)
    return dist

def reward_dist(reward, reward2, cfg, r_vars=None):
    with torch.no_grad():
        if cfg.r_dist == 'L1':
            r_dist = (reward - reward2).abs()  # shape: (batch_size, action_dim)
        elif cfg.r_dist == 'L2':
            r_dist = (reward - reward2).pow(2)
        elif cfg.r_dist == 'huber':
            r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
        elif cfg.r_dist == 'rap_sqrt':
            # Adapt from https://github.com/jianda-chen/RAP_distance/blob/main/agent/rap.py
            r_dist = (reward - reward2).pow(2.)
            r_dist = F.relu(r_dist - r_vars[0] - r_vars[1])
            r_dist = r_dist.sqrt()
        elif cfg.r_dist == 'rap':
            # Adapt from https://github.com/jianda-chen/RAP_distance/blob/main/agent/rap.py
            r_dist = (reward - reward2).pow(2.)
            r_dist = r_dist - r_vars[0] - r_vars[1]
        else:
            raise NotImplementedError(f"Unsupported r_dist: {cfg.r_dist}")
    return r_dist.squeeze(dim=-1)
