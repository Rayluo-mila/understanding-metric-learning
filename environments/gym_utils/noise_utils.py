import numpy as np
import torch
from functools import lru_cache


@lru_cache(maxsize=None)
def get_noise_dist(noise_dim, noise_std, noise_mean=0):
    """
    Returns the mean and covariance matrix for a multivariate normal distribution,
    cached for each unique (noise_dim, noise_std) pair.
    """
    mean = np.full(noise_dim, noise_mean)
    cov = (noise_std ** 2) * np.eye(noise_dim)
    return mean, cov

def generate_full_rank_matrix(n, m, noise_std):
    """
    Generate an n x m Gaussian random matrix with entries drawn from N(0, noise_std^2)
    and ensure that it has full column rank (i.e., rank == m). A Gaussian random matrix is 
    full rank with probability 1, but this check is included for extra safety.
    """
    while True:
        A = np.random.normal(0, noise_std, (n, m))
        if np.linalg.matrix_rank(A) == m:
            return A

def get_inverse_matrix(matrix):
    if matrix.shape[0] == matrix.shape[1]:
        return torch.inverse(torch.from_numpy(matrix)).numpy()
    else:
        return torch.pinverse(torch.from_numpy(matrix), rcond=1e-20).numpy()

def recover_obs(projected_obs, inv_proj_matrix):
    """
    Recovers the original observation from the projected observation using the pseudoinverse
    of the projection matrix. Since the matrix is full column rank, a left inverse exists.
    
    @param projected_obs (np.ndarray): The observation after projection.
    @return (np.ndarray): The recovered original observation.
    """
    # Recover the original observation: x = A_pinv @ y
    original_obs = torch.matmul(torch.from_numpy(inv_proj_matrix), torch.from_numpy(projected_obs)).numpy()
    return original_obs

def random_proj(obs, proj_matrix):
    """
    Projects the original observation from R^m to R^(m + noise_dim) using a
    precomputed Gaussian random matrix.
    """
    projected_obs = torch.matmul(torch.from_numpy(proj_matrix), torch.from_numpy(obs)).numpy()
    return projected_obs

def append_white_noise(obs, noise_dim, noise_std, noise_mean=0):
    """
    Appends white noise sampled from a multivariate normal distribution
    to the observation `obs`. The noise is generated using NumPy.
    """
    if noise_dim == 0:
        return obs
    mean, cov = get_noise_dist(noise_dim, noise_std, noise_mean)
    noise = np.random.multivariate_normal(mean, cov)
    new_obs = np.concatenate([obs, noise])
    return new_obs