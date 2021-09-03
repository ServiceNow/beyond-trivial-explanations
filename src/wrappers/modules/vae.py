import torch

def get_kl_loss(mu, logvar, prior_mu=0):
    return 0.5 * (-1 - logvar + mu ** 2 + torch.exp(logvar)).sum() / mu.size(0)