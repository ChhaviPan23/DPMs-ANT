import math
import numpy as np
import torch
import torch.nn.functional as F


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Reference from Improved Denoising Diffusion Probabilistic Models (https://github.com/openai/improved-diffusion).
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        beta_start = beta_start
        beta_end = beta_end
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def get_alpha_and_beta(config):
    betas = torch.from_numpy(get_named_beta_schedule(schedule_name=config.dm.schedule_name,
                                                     num_diffusion_timesteps=config.dm.num_diffusion_timesteps,
                                                     beta_start=config.dm.beta_start,
                                                     beta_end=config.dm.beta_end)).float()
    betas = betas.cuda(non_blocking=True)
    alphas = (1 - betas).cumprod(dim=0).cuda(non_blocking=True)
    return alphas, betas


def extract(x, t):
    return x.index_select(0, t + 1).view(-1, 1, 1, 1)


def generate_xt(x0, a, B, num_timesteps, e=None):
    if e is None:
        e = torch.randn_like(x0).requires_grad_(True).cuda(non_blocking=True)

    # antithetic sampling
    t = torch.randint(low=0, high=num_timesteps, size=(B // 2 + 1,)).cuda()
    t = torch.cat([t, num_timesteps - t - 1], dim=0)[:B]
    at = a.index_select(0, t).view(-1, 1, 1, 1)
    xt = x0 * at.sqrt() + e * (1.0 - at).sqrt()

    return e, xt, t


def cond_fn(classifier, x, t, y, classifier_scale=1.0):
    assert x.shape[0] == y.shape[0], "The image label size do not match the noised image size"
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
