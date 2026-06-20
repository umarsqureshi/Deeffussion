"""
Diffusion machinery: the cosine noise schedule, the discrete pdg-flip
schedule, and the `DDPM` wrapper that owns the forward/reverse
process for the continuous features.

The process label is never diffused or regenerated. 
Only the 7 continuous features (Gaussian DDPM)
and the pdg label (discrete uniform flips) are noised.
"""

import math

import numpy as np
import torch


def make_cosine_beta_schedule(T: int, s: float = 0.008, device: str = "cpu"):
    """Nichol & Dhariwal cosine schedule. Returns (betas, alphas, acp, acp_prev)."""
    steps = torch.arange(T + 1, device=device) / T
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0, 0.999)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
    return betas, alphas, acp, acp_prev


def make_linear_gamma_schedule(T: int, g0: float, g1: float, device: str):
    """Linear flip-probability schedule for discrete pdg diffusion."""
    return torch.linspace(g0, g1, T, device=device)


def q_sample_pdg(pdg0: torch.Tensor, t: torch.Tensor, gammas: torch.Tensor,
                 n_classes: int, mask: torch.Tensor):
    """Discrete forward diffusion of pdg only (process is NOT diffused in
    this version). Each real particle's pdg label is flipped to a uniform
    random class with probability ``gammas[t]``."""
    B, K = pdg0.shape
    g = gammas[t].view(B, 1)
    u = torch.rand((B, K), device=pdg0.device)
    flip = (u < g) & mask
    pdg_t = pdg0.clone()
    if flip.any():
        pdg_t[flip] = torch.randint(0, n_classes, (int(flip.sum()),), device=pdg0.device)
    return pdg_t


class DDPM:
    """DDPM wrapper around a denoiser model.

    Owns the cosine schedule buffers and the forward (``q_sample``) and
    reverse (``p_sample`` / ``sample``) processes for the continuous
    features.
    """

    def __init__(self, model, T, device, cosine_s=0.008, clip_x_norm: float | None = None):
        self.model = model
        self.T = T
        self.device = device
        # Hard clamp on the normalised reverse-diffusion state. The reverse
        # process is otherwise unbounded; on a non-Gaussian (e.g. bimodal
        # in z) target the model can drift by several sigma per step and the
        # accumulated drift unnormalises into outliers many sigma off.
        # Clamping in NORMALISED space at, say, 4 is well outside the body
        # of any feature so it doesn't bias training data, but it gates the
        # tails. Set to None or <=0 to disable.
        self.clip_x_norm = float(clip_x_norm) if clip_x_norm and clip_x_norm > 0 else None

        betas, alphas, acp, acp_prev = make_cosine_beta_schedule(T, s=cosine_s, device=device)
        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.acp_prev = acp_prev
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_1m_acp = torch.sqrt(1.0 - acp)
        self.posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)

    def q_sample(self, x0, t, noise):
        """Forward process: add t steps of noise to a clean state x0."""
        B = x0.shape[0]
        a = self.sqrt_acp[t].view(B, 1, 1)
        b = self.sqrt_1m_acp[t].view(B, 1, 1)
        return a * x0 + b * noise

    def p_sample(self, x_t, t, pdg_t, proc, mask):
        """Single denoising step.
        Returns (x_prev, pdg_logits). proc is conditioning; it is not generated."""
        B = x_t.shape[0]
        eps_hat, pdg_logits = self.model(x_t, t, pdg_t, proc, mask)

        beta_t  = self.betas[t].view(B, 1, 1)
        alpha_t = self.alphas[t].view(B, 1, 1)
        acp_t   = self.acp[t].view(B, 1, 1)

        mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_hat)
        var = self.posterior_variance[t].view(B, 1, 1)
        z = torch.zeros_like(x_t) if t[0].item() == 0 else torch.randn_like(x_t)
        x_prev = (mu + torch.sqrt(var) * z) * mask.unsqueeze(-1)
        if self.clip_x_norm is not None:
            x_prev = x_prev.clamp(-self.clip_x_norm, self.clip_x_norm)
        return x_prev, pdg_logits

    @torch.no_grad()
    def sample(self, mask, pdg_init, proc_init, num_steps=None):
        """Reverse diffusion. proc_init is held FIXED across all timesteps
        (it is conditioning, not a generated quantity). pdg is still
        regenerated at each step from the model's logits."""
        B, K = mask.shape
        x = torch.randn((B, K, 7), device=self.device) * mask.unsqueeze(-1)
        pdg = pdg_init.clone()
        proc = proc_init.clone()  # never updated; just for safety on dtype/device

        if num_steps is None:
            timesteps = list(reversed(range(self.T)))
        else:
            num_steps = max(1, min(num_steps, self.T))
            timesteps = np.linspace(0, self.T - 1, num_steps, dtype=int)
            timesteps = list(reversed(timesteps))

        for ti in timesteps:
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)
            x, pdg_logits = self.p_sample(x, t, pdg, proc, mask)

            pdg_probs = torch.softmax(pdg_logits, dim=-1)
            pdg_samp  = torch.multinomial(pdg_probs.view(-1, pdg_probs.size(-1)), 1).view(B, K)
            pdg       = torch.where(mask, pdg_samp, pdg)

        return x, pdg, proc
