"""
Loss functions for Deeffussion.
"""

import torch
import torch.nn.functional as F


def charge_balance_loss(pdg_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Soft charge balance: penalises squared expected (N(e-) - N(e+))."""
    probs    = torch.softmax(pdg_logits, dim=-1)
    p_eminus = probs[:, :, 0] * mask
    p_eplus  = probs[:, :, 1] * mask
    imbalance = (p_eminus - p_eplus).sum(dim=1)
    return (imbalance ** 2).mean()


def diffusion_mse(eps_hat: torch.Tensor, noise: torch.Tensor,
                  mask: torch.Tensor, proc_weights_per_part: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency-weighted MSE on the predicted noise.

    Args:
        eps_hat: predicted noise, (B, K, 7)
        noise:   target noise,    (B, K, 7)
        mask:    valid-particle mask, (B, K)
        proc_weights_per_part: per-particle reweighting, (B, K)
    """
    mse = (eps_hat - noise).pow(2).sum(dim=-1)            # (B, K)
    denom_w = (mask * proc_weights_per_part).sum().clamp(min=1)
    return (mse * mask * proc_weights_per_part).sum() / denom_w


def compute_total_loss(eps_hat, pdg_logits, noise, pdg0, mask,
                       proc_weights_per_part, lambda_pdg, lambda_charge):
    """Combine the three loss terms. Returns (total, diff, pdg, charge)."""
    diff_loss = diffusion_mse(eps_hat, noise, mask, proc_weights_per_part)
    pdg_loss = F.cross_entropy(pdg_logits[mask], pdg0[mask])
    c_loss = charge_balance_loss(pdg_logits, mask)
    total = diff_loss + lambda_pdg * pdg_loss + lambda_charge * c_loss
    return total, diff_loss, pdg_loss, c_loss
