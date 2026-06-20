"""
Model definition for Deeffusion.

The backbone is a pre-norm transformer encoder that
predicts the Gaussian noise on the 7 continuous features and the pdg
(charge) logits. The process label enters as an
embedding but is never predicted by a head.

Token embedding:
    h^(0) = phi_x(x_t) + phi_t(t) + phi_pdg(pdg_t) + phi_proc(pi_0) + phi_K(log K)
followed by L pre-norm encoder layers with a residual skip, then two
linear heads (eps and pdg).
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of the (scalar) diffusion timestep."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ParticleDenoiser(nn.Module):
    """Process is now CLEAN conditioning, not a generated quantity.

    Forward signature: (x_t, t, pdg_t, proc, mask) -> (eps_hat, pdg_logits)
    The proc tensor is always the ground-truth (training) or seed (sampling)
    process label; it is NEVER replaced by a model prediction.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1,
                 n_pdg=2, n_process=3):
        super().__init__()
        self.d_model = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.mom_proj = nn.Linear(7, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, 7)

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.k_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        self.pdg_emb = nn.Embedding(n_pdg, d_model)
        self.pdg_head = nn.Linear(d_model, n_pdg)

        # Process is CONDITIONING ONLY: embedding stays, head is gone.
        self.process_emb = nn.Embedding(n_process, d_model)

        self.skip_alpha = 0.2

    def forward(self, x_t, t, pdg_t, proc, mask):
        B, K, _ = x_t.shape

        t_emb = self.time_emb(t)
        t_emb = self.t_mlp(t_emb).unsqueeze(1).expand(B, K, self.d_model)

        mom_emb = self.mom_proj(x_t)
        pdg_emb = self.pdg_emb(pdg_t.clamp(0, self.pdg_emb.num_embeddings - 1))
        proc_emb = self.process_emb(proc.clamp(0, self.process_emb.num_embeddings - 1))

        h = t_emb + mom_emb + pdg_emb + proc_emb

        K_event = mask.sum(dim=1)
        k = torch.log(K_event.float().clamp(min=1)).unsqueeze(-1)
        k_emb = self.k_mlp(k).unsqueeze(1)
        h = h + k_emb

        src_key_padding_mask = ~mask
        h_in = h
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = h + self.skip_alpha * h_in
        h = h * mask.unsqueeze(-1)

        eps_hat = self.output(h) * mask.unsqueeze(-1)
        pdg_logits = self.pdg_head(h)
        return eps_hat, pdg_logits
