"""
Configuration for the BiBDM diffusion model.

All training/sampling hyper-parameters live in a single dataclass so that
the defaults are documented in one place and can be overridden either from
the command line (see ``main.py``) or programmatically.
"""

from dataclasses import dataclass

import torch


@dataclass
class CFG:
    # ----- I/O -----------------------------------------------------------
    data_path: str = "/fs/ddn/sdf/group/atlas/d/umarsqur/FCC_DDPM/guineapig_raw_trimmed_with_process_full.npy"
    outdir: str = "/fs/ddn/sdf/group/atlas/d/umarsqur/FCC_DDPM_noised_with_process_condclean_full/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Dataset -------------------------------------------------------
    max_particles: int = 1300
    min_particles: int = 1
    keep_fraction: float = 1.0

    # ----- Diffusion schedule -------------------------------------------
    T: int = 1000
    cosine_s: float = 0.002273608087915074

    # ----- Transformer trunk --------------------------------------------
    d_model: int = 512
    nhead: int = 2
    num_layers: int = 3
    dropout: float = 0.08545214831324029

    # ----- Optimisation --------------------------------------------------
    batch_size: int = 2
    lr: float = 0.00015982608583855863
    epochs: int = 50
    num_workers: int = 0
    grad_clip: float = 4.587196486849941
    seed: int = 123

    me: float = 0.00051099895069  # electron mass [GeV]

    # ----- Feature / class dimensions -----------------------------------
    feat_dim: int = 7
    n_pdg: int = 2
    n_process: int = 3
    lambda_pdg: float = 0.3522458811249478
    lambda_charge: float = 0.011241862095793064

    # pdg is still discretely diffused (we GENERATE pdg). Process is NOT.
    gamma_start: float = 5.232216089948759e-05
    gamma_end: float = 0.17539090890647513

    # Inverse-frequency loss weighting for the diffusion MSE.
    # weights ~ 1 / counts^proc_weight_alpha, then normalised so mean=1
    # over the dataset, then clipped. alpha=1 is full inverse-freq;
    # alpha=0.5 is square-root, which has lower gradient variance for the
    # rarest class.
    proc_weight_alpha: float = 0.5
    proc_weight_min: float = 0.2
    proc_weight_max: float = 5.0

    # ----- Sampling ------------------------------------------------------
    n_events: int = 3000
    sample_batch_size: int = 1
    num_steps: int = None

    # ----- OneCycleLR ----------------------------------------------------
    pct_start: float = 0.07173423081368346
    div_factor: float = 30.86403908528625
