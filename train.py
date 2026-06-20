"""
Training loop for Deeffusion.
"""

import os
import sys
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import MCPDataset
from diffusion import DDPM, make_linear_gamma_schedule, q_sample_pdg
from losses import compute_total_loss
from model import ParticleDenoiser
from utils import set_seed, split_indices, mem_snapshot


def train(args):
    cfg = CFG()
    print("Using device:", cfg.device)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    if args.data_path:    cfg.data_path = args.data_path
    if args.outdir:       cfg.outdir = args.outdir
    if args.max_particles:cfg.max_particles = args.max_particles
    if args.epochs:       cfg.epochs = args.epochs
    if args.batch_size:   cfg.batch_size = args.batch_size
    if args.T:            cfg.T = args.T
    if args.seed:         cfg.seed = args.seed

    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)

    # cuDNN's heuristic kernel selection occasionally lands on a path that
    # SIGSEGVs in the driver on the SDF Ampere nodes (driver 535.161, A100).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    val_frac = 0.1

    ds_full = MCPDataset(
        cfg.data_path,
        max_particles=cfg.max_particles,
        min_particles=cfg.min_particles,
        keep_fraction=cfg.keep_fraction,
    )

    # Inverse-frequency reweighting for the diffusion MSE.
    counts = ds_full.proc_counts.astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = counts ** (-cfg.proc_weight_alpha)
    w = inv / inv.mean()                          # mean = 1
    w = np.clip(w, cfg.proc_weight_min, cfg.proc_weight_max)
    w = w / w.mean()                              # re-normalise after clip
    proc_weights_t = torch.tensor(w, dtype=torch.float32, device=cfg.device)
    print(
        f"[reweight] per-process counts={counts.astype(np.int64).tolist()} "
        f"-> weights={w.round(3).tolist()} (alpha={cfg.proc_weight_alpha}, "
        f"clip=[{cfg.proc_weight_min},{cfg.proc_weight_max}])"
    )

    meta_path = os.path.join(cfg.outdir, "meta.pt")
    if getattr(args, "resume", False) and os.path.exists(meta_path):
        meta_old = torch.load(meta_path, map_location="cpu")
        if "train_idx" in meta_old and "val_idx" in meta_old:
            train_idx = np.asarray(meta_old["train_idx"])
            val_idx   = np.asarray(meta_old["val_idx"])
        else:
            train_idx, val_idx = split_indices(len(ds_full), val_frac, cfg.seed)
    else:
        train_idx, val_idx = split_indices(len(ds_full), val_frac, cfg.seed)

    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val   = torch.utils.data.Subset(ds_full, val_idx)

    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )

    model = ParticleDenoiser(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        n_process=cfg.n_process,
    ).to(cfg.device)

    ddpm = DDPM(model, cfg.T, cfg.device, cosine_s=cfg.cosine_s)
    gammas = make_linear_gamma_schedule(cfg.T, cfg.gamma_start, cfg.gamma_end, cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(dl_train),
        pct_start=cfg.pct_start,
        anneal_strategy='cos',
        div_factor=cfg.div_factor,
        final_div_factor=1e4,
    )

    # ----------------------------
    # Resume (optional)
    # ----------------------------
    start_epoch = 0
    train_losses = []
    val_losses = []

    ckpt_path = os.path.join(cfg.outdir, "ckpt_last.pt")
    if getattr(args, "resume", False) and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt and ckpt["opt"] is not None:
            opt.load_state_dict(ckpt["opt"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

        tl_path = os.path.join(cfg.outdir, "train_losses.npy")
        vl_path = os.path.join(cfg.outdir, "val_losses.npy")
        if os.path.exists(tl_path) and os.path.exists(vl_path):
            train_losses = list(np.load(tl_path).astype(float))
            val_losses   = list(np.load(vl_path).astype(float))

        print(f"Resuming from {ckpt_path} at epoch {start_epoch}")
    else:
        print("Starting training from scratch")

    meta = {
        "multiplicities": ds_full.multiplicities,
        "proc_fracs": ds_full.proc_fracs,
        "proc_counts": ds_full.proc_counts,
        "proc_weights": w.astype(np.float32),
        "feat_mean": ds_full.feat_mean,
        "feat_std": ds_full.feat_std,
        "feat_dim": ds_full.feat_dim,
        "me": cfg.me,
        "n_pdg": cfg.n_pdg,
        "n_process": cfg.n_process,
        "idx_to_pdg": ds_full.idx_to_pdg,
        "max_particles": cfg.max_particles,
        "T": cfg.T,
        "cosine_s": cfg.cosine_s,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "val_frac": val_frac,
        "seed": cfg.seed,
        "n_events": len(ds_full),
        "n_train_events": len(train_idx),
        "n_val_events": len(val_idx),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "conditioning": "clean_process",  # marker so analysis knows
    }
    if not os.path.exists(meta_path):
        torch.save(meta, meta_path)

    mem_snapshot("startup", -1)
    nan_warned = {"v": False}

    for epoch in range(start_epoch, cfg.epochs):
        # TRAIN
        model.train()
        total_train = 0.0
        n_train = 0

        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [train]", leave=False)
        for step, (x0, pdg0, proc0, mask) in enumerate(pbar):
            x0    = x0.to(cfg.device, non_blocking=False)
            pdg0  = pdg0.to(cfg.device, non_blocking=False)
            proc0 = proc0.to(cfg.device, non_blocking=False)
            mask  = mask.to(cfg.device, non_blocking=False)

            B = x0.shape[0]
            t = torch.randint(0, cfg.T, (B,), device=cfg.device)

            noise = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t   = ddpm.q_sample(x0, t, noise)

            pdg_t = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)
            # process is CLEAN conditioning; pass proc0 unchanged.
            eps_hat, pdg_logits = model(x_t, t, pdg_t, proc0, mask)

            w_per_part = proc_weights_t[proc0]                    # (B, K)
            loss, _, _, _ = compute_total_loss(
                eps_hat, pdg_logits, noise, pdg0, mask,
                w_per_part, cfg.lambda_pdg, cfg.lambda_charge,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()

            loss_val = loss.item()
            if not nan_warned["v"] and not math.isfinite(loss_val):
                nan_warned["v"] = True
                print(
                    f"[warn] non-finite loss={loss_val} at epoch={epoch+1} step={step}",
                    file=sys.stderr, flush=True,
                )

            total_train += loss_val
            n_train += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step % 1000 == 0:
                mem_snapshot(f"epoch{epoch+1:03d}", step)

        train_loss = total_train / max(n_train, 1)

        # VALIDATION
        model.eval()
        total_val = 0.0
        n_val = 0

        with torch.no_grad():
            pbarv = tqdm(dl_val, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [val]", leave=False)
            for x0, pdg0, proc0, mask in pbarv:
                x0    = x0.to(cfg.device, non_blocking=False)
                pdg0  = pdg0.to(cfg.device, non_blocking=False)
                proc0 = proc0.to(cfg.device, non_blocking=False)
                mask  = mask.to(cfg.device, non_blocking=False)

                B = x0.shape[0]
                t = torch.randint(0, cfg.T, (B,), device=cfg.device)

                noise = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t   = ddpm.q_sample(x0, t, noise)

                pdg_t = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)
                eps_hat, pdg_logits = model(x_t, t, pdg_t, proc0, mask)

                w_per_part = proc_weights_t[proc0]
                loss, _, _, _ = compute_total_loss(
                    eps_hat, pdg_logits, noise, pdg0, mask,
                    w_per_part, cfg.lambda_pdg, cfg.lambda_charge,
                )

                total_val += loss.item()
                n_val += 1
                pbarv.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = total_val / max(n_val, 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(cfg.outdir, "val_losses.npy"),   np.array(val_losses))

        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        torch.save(
            {
                "model":      model.state_dict(),
                "opt":        opt.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
            },
            os.path.join(cfg.outdir, "ckpt_last.pt"),
        )

    np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cfg.outdir, "val_losses.npy"),   np.array(val_losses))
    print("Training complete. Outputs saved to:", cfg.outdir)
