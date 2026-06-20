"""
Sampling/generation for Deeffusion.

Loads a trained checkpoint plus its ``meta.pt``, draws per-event
multiplicities and per-event process compositions from the real training
distribution, runs reverse diffusion, and decodes the normalised model
output back to physical units. The output is written as an object array of
(K_i, 9) events with columns:
    [pdg, E, betax, betay, betaz, x, y, z, process]
"""

import os
import math
import time

import numpy as np
import torch
from tqdm import tqdm

from diffusion import DDPM
from model import ParticleDenoiser
from utils import GPUMonitor, beta_squash_np


def load_meta_and_model(outdir: str, device: str, clip_x_norm: float | None = None):
    """Reconstruct the model + DDPM wrapper from a training output dir."""
    meta_path = os.path.join(outdir, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)

    model = ParticleDenoiser(
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
        n_process=int(meta.get("n_process", 3)),
    ).to(device)

    ckpt_path = os.path.join(outdir, "ckpt_last.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(
        model=model,
        T=int(meta["T"]),
        device=device,
        cosine_s=float(meta.get("cosine_s", 0.008)),
        clip_x_norm=clip_x_norm,
    )
    return meta, model, ddpm


def decode_batch(x_norm_batch, pdg_idx_batch, proc_idx_batch, Ks, meta):
    """Convert normalised model outputs back to physical units.
    Returns a list of (K_i, 9) float32 arrays:
        [pdg, E, betax, betay, betaz, x, y, z, process]
    """
    mean = np.asarray(meta["feat_mean"], dtype=np.float32)
    std  = np.asarray(meta["feat_std"],  dtype=np.float32)
    idx_to_pdg = meta["idx_to_pdg"]

    out = []
    for i, K in enumerate(Ks):
        x_i    = x_norm_batch[i, :K]
        pdg_i  = pdg_idx_batch[i, :K]
        proc_i = proc_idx_batch[i, :K]

        cont = x_i * std + mean
        logE = cont[:, 0]
        u    = cont[:, 1:4]
        pos  = cont[:, 4:7]

        E    = np.exp(logE)
        beta = beta_squash_np(u)
        pdg  = np.array([idx_to_pdg[int(j)] for j in pdg_i], dtype=np.int64)

        ev = np.concatenate(
            [pdg[:, None].astype(np.float32),
             E[:, None].astype(np.float32),
             beta.astype(np.float32),
             pos.astype(np.float32),
             proc_i[:, None].astype(np.float32)],
            axis=1,
        )
        out.append(ev)
    return out


def sample_batch(meta: dict, ddpm: DDPM, device: str, batch_size: int, num_steps=None):
    """Generate a batch of events conditioned on per-event multiplicity AND
    per-event process composition drawn from the same real event."""
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    proc_fracs    = np.asarray(meta["proc_fracs"], dtype=np.float32)
    Kmax  = int(meta["max_particles"])
    n_pdg = int(meta["n_pdg"])

    event_idx = np.random.randint(len(multiplicities), size=batch_size)
    Ks = np.clip(multiplicities[event_idx], 1, Kmax).astype(np.int64)

    mask_np = np.zeros((batch_size, Kmax), dtype=np.bool_)
    for i, K in enumerate(Ks):
        mask_np[i, :K] = True
    mask_t = torch.from_numpy(mask_np).to(device)

    pdg_init = torch.randint(0, n_pdg, (batch_size, Kmax), device=device)
    pdg_init = pdg_init * mask_t.long()

    # Per-event process composition: draws from same real event as the
    # multiplicity. With clean conditioning this composition is preserved
    # across all reverse-diffusion steps (DDPM.sample never overwrites it).
    proc_np = np.zeros((batch_size, Kmax), dtype=np.int64)
    for i, K in enumerate(Ks):
        frac = proc_fracs[event_idx[i]]
        proc_np[i, :K] = np.random.choice(3, size=K, p=frac)
    proc_init = torch.from_numpy(proc_np).to(device)

    with torch.no_grad():
        x_norm, pdg_idx, proc_idx = ddpm.sample(
            mask_t, pdg_init, proc_init, num_steps=num_steps
        )

    x_norm_np  = x_norm.cpu().numpy()
    pdg_idx_np = pdg_idx.cpu().numpy()
    proc_idx_np = proc_idx.cpu().numpy()

    return decode_batch(x_norm_np, pdg_idx_np, proc_idx_np, Ks, meta)


def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir      = args.outdir
    n_events    = args.n_events
    batch_size  = args.sample_batch_size
    num_steps   = getattr(args, "num_steps", None)
    clip_x_norm = getattr(args, "clip_x_norm", None)
    print("Using device:", device)
    print("n_events:", n_events)
    print("sample_batch_size:", batch_size)
    print("num_steps:", num_steps)
    print("clip_x_norm:", clip_x_norm if (clip_x_norm and clip_x_norm > 0) else "off")

    meta, model, ddpm = load_meta_and_model(outdir, device, clip_x_norm=clip_x_norm)
    if num_steps is not None:
        print(f"Sampling with {num_steps} steps (out of {meta['T']} total)")
    else:
        print(f"Sampling with full {meta['T']} steps")

    monitor = GPUMonitor(interval=0.2)
    monitor.start()
    start_time = time.time()

    out_events = []
    n_done = 0
    n_batches = math.ceil(n_events / batch_size)
    for _ in tqdm(range(n_batches), desc="Generating batches"):
        remaining = n_events - n_done
        bs = min(batch_size, remaining)
        out_events.extend(sample_batch(meta, ddpm, device, bs, num_steps=num_steps))
        n_done += bs

    end_time = time.time()
    monitor.stop()
    monitor.join()

    elapsed = end_time - start_time
    print("\n" + "="*60)
    print(f"Generated {n_events} events in {elapsed:.2f} seconds")
    print(f"Average time per event: {(elapsed/n_events)*1000:.4f} ms")
    print(f"Throughput: {n_events/elapsed:.1f} events/sec")
    print("="*60)
    monitor.print_stats()
    print("="*60)

    out_path = os.path.join(outdir, f"generated_events_{num_steps}steps.npy")
    np.save(out_path, np.array(out_events, dtype=object))
    print("Saved:", out_path)
