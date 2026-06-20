"""
Dataset for the FCC-ee beam-induced-background events.

Each raw event is an (N, >=8) float array with columns:
    [E_signed, betax, betay, betaz, x, y, z, process]

The continuous 7-feature representation fed to the model is:
    [log|E|, u_x, u_y, u_z, x, y, z]
where (u_x, u_y, u_z) are the unsquashed velocities. 
The sign of E encodes the lepton charge,
which is split off into a discrete pdg label {11: e-, -11: e+}.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import beta_unsquash_np


class MCPDataset(Dataset):
    """8-col input: [E_signed, betax, betay, betaz, x, y, z, process]."""

    def __init__(self, path, max_particles=64, min_particles=1, keep_fraction=1.0):
        raw = np.load(path, allow_pickle=True)
        if keep_fraction < 1.0:
            raw = raw[: int(len(raw) * keep_fraction)]

        self.pdg_to_idx = {11: 0, -11: 1}
        self.idx_to_pdg = {0: 11, 1: -11}

        events_cont, events_pdg, events_proc = [], [], []
        for ev in raw:
            if ev is None:
                continue
            ev = np.asarray(ev)
            if ev.ndim != 2 or ev.shape[1] < 8:
                raise RuntimeError(
                    f"Event has {ev.shape[1]} columns but 8 are required. "
                    "Re-run track_data.py to generate data with process labels."
                )
            if len(ev) < min_particles:
                continue

            ev = ev.astype(np.float32)
            E_signed = ev[:, 0]
            beta = ev[:, 1:4]
            pos  = ev[:, 4:7]
            proc = ev[:, 7].astype(np.int64).clip(0, 2)

            pdg = np.where(E_signed >= 0.0, 11, -11)
            pdg_idx = np.where(pdg == 11, 0, 1).astype(np.int64)
            Eabs = np.maximum(np.abs(E_signed), 1e-12)
            logE = np.log(Eabs)
            u = beta_unsquash_np(beta)
            cont = np.concatenate([logE[:, None], u, pos], axis=1).astype(np.float32)

            events_cont.append(cont)
            events_pdg.append(pdg_idx)
            events_proc.append(proc)

        if not events_cont:
            raise RuntimeError("No events left.")

        self.events_cont = events_cont
        self.events_pdg  = events_pdg
        self.events_proc = events_proc

        # Per-event process composition (used at sampling to draw correlated
        # per-event proc fractions from real events).
        proc_fracs = []
        for proc in self.events_proc:
            counts = np.bincount(proc, minlength=3).astype(np.float32)
            proc_fracs.append(counts / counts.sum())
        self.proc_fracs = np.array(proc_fracs, dtype=np.float32)

        self.max_particles = max_particles
        self.feat_dim = 7

        all_feats = np.concatenate(events_cont, axis=0)
        self.feat_mean = all_feats.mean(axis=0).astype(np.float32)
        self.feat_std  = np.maximum(all_feats.std(axis=0), 1e-6).astype(np.float32)

        self.multiplicities = np.array([len(ev) for ev in events_cont], dtype=np.int64)

        # Global per-process particle counts -- used for inverse-frequency
        # loss weighting in train().
        all_proc = np.concatenate(events_proc, axis=0)
        self.proc_counts = np.bincount(all_proc, minlength=3).astype(np.int64)

    def __len__(self):
        return len(self.events_cont)

    def __getitem__(self, idx):
        cont = self.events_cont[idx]
        pdg  = self.events_pdg[idx]
        proc = self.events_proc[idx]

        N = len(cont)
        Kmax = self.max_particles
        if N <= Kmax:
            chosen = np.arange(N)
        else:
            chosen = torch.randperm(N)[:Kmax].numpy()

        cont = cont[chosen]
        pdg  = pdg[chosen]
        proc = proc[chosen]
        K = cont.shape[0]

        cont_norm = (cont - self.feat_mean) / self.feat_std

        x0   = np.zeros((Kmax, self.feat_dim), dtype=np.float32)
        pdg0 = np.zeros((Kmax,), dtype=np.int64)
        proc0 = np.zeros((Kmax,), dtype=np.int64)
        mask = np.zeros((Kmax,), dtype=np.bool_)

        x0[:K] = cont_norm
        pdg0[:K] = pdg
        proc0[:K] = proc
        mask[:K] = True

        return (
            torch.from_numpy(x0),
            torch.from_numpy(pdg0),
            torch.from_numpy(proc0),
            torch.from_numpy(mask),
        )
