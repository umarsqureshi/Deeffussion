"""
General-purpose utilities.
"""

import sys
import subprocess
import threading
import time

import numpy as np
import torch


def set_seed(seed: int):
    """Seed python, numpy and torch (incl. all CUDA devices)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n, val_frac, seed):
    """Deterministic shuffled train/val index split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    return idx[n_val:], idx[:n_val]


def beta_squash_np(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map an unbounded R^3 vector into the open unit ball (|beta| < 1)."""
    u = np.asarray(u, dtype=np.float32)
    umag = np.linalg.norm(u, axis=1, keepdims=True)
    uhat = u / (umag + 1e-12)
    s = np.tanh(umag)
    return (1.0 - eps) * s * uhat


def beta_unsquash_np(beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse of :func:`beta_squash_np`: unit ball -> unbounded R^3."""
    beta = np.asarray(beta, dtype=np.float32)
    bmag = np.linalg.norm(beta, axis=1, keepdims=True)
    bmag = np.clip(bmag, 0.0, 1.0 - eps)
    bhat = beta / (bmag + 1e-12)
    s = bmag / (1.0 - eps)
    umag = np.arctanh(np.clip(s, 0.0, 1.0 - 1e-7))
    return umag * bhat


def mem_snapshot(tag: str, step: int):
    """Print an RSS / CUDA / /dev/shm memory snapshot to stderr."""
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss_kb / (1024 * 1024)
    except Exception:
        rss_gb = float("nan")
    try:
        shm = subprocess.check_output(
            ["df", "-P", "-BM", "/dev/shm"], encoding="utf-8"
        ).strip().splitlines()[-1].split()
        shm_used, shm_avail = shm[2], shm[3]
    except Exception:
        shm_used, shm_avail = "?", "?"
    cuda_alloc_gb = (
        torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    )
    cuda_reserved_gb = (
        torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
    )
    print(
        f"[mem] {tag} step={step} rss={rss_gb:.2f}GB "
        f"cuda_alloc={cuda_alloc_gb:.2f}GB cuda_reserved={cuda_reserved_gb:.2f}GB "
        f"/dev/shm used={shm_used} avail={shm_avail}",
        file=sys.stderr, flush=True,
    )


class GPUMonitor(threading.Thread):
    """Background thread that polls ``nvidia-smi`` for utilisation/VRAM."""

    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.gpu_utils = []
        self.mem_used = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    encoding="utf-8",
                )
                lines = result.strip().split('\n')
                if lines:
                    util, mem = lines[0].split(',')
                    self.gpu_utils.append(float(util.strip()))
                    self.mem_used.append(float(mem.strip()))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

    def print_stats(self):
        if not self.gpu_utils:
            print("No GPU stats collected.")
            return
        avg_util = sum(self.gpu_utils) / len(self.gpu_utils)
        max_util = max(self.gpu_utils)
        max_mem = max(self.mem_used)
        print(f"--- GPU Monitoring Stats ---")
        print(f"Avg GPU Utilization: {avg_util:.1f}%")
        print(f"Max GPU Utilization: {max_util:.1f}%")
        print(f"Max VRAM Used:       {max_mem:.0f} MB")
        if torch.cuda.is_available():
            print(f"PyTorch Max VRAM:    {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
        print(f"----------------------------")
