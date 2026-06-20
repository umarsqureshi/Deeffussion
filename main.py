#!/usr/bin/env python3
"""
Command-line entry point for BiBDM.

Two sub-commands:
    train   -- train the conditional diffusion model
    sample  -- generate synthetic events from a trained checkpoint

Examples:
    python main.py train  --data_path data.npy --outdir runs/exp1 --resume
    python main.py sample --outdir runs/exp1 --n_events 3000 --num_steps 250
"""

import sys
import signal
import argparse
import faulthandler

# Print a Python traceback (and C-level frames where possible) the instant
# the process is hit by SIGBUS / SIGSEGV / SIGFPE / SIGABRT / SIGILL.
faulthandler.enable(file=sys.stderr, all_threads=True)
for _sig_name in ("SIGUSR1", "SIGUSR2"):
    _sig = getattr(signal, _sig_name, None)
    if _sig is not None:
        try:
            faulthandler.register(_sig, file=sys.stderr, all_threads=True, chain=False)
        except (ValueError, RuntimeError):
            pass

from config import CFG
from train import train
from sample import sample


def main():
    cfg = CFG()
    parser = argparse.ArgumentParser(
        description="BiBDM: conditional DDPM for FCC-ee beam-induced background "
                    "(clean process conditioning + inverse-freq weighting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    train_parser = subparsers.add_parser('train', help='Train the diffusion model')
    train_parser.add_argument('--data_path',     type=str, help='Path to training data (.npy)')
    train_parser.add_argument('--outdir',        type=str, help='Output directory')
    train_parser.add_argument('--max_particles', type=int, help='Max particles per event')
    train_parser.add_argument('--epochs',        type=int, help='Number of epochs')
    train_parser.add_argument('--batch_size',    type=int, help='Batch size')
    train_parser.add_argument('--T',             type=int, help='Diffusion steps')
    train_parser.add_argument('--seed',          type=int, help='Random seed')
    train_parser.add_argument('--resume', action='store_true',
                              help='Resume from outdir/ckpt_last.pt if it exists')

    sample_parser = subparsers.add_parser('sample', help='Generate synthetic events')
    sample_parser.add_argument('--outdir',             type=str, default=cfg.outdir)
    sample_parser.add_argument('--n_events',           type=int, default=cfg.n_events)
    sample_parser.add_argument('--sample_batch_size',  type=int, default=cfg.sample_batch_size)
    sample_parser.add_argument('--num_steps',          type=int, default=None)
    sample_parser.add_argument(
        '--clip_x_norm', type=float, default=4.0,
        help="Clamp the normalised reverse-diffusion state at +/- this many "
             "sigma at every step. Kills heavy-tail outliers from unbounded "
             "DDPM updates on non-Gaussian targets (e.g. bimodal LL z). "
             "Pass 0 to disable. Default: 4.0",
    )

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
