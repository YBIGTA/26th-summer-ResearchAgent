"""Command-line entrypoint for iterative chess verifier pipelines."""
from __future__ import annotations

import argparse
import logging

from .iterative_dpo import run_iterative_dpo
from .iterative_sft import run_iterative_sft


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chess verifier pipelines")
    parser.add_argument("--mode", choices=["sft_iter", "dpo_iter"], required=True)
    parser.add_argument("--engine_path", required=True, help="Path to Stockfish (or UCI engine)")
    parser.add_argument("--fen", default=None, help="Position FEN; defaults to starting FEN if omitted")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--time_ms", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="star/data")
    parser.add_argument("--enable_rationale", action="store_true", help="Toggle rationale parsing for SFT")
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    parser = build_argparser()
    args = parser.parse_args()

    if args.mode == "sft_iter":
        run_iterative_sft(args)
    elif args.mode == "dpo_iter":
        run_iterative_dpo(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
