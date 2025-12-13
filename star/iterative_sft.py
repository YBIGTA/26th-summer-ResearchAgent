import argparse
import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import List, Tuple

import chess

from .data.io import append_jsonl
from .data.schema import (
    ContrastRecord,
    DeltaRecord,
    EngineEvalRecord,
    EngineMetadata,
    RationaleRecord,
    SampleRecord,
)
from .engine.stockfish_wrapper import EngineConfig, StockfishWrapper
from .rationale.verify import parse_and_verify


logging.basicConfig(level=logging.INFO)


PROMPT_MOVE_ONLY = """You are a chess engine assistant.\nGiven the position in FEN: {fen}\nOutput a single best move in UCI format.\nOutput format:\nMOVE: <uci>\n"""

PROMPT_MOVE_RATIONALE = """You are a chess reasoning agent.\nGiven the position in FEN: {fen}\nPropose one move in UCI and a verifiable rationale.\nConstraints:\n- Provide a short evidence line in UCI (4 to 8 ply).\n- Provide a contrast move and a refutation line.\nOutput format:\nMOVE: ...\nCLAIM: ...\nEVIDENCE_PV: ...\nCONTRAST_MOVE: ...\nCONTRAST_PV: ...\n"""


def _sample_candidates(board: chess.Board, num_candidates: int) -> List[str]:
    moves = list(board.legal_moves)
    random.shuffle(moves)
    return [m.uci() for m in moves[:num_candidates]]


def _best_eval(evals: List[EngineEvalRecord]) -> EngineEvalRecord:
    def key_fn(ev: EngineEvalRecord) -> Tuple:
        if ev.mate is not None:
            return (100000 - abs(ev.mate),)
        return (ev.cp or -999999,)

    return sorted(evals, key=key_fn, reverse=True)[0]


def _calc_delta(best: EngineEvalRecord, rejected: EngineEvalRecord) -> DeltaRecord:
    cp_best = best.cp or 0
    cp_rej = rejected.cp or 0
    wdl_best = best.wdl
    wdl_rej = rejected.wdl
    win_prob = None
    if wdl_best and wdl_rej:
        total_best = sum(wdl_best)
        total_rej = sum(wdl_rej)
        if total_best and total_rej:
            win_prob = (wdl_best[0] / total_best) - (wdl_rej[0] / total_rej)
    return DeltaRecord(cp=cp_best - cp_rej, wdl_win_prob=win_prob)


def _make_rationale(
    fen: str,
    best: EngineEvalRecord,
    rejected: EngineEvalRecord,
    rationale_enabled: bool,
) -> RationaleRecord:
    rationale_text = (
        f"MOVE: {best.move}\n"
        f"CLAIM: Engine preferred move with strongest evaluation\n"
        f"EVIDENCE_PV: {' '.join(best.pv[:8])}\n"
        f"CONTRAST_MOVE: {rejected.move}\n"
        f"CONTRAST_PV: {' '.join(rejected.pv[:8])}\n"
    )
    parsed = parse_and_verify(
        fen,
        rationale_text,
        wdl_best=best.wdl,
        wdl_rejected=rejected.wdl,
        delta_cp=(best.cp or 0) - (rejected.cp or 0),
    )
    verification = parsed["verification"]
    return RationaleRecord(
        claim=parsed.get("claim", ""),
        evidence_pv=parsed.get("evidence_pv", []),
        contrast=ContrastRecord(
            alt_move=parsed.get("contrast_move", rejected.move),
            refutation_pv=parsed.get("contrast_pv", rejected.pv[:8]),
            note="Engine generated contrast",
        ),
        verified=rationale_enabled and verification.verified,
        error_type=verification.error_type,
    )


def run_iteration(
    fen: str,
    engine: StockfishWrapper,
    num_candidates: int,
    rationale_enabled: bool,
) -> SampleRecord:
    board = chess.Board(fen)
    candidates = _sample_candidates(board, num_candidates)
    engine_meta = EngineMetadata(
        name=engine.engine_name(),
        depth=engine.config.depth,
        multipv=engine.config.multipv,
        time_ms=engine.config.time_ms or 0,
    )

    all_evals: List[EngineEvalRecord] = []
    for cand in candidates:
        evals = engine.analyze(fen, moves=[cand], multipv=engine.config.multipv)
        all_evals.extend(
            [
                EngineEvalRecord(
                    move=e.move,
                    cp=e.cp,
                    mate=e.mate,
                    wdl=e.wdl,
                    pv=e.pv,
                )
                for e in evals
            ]
        )

    if not all_evals:
        raise RuntimeError("Engine returned no evaluations")

    best_eval = _best_eval(all_evals)
    rejected_eval = min(all_evals, key=lambda e: e.cp or -999999)
    delta = _calc_delta(best_eval, rejected_eval)
    rationale = _make_rationale(fen, best_eval, rejected_eval, rationale_enabled)

    return SampleRecord(
        id=str(uuid.uuid4()),
        fen=fen,
        side_to_move="w" if board.turn == chess.WHITE else "b",
        candidates=candidates,
        engine=engine_meta,
        evals=all_evals,
        chosen=best_eval.move,
        rejected=rejected_eval.move,
        delta=delta,
        rationale=rationale,
    )


def run_iterative_sft(args: argparse.Namespace) -> None:
    fen = args.fen or chess.STARTING_FEN
    config = EngineConfig(
        path=args.engine_path,
        depth=args.depth,
        multipv=args.multipv,
        time_ms=args.time_ms,
    )
    engine = StockfishWrapper(config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "sft_iter_data.jsonl"

    for step in range(args.iterations):
        logging.info("Iteration %d", step)
        sample = run_iteration(fen, engine, args.num_candidates, args.enable_rationale)
        append_jsonl(str(output_path), [sample])
        logging.info("Saved sample %s", sample.id)
        _mock_train_step(sample, out_dir)

    engine.close()


def _mock_train_step(sample: SampleRecord, out_dir: Path) -> None:
    """Placeholder for policy distillation / SFT update."""
    train_log = out_dir / "train_log.json"
    payload = {
        "sample_id": sample.id,
        "timestamp": time.time(),
        "chosen": sample.chosen,
    }
    with train_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload))
        f.write("\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterative SFT for chess")
    parser.add_argument("--mode", choices=["sft_iter"], default="sft_iter")
    parser.add_argument("--engine_path", required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--time_ms", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="star/data")
    parser.add_argument("--enable_rationale", action="store_true")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.mode != "sft_iter":
        raise ValueError("Only sft_iter mode supported in this entry point")
    run_iterative_sft(args)


if __name__ == "__main__":
    main()
