import argparse
import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import List

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


def _choose_preference(evals: List[EngineEvalRecord]) -> (EngineEvalRecord, EngineEvalRecord, DeltaRecord):
    sorted_evals = sorted(evals, key=lambda e: e.cp or -999999, reverse=True)
    chosen = sorted_evals[0]
    rejected = sorted_evals[-1]
    delta = DeltaRecord(cp=(chosen.cp or 0) - (rejected.cp or 0), wdl_win_prob=None)
    return chosen, rejected, delta


def _construct_rationale(fen: str, chosen: EngineEvalRecord, rejected: EngineEvalRecord) -> RationaleRecord:
    rationale_text = (
        f"MOVE: {chosen.move}\n"
        f"CLAIM: Engine preferred move for preference optimization\n"
        f"EVIDENCE_PV: {' '.join(chosen.pv[:8])}\n"
        f"CONTRAST_MOVE: {rejected.move}\n"
        f"CONTRAST_PV: {' '.join(rejected.pv[:8])}\n"
    )
    parsed = parse_and_verify(
        fen,
        rationale_text,
        wdl_best=chosen.wdl,
        wdl_rejected=rejected.wdl,
        delta_cp=(chosen.cp or 0) - (rejected.cp or 0),
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
        verified=verification.verified,
        error_type=verification.error_type,
    )


def _mock_dpo_update(out_dir: Path, sample: SampleRecord) -> None:
    log_path = out_dir / "dpo_log.json"
    payload = {
        "sample_id": sample.id,
        "timestamp": time.time(),
        "chosen": sample.chosen,
        "rejected": sample.rejected,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload))
        f.write("\n")


def run_iteration(
    fen: str,
    engine: StockfishWrapper,
    num_candidates: int,
) -> SampleRecord:
    board = chess.Board(fen)
    candidates = list(board.legal_moves)
    random.shuffle(candidates)
    candidates = [m.uci() for m in candidates[:num_candidates]]
    engine_meta = EngineMetadata(
        name=engine.engine_name(),
        depth=engine.config.depth,
        multipv=engine.config.multipv,
        time_ms=engine.config.time_ms or 0,
    )

    evals: List[EngineEvalRecord] = []
    for move in candidates:
        engine_evals = engine.analyze(fen, moves=[move], multipv=engine.config.multipv)
        evals.extend(
            [
                EngineEvalRecord(
                    move=e.move,
                    cp=e.cp,
                    mate=e.mate,
                    wdl=e.wdl,
                    pv=e.pv,
                )
                for e in engine_evals
            ]
        )

    if not evals:
        raise RuntimeError("Engine returned no evaluations")

    chosen, rejected, delta = _choose_preference(evals)
    rationale = _construct_rationale(fen, chosen, rejected)

    return SampleRecord(
        id=str(uuid.uuid4()),
        fen=fen,
        side_to_move="w" if board.turn == chess.WHITE else "b",
        candidates=candidates,
        engine=engine_meta,
        evals=evals,
        chosen=chosen.move,
        rejected=rejected.move,
        delta=delta,
        rationale=rationale,
    )


def run_iterative_dpo(args: argparse.Namespace) -> None:
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
    output_path = out_dir / "dpo_iter_data.jsonl"

    for step in range(args.iterations):
        logging.info("DPO Iteration %d", step)
        sample = run_iteration(fen, engine, args.num_candidates)
        append_jsonl(str(output_path), [sample])
        _mock_dpo_update(out_dir, sample)
        logging.info("Saved DPO sample %s", sample.id)

    engine.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterative DPO for chess")
    parser.add_argument("--mode", choices=["dpo_iter"], default="dpo_iter")
    parser.add_argument("--engine_path", required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--time_ms", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="star/data")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.mode != "dpo_iter":
        raise ValueError("Only dpo_iter mode supported in this entry point")
    run_iterative_dpo(args)


if __name__ == "__main__":
    main()
