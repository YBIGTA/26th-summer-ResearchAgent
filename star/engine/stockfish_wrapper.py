import contextlib
import dataclasses
import logging
from typing import Dict, List, Optional

import chess
import chess.engine


@dataclasses.dataclass
class EngineConfig:
    path: str
    depth: int = 18
    multipv: int = 1
    time_ms: Optional[int] = None
    options: Optional[Dict[str, str]] = None


@dataclasses.dataclass
class EngineEval:
    move: str
    cp: Optional[int]
    mate: Optional[int]
    wdl: Optional[List[int]]
    pv: List[str]


class StockfishWrapper:
    """Lightweight wrapper around a UCI chess engine (Stockfish compatible)."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine: Optional[chess.engine.SimpleEngine] = None

    @property
    def engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            logging.info("Initializing engine at %s", self.config.path)
            self._engine = chess.engine.SimpleEngine.popen_uci(self.config.path)
            if self.config.options:
                for key, value in self.config.options.items():
                    try:
                        self._engine.configure({key: value})
                    except chess.engine.EngineError:
                        logging.warning("Engine option %s not supported", key)
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.quit()
        self._engine = None

    def engine_name(self) -> str:
        """Return the engine-reported name if available."""
        _ = self.engine  # ensure engine is initialized
        if self._engine and getattr(self._engine, "id", None):
            return self._engine.id.get("name", "stockfish")
        return "stockfish"

    def analyze(
        self,
        fen: str,
        moves: Optional[List[str]] = None,
        multipv: Optional[int] = None,
        depth: Optional[int] = None,
        time_ms: Optional[int] = None,
    ) -> List[EngineEval]:
        board = chess.Board(fen)
        pv_count = multipv or self.config.multipv
        limit = chess.engine.Limit(
            depth=depth or self.config.depth,
            time=(time_ms or self.config.time_ms or 0) / 1000 if (time_ms or self.config.time_ms) else None,
        )

        root_moves = None
        if moves:
            root_moves = []
            for uci in moves:
                try:
                    root_moves.append(chess.Move.from_uci(uci))
                except ValueError:
                    logging.warning("Skipping illegal move provided to searchmoves: %s", uci)

        info = self.engine.analyse(
            board,
            limit=limit,
            multipv=pv_count,
            info=chess.engine.INFO_ALL,
            root_moves=root_moves,
        )

        if isinstance(info, dict):
            info = [info]

        evals: List[EngineEval] = []
        for entry in info:
            move = entry.get("pv", [None])[0]
            if move is None:
                continue
            score = entry["score"].pov(board.turn)
            cp = score.score(mate_score=None)
            mate = score.mate()
            wdl = None
            if entry.get("wdl"):
                wdl_obj = entry.get("wdl")
                wdl = [wdl_obj.wins, wdl_obj.draws, wdl_obj.losses]
            pv = [m.uci() for m in entry.get("pv", [])]
            evals.append(EngineEval(move=move.uci(), cp=cp, mate=mate, wdl=wdl, pv=pv))
        return evals

    def __del__(self):
        self.close()
