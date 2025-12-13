"""Data schemas for iterative chess verifier pipelines."""
from __future__ import annotations

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class EngineMetadata:
    name: str = "stockfish"
    depth: int = 18
    multipv: int = 1
    time_ms: int = 0


@dataclasses.dataclass
class EngineEvalRecord:
    move: str
    cp: Optional[int]
    mate: Optional[int]
    wdl: Optional[List[int]]
    pv: List[str]


@dataclasses.dataclass
class DeltaRecord:
    cp: int
    wdl_win_prob: Optional[float]


@dataclasses.dataclass
class ContrastRecord:
    alt_move: str
    refutation_pv: List[str]
    note: str = ""


@dataclasses.dataclass
class RationaleRecord:
    claim: str
    evidence_pv: List[str]
    contrast: ContrastRecord
    verified: bool
    error_type: str


@dataclasses.dataclass
class SampleRecord:
    id: str
    fen: str
    side_to_move: str
    candidates: List[str]
    engine: EngineMetadata
    evals: List[EngineEvalRecord]
    chosen: str
    rejected: str
    delta: DeltaRecord
    rationale: RationaleRecord

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "SampleRecord":
        return cls(
            id=payload["id"],
            fen=payload["fen"],
            side_to_move=payload["side_to_move"],
            candidates=list(payload.get("candidates", [])),
            engine=EngineMetadata(**payload["engine"]),
            evals=[EngineEvalRecord(**ev) for ev in payload.get("evals", [])],
            chosen=payload["chosen"],
            rejected=payload["rejected"],
            delta=DeltaRecord(**payload["delta"]),
            rationale=RationaleRecord(
                claim=payload["rationale"]["claim"],
                evidence_pv=list(payload["rationale"].get("evidence_pv", [])),
                contrast=ContrastRecord(**payload["rationale"]["contrast"]),
                verified=payload["rationale"].get("verified", False),
                error_type=payload["rationale"].get("error_type", ""),
            ),
        )
