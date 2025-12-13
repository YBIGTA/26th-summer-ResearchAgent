import dataclasses
import logging
from typing import Dict, List, Optional

import chess

ERROR_NONE = "NONE"
ERROR_ILLEGAL = "ILLEGAL"
ERROR_TACTICAL = "TACTICAL_BLUNDER"
ERROR_STRATEGIC = "STRATEGIC"
ERROR_MISMATCH = "RATIONALE_MISMATCH"
ERROR_STATE = "STATE_TRACKING"


@dataclasses.dataclass
class VerificationResult:
    verified: bool
    error_type: str = ERROR_NONE
    refutation_pv: Optional[List[str]] = None


RATIONAL_TEMPLATE_KEYS = [
    "MOVE",
    "CLAIM",
    "EVIDENCE_PV",
    "CONTRAST_MOVE",
    "CONTRAST_PV",
]


def parse_rationale_text(text: str) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in RATIONAL_TEMPLATE_KEYS:
            payload[key] = value
    return payload


def is_pv_legal(fen: str, pv: List[str]) -> bool:
    board = chess.Board(fen)
    try:
        for move in pv:
            board.push_uci(move)
    except ValueError:
        return False
    return True


def _win_prob_from_wdl(wdl: Optional[List[int]]) -> Optional[float]:
    if not wdl:
        return None
    total = sum(wdl)
    if total == 0:
        return None
    return wdl[0] / total


def verify_rationale(
    fen: str,
    evidence_pv: List[str],
    contrast_pv: List[str],
    wdl_best: Optional[List[int]] = None,
    wdl_rejected: Optional[List[int]] = None,
    delta_cp: Optional[int] = None,
    blunder_threshold: int = 180,
) -> VerificationResult:
    if not is_pv_legal(fen, evidence_pv) or not is_pv_legal(fen, contrast_pv):
        return VerificationResult(verified=False, error_type=ERROR_ILLEGAL)

    win_best = _win_prob_from_wdl(wdl_best)
    win_rej = _win_prob_from_wdl(wdl_rejected)
    if delta_cp is not None and abs(delta_cp) >= blunder_threshold:
        return VerificationResult(
            verified=True, error_type=ERROR_TACTICAL, refutation_pv=contrast_pv
        )

    if win_best is not None and win_rej is not None:
        if win_best - win_rej < 0.01:
            return VerificationResult(verified=True, error_type=ERROR_NONE)
    return VerificationResult(verified=True, error_type=ERROR_NONE)


def parse_and_verify(
    fen: str,
    rationale_text: str,
    wdl_best: Optional[List[int]] = None,
    wdl_rejected: Optional[List[int]] = None,
    delta_cp: Optional[int] = None,
    blunder_threshold: int = 180,
) -> Dict:
    payload = parse_rationale_text(rationale_text)
    missing = [k for k in RATIONAL_TEMPLATE_KEYS if k not in payload]
    if missing:
        logging.warning("Missing fields in rationale text: %s", ",".join(missing))
    evidence_pv = payload.get("EVIDENCE_PV", "").split()
    contrast_pv = payload.get("CONTRAST_PV", "").split()
    verification = verify_rationale(
        fen,
        evidence_pv,
        contrast_pv,
        wdl_best=wdl_best,
        wdl_rejected=wdl_rejected,
        delta_cp=delta_cp,
        blunder_threshold=blunder_threshold,
    )
    return {
        "claim": payload.get("CLAIM", ""),
        "evidence_pv": evidence_pv,
        "contrast_move": payload.get("CONTRAST_MOVE", ""),
        "contrast_pv": contrast_pv,
        "verification": verification,
    }
