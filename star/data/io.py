"""JSONL helpers for chess verifier data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .schema import SampleRecord


def append_jsonl(path: str, records: Iterable[SampleRecord]) -> None:
    """Append SampleRecord entries to a JSONL file, creating directories as needed."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("a", encoding="utf-8") as f:
        for rec in records:
            payload = rec.to_dict() if hasattr(rec, "to_dict") else rec
            f.write(json.dumps(payload))
            f.write("\n")


def read_jsonl(path: str) -> List[SampleRecord]:
    """Load JSONL samples into SampleRecord objects."""
    records: List[SampleRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, SampleRecord):
                records.append(payload)
            else:
                records.append(SampleRecord.from_dict(payload))
    return records
