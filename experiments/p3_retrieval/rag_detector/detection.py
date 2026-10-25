"""LettuceDetect detection step: load records, run the detector, store per-question spans."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Use the LettuceDetect package bundled next to this folder (no pip install of it needed).
_BUNDLE_ROOT = Path(__file__).resolve().parent.parent
if (_BUNDLE_ROOT / "lettucedetect").is_dir() and str(_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_ROOT))

# Decisions after which the system did not produce an answer to check.
REFUSAL_DECISIONS = {"abstain", "refuse", "refused", "reject", "no_answer"}
REFUSAL_PATTERN = re.compile(
    r"(không đủ thông tin|không tìm thấy thông tin|không thể trả lời|"
    r"i (?:do not|don't) have enough information|cannot answer)",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write dicts to a JSONL file, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_refusal(record: dict) -> bool:
    """Whether the system declined to answer (context insufficient), so there is nothing to check."""
    if record.get("refused") is not None:
        return bool(record["refused"])
    if str(record.get("decision", "")).lower() in REFUSAL_DECISIONS:
        return True
    answer = (record.get("answer") or "").strip()
    return not answer or bool(REFUSAL_PATTERN.search(answer))


def build_detector(method: str, model: str, device: str | None, max_length: int, lang: str):
    """Instantiate the LettuceDetect detector (once per run, shared by all scenarios)."""
    from lettucedetect.models.inference import HallucinationDetector

    if method == "llm":
        return HallucinationDetector(method="llm", model=model, lang=lang)
    kwargs = {"model_path": model, "max_length": max_length, "lang": lang}
    if device:
        kwargs["device"] = device
    return HallucinationDetector(method="transformer", **kwargs)


def detect_records(detector, records: list[dict], done: dict[str, dict] | None = None) -> list[dict]:
    """Run the detector on every record, reusing results already in ``done``.

    ``min_confidence`` is left at 0 so the tau sweep can be computed offline from the stored
    per-record ``max_confidence``.
    """
    done = done or {}
    results = []
    for i, rec in enumerate(records):
        rid = str(rec.get("id", i))
        if rid in done:
            results.append(done[rid])
            continue
        out = {
            "id": rid,
            "refused": is_refusal(rec),
            "correct": rec.get("correct"),
            "n_contexts": len(rec.get("contexts", [])),
            "spans": [],
            "max_confidence": 0.0,
        }
        if not out["refused"]:
            spans = detector.predict(
                context=rec["contexts"],
                question=rec.get("question"),
                answer=rec["answer"],
                output_format="spans",
            )
            # LLM-based spans carry no confidence unless reasoning is enabled.
            for s in spans:
                s.setdefault("confidence", 1.0)
            out["spans"] = spans
            out["max_confidence"] = max((s["confidence"] for s in spans), default=0.0)
        results.append(out)
    return results
