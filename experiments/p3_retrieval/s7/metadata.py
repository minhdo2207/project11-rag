"""S4 — Scope and version metadata.

The hard negative no retriever can see
--------------------------------------
A superseded policy chunk and its current replacement are near-identical: same
topic, same vocabulary, same structure, often the same sentences with one number
changed. Dense similarity cannot tell them apart — they are *supposed* to look
alike. A cross-encoder cannot either; it has been observed actively promoting
the topically-right, version-wrong chunk.

The only thing that separates them is metadata that was never in the text:
effective dates, status, audience, product line. So S4 is not a ranking
improvement — it is a filter applied BEFORE anything is scored.

Why "before" is the whole point
-------------------------------
Post-filtering (score first, drop non-matching results afterwards) silently
shrinks top-k: ask for 5, get 2, and the model receives thinner evidence than it
was configured to. Pre-filtering restricts the candidate pool, so top-k is still
top-k — drawn from the right universe.

This module holds: the metadata contract, cheap extractors for documents that
carry their own effective dates, and a filter compiler.

Dependency: stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "FIELDS", "FilterError", "compile_filter", "extract_metadata",
    "validate_metadata", "as_of_filter", "coverage_report",
]


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #

FIELDS = {
    "source":         "file or document the chunk came from",
    "doc_type":       "policy | faq | tariff | procedure | announcement",
    "version":        "document version label, e.g. 'v3' or '2026.01'",
    "status":         "current | superseded | draft  — the single most useful field",
    "effective_from": "ISO date the content takes effect",
    "effective_to":   "ISO date it stops applying; None = still in force",
    "audience":       "retail | enterprise | internal  — who the rule applies to",
    "region":         "optional geographic scope",
}


class FilterError(ValueError):
    """Raised for a malformed filter spec. Loud on purpose: a filter that is
    silently ignored is worse than no filter, because it looks like it worked."""


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

_DATE_PATTERNS = [
    re.compile(r"hiệu\s*lực\s*(?:từ|kể từ)\s*(?:ngày\s*)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})", re.I),
    re.compile(r"áp\s*dụng\s*từ\s*(?:ngày\s*)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})", re.I),
    re.compile(r"effective\s*(?:from\s*)?(\d{4})-(\d{2})-(\d{2})", re.I),
]
_END_PATTERNS = [
    re.compile(r"(?:hết\s*hiệu\s*lực|áp\s*dụng\s*đến)\s*(?:ngày\s*)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})", re.I),
    re.compile(r"thay\s*thế\s*bởi", re.I),
]
_SUPERSEDED = re.compile(
    r"\b(đã\s*hết\s*hiệu\s*lực|được\s*thay\s*thế|bản\s*cũ|ngừng\s*(?:áp\s*dụng|cung\s*cấp|bán)"
    r"|superseded|deprecated)\b", re.I)
_VERSION = re.compile(r"\b(?:phiên\s*bản|version|v)\s*[:.]?\s*([\w.]+)", re.I)


def _norm_date(a: str, b: str, c: str) -> str | None:
    """Accepts dd/mm/yyyy or yyyy-mm-dd fragments; returns ISO or None."""
    try:
        if len(a) == 4:
            y, m, d = int(a), int(b), int(c)
        else:
            d, m, y = int(a), int(b), int(c)
        return date(y, m, d).isoformat()
    except (ValueError, TypeError):
        return None


def extract_metadata(text: str, base: dict | None = None) -> dict:
    """Best-effort extraction from the document body.

    Deliberately conservative: it only fills a field when the document states it
    outright. Guessing an effective date is worse than leaving it empty, because
    a wrong date filters out the right answer.
    """
    meta = dict(base or {})

    for rx in _DATE_PATTERNS:
        m = rx.search(text)
        if m:
            iso = _norm_date(*m.groups())
            if iso:
                meta.setdefault("effective_from", iso)
            break

    for rx in _END_PATTERNS:
        m = rx.search(text)
        if m and m.groups():
            iso = _norm_date(*m.groups())
            if iso:
                meta.setdefault("effective_to", iso)
            break

    m = _VERSION.search(text)
    if m:
        meta.setdefault("version", m.group(1))

    if "status" not in meta:
        meta["status"] = "superseded" if _SUPERSEDED.search(text) else "current"

    return meta


def validate_metadata(chunks: Sequence[Any], required=("status",)) -> dict:
    """Metadata you did not write is metadata you cannot filter on. Reports
    coverage per field so gaps are visible before they cost you a filter."""
    n = len(chunks)
    cov = {f: 0 for f in FIELDS}
    for c in chunks:
        meta = getattr(c, "metadata", None) or {}
        for f in FIELDS:
            if meta.get(f) not in (None, ""):
                cov[f] += 1
    report = {f: cov[f] / n if n else 0.0 for f in FIELDS}
    report["_missing_required"] = [f for f in required if report.get(f, 0.0) < 1.0]
    return report


# --------------------------------------------------------------------------- #
# filter compilation
# --------------------------------------------------------------------------- #

_OPS = {"eq", "ne", "in", "nin", "lte", "gte", "lt", "gt", "exists"}


def compile_filter(where: dict | Callable | None) -> Callable[[dict], bool] | None:
    """Turn a filter spec into a predicate over a chunk's metadata.

        {"status": "current"}                      equality
        {"doc_type": ["faq", "policy"]}            membership
        {"effective_from": {"lte": "2026-10-07"}}  comparison
        {"effective_to": {"exists": False}}        still in force

    Several keys are ANDed. Unknown operators raise rather than being ignored:
    a filter that silently does nothing is the worst outcome here, because the
    system looks filtered and is not.
    """
    if where is None:
        return None
    if callable(where):
        return lambda meta: bool(where(meta))

    if not isinstance(where, dict):
        raise FilterError(f"filter must be a dict or callable, got {type(where).__name__}")

    checks: list[Callable[[dict], bool]] = []
    for key, cond in where.items():
        if isinstance(cond, dict):
            for op, val in cond.items():
                if op not in _OPS:
                    raise FilterError(f"unknown operator {op!r} on field {key!r}; "
                                      f"valid: {sorted(_OPS)}")
                checks.append(_make_check(key, op, val))
        elif isinstance(cond, (list, tuple, set)):
            checks.append(_make_check(key, "in", set(cond)))
        else:
            checks.append(_make_check(key, "eq", cond))

    def predicate(meta: dict) -> bool:
        return all(chk(meta) for chk in checks)

    return predicate


def _make_check(key: str, op: str, val: Any) -> Callable[[dict], bool]:
    def chk(meta: dict) -> bool:
        got = meta.get(key)
        if op == "exists":
            return (got not in (None, "")) is bool(val)
        if got in (None, ""):
            # A chunk with no value for the field cannot be proven to match.
            # Excluding it is the conservative choice: better to miss a chunk
            # than to serve one whose scope is unknown.
            return False
        if op == "eq":
            return got == val
        if op == "ne":
            return got != val
        if op == "in":
            return got in val
        if op == "nin":
            return got not in val
        if op == "lte":
            return str(got) <= str(val)
        if op == "gte":
            return str(got) >= str(val)
        if op == "lt":
            return str(got) < str(val)
        if op == "gt":
            return str(got) > str(val)
        raise FilterError(f"unhandled operator {op!r}")
    return chk


def as_of_filter(as_of: str | date | None = None, *, audience: str | None = None,
                 include_superseded: bool = False) -> dict:
    """The filter most Q&A systems actually need: what is in force today.

        as_of_filter()                         -> current rules, today
        as_of_filter("2026-01-15")             -> what applied on that date
        as_of_filter(audience="enterprise")    -> plus audience scoping
    """
    if as_of is None:
        as_of = date.today()
    if isinstance(as_of, date):
        as_of = as_of.isoformat()

    where: dict[str, Any] = {"effective_from": {"lte": as_of}}
    if not include_superseded:
        where["status"] = "current"
    if audience:
        where["audience"] = audience
    return where


def coverage_report(chunks: Sequence[Any], where: dict | Callable | None) -> dict:
    """How much of the corpus survives a filter. A filter that keeps 2% of the
    index will mostly produce abstentions — worth knowing before you ship it."""
    pred = compile_filter(where)
    if pred is None:
        return {"kept": len(chunks), "total": len(chunks), "ratio": 1.0}
    kept = sum(1 for c in chunks if pred(getattr(c, "metadata", None) or {}))
    return {"kept": kept, "total": len(chunks),
            "ratio": kept / len(chunks) if chunks else 0.0}
