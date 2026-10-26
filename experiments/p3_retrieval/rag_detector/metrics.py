"""Metrics computed from stored detector results.

Each question falls into one of three outcomes (N = all questions):
- rejected: the system declined because the retrieved context was insufficient
  (S7 abstain, or a refusal answer). The detector is not run on these.
- hallucinated: an answer was given and the detector flagged a span with confidence >= tau.
- correct: an answer was given, the detector did not flag it, and it matches the gold answer.
An unflagged answer that does not match gold is reported as ``incorrect_unflagged``.

Metrics (all in percent, as in Eq. 11 of the reference paper):
- accuracy             = correct / N
- hallucination_rate   = hallucinated / N
- rejection_rate       = rejected / N
- adjusted_accuracy    = correct / (correct + hallucinated)
"""

from __future__ import annotations


def _pct(num: float, den: float) -> float | None:
    return 100.0 * num / den if den else None


def metrics_at(results: list[dict], tau: float) -> dict:
    """Compute all metrics for a hallucination-flagging threshold ``tau``."""
    n = len(results)
    answered = [r for r in results if not r["refused"]]
    hallucinated = [r for r in answered if r["max_confidence"] >= tau]
    unflagged = [r for r in answered if r["max_confidence"] < tau]

    has_gold = all(r.get("correct") is not None for r in unflagged)
    correct = sum(1 for r in unflagged if r["correct"]) if has_gold else None

    return {
        "tau": tau,
        "n": n,
        "rejected": n - len(answered),
        "hallucinated": len(hallucinated),
        "correct": correct,
        "incorrect_unflagged": len(unflagged) - correct if has_gold else None,
        "accuracy": _pct(correct, n) if has_gold else None,
        "hallucination_rate": _pct(len(hallucinated), n),
        "rejection_rate": _pct(n - len(answered), n),
        "adjusted_accuracy": _pct(correct, correct + len(hallucinated)) if has_gold else None,
    }


def summarize(results: list[dict], tau: float, taus: list[float]) -> dict:
    """Metrics at the operating ``tau`` plus a sweep over ``taus``."""
    return {
        "operating_point": metrics_at(results, tau),
        "tau_sweep": [metrics_at(results, t) for t in sorted(set(taus))],
    }
