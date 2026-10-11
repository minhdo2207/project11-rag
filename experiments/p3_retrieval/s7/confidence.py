"""S7 — Confidence calibration and the decision to answer.

Everything before this scenario improves the evidence. S7 decides what to DO
when the evidence is weak, which is the only point in the retrieval layer that
can stop a confident wrong answer.

Why a binary refuse is the wrong design here
--------------------------------------------
Users come to a Q&A system to be given information. A system that refuses
whenever it is not certain is safe and useless. So the decision is graded:

    answer              strong evidence; proceed normally
    answer_conditional  usable evidence, stated with its limits and citations
    evidence_only       not enough to conclude, but here are the closest passages
    abstain             nothing worth showing

`evidence_only` is the tier that matters most for a user-facing system: it costs
nothing in hallucination terms, because no claim is asserted, while still giving
the user something to act on.

The parameter you must set yourself
-----------------------------------
`cost_ratio` = cost of one confident wrong answer ÷ cost of one unnecessary
refusal. There is no universal value. A billing or policy answer that is wrong
can cost a complaint and a lost customer; a refusal costs an extra click. If
that feels like 5:1, say 5. The threshold then follows from data instead of
taste, and `pick_threshold` reports what that choice buys and costs.

Calibration needs comparable scores
-----------------------------------
A threshold only transfers across queries if the score does. RRF scores do not —
they are flat and query-relative. This is why S7 depends on the reranker scores
from S3, and why S3's `overlap` diagnostic is the thing that decides whether S7
can work at all.

Dependency: numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "ConfidenceConfig", "Decision", "decide",
    "calibrate", "pick_threshold", "threshold_curve",
]


ACTIONS = ("answer", "answer_conditional", "evidence_only", "abstain")


@dataclass
class ConfidenceConfig:
    # Thresholds on the top reranker score. Set these from calibrate(), not by
    # hand — hand-set values are the most common reason S7 behaves erratically.
    t_high: float = 0.80           # >= t_high            -> answer
    t_mid: float = 0.50            # [t_mid, t_high)      -> answer_conditional
    t_low: float = 0.25            # [t_low, t_mid)       -> evidence_only
                                   # < t_low              -> abstain

    # A second, independent signal: how far ahead the best chunk is. A flat top
    # of the ranking means the retriever found several equally plausible
    # candidates and cannot tell them apart — which is exactly the state that
    # produces a confident wrong answer.
    use_margin: bool = True
    min_margin: float = 0.05       # top1 - top2 below this downgrades one tier

    evidence_only_k: int = 3       # how many passages to show in that tier


@dataclass
class Decision:
    action: str
    confidence: float              # the top score the decision was made on
    margin: float
    tier_reason: str
    hits: list = field(default_factory=list)
    message: str = ""

    @property
    def may_assert(self) -> bool:
        """True only when the generation layer is allowed to state a claim.
        `evidence_only` and `abstain` both return False."""
        return self.action in ("answer", "answer_conditional")


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #

def decide(hits: Sequence[Any], cfg: ConfidenceConfig,
           empty_reason: str = "") -> Decision:
    """Map a ranked hit list onto one of four actions."""
    if not hits:
        return Decision("abstain", 0.0, 0.0, "no candidates",
                        message=empty_reason or "Không tìm thấy tài liệu phù hợp.")

    scores = [h.rerank_score if getattr(h, "rerank_score", None) is not None
              else h.score for h in hits]
    top = float(scores[0])
    margin = float(top - scores[1]) if len(scores) > 1 else float(top)

    if top >= cfg.t_high:
        action, reason = "answer", "score >= t_high"
    elif top >= cfg.t_mid:
        action, reason = "answer_conditional", "t_mid <= score < t_high"
    elif top >= cfg.t_low:
        action, reason = "evidence_only", "t_low <= score < t_mid"
    else:
        action, reason = "abstain", "score < t_low"

    # Downgrade on a flat top: several candidates scoring the same means the
    # retriever could not discriminate, no matter how high the absolute score.
    if cfg.use_margin and margin < cfg.min_margin and action != "abstain":
        idx = ACTIONS.index(action)
        action = ACTIONS[min(idx + 1, len(ACTIONS) - 1)]
        reason += f" | downgraded: margin {margin:.3f} < {cfg.min_margin}"

    if action == "evidence_only":
        hits = list(hits)[: cfg.evidence_only_k]
        msg = ("Chưa đủ cơ sở để kết luận. Dưới đây là các đoạn tài liệu "
               "liên quan nhất để bạn tham khảo.")
    elif action == "answer_conditional":
        msg = "Thông tin dưới đây dựa trên tài liệu tìm được, vui lòng đối chiếu nguồn."
    elif action == "abstain":
        hits, msg = [], "Không tìm thấy thông tin đủ tin cậy cho câu hỏi này."
    else:
        msg = ""

    return Decision(action, top, margin, reason, list(hits), msg)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def calibrate(answerable_scores: Sequence[float],
              unanswerable_scores: Sequence[float]) -> dict:
    """Describe the two score distributions.

    `separation` is the headline: the share of unanswerable questions scoring
    below the 10th percentile of answerable ones. Near 1.0 means a threshold can
    cleanly divide them. Near 0 means the score carries no information about
    answerability and no threshold will help — fix S3 first.
    """
    a = np.asarray([s for s in answerable_scores if s is not None], dtype=float)
    u = np.asarray([s for s in unanswerable_scores if s is not None], dtype=float)
    if a.size == 0 or u.size == 0:
        return {"usable": False,
                "note": "need both answerable and unanswerable questions"}

    a10 = float(np.percentile(a, 10))
    return {
        "usable": True,
        "answerable_median": float(np.median(a)),
        "answerable_p10": a10,
        "answerable_p25": float(np.percentile(a, 25)),
        "unanswerable_median": float(np.median(u)),
        "unanswerable_p90": float(np.percentile(u, 90)),
        "separation": float(np.mean(u < a10)),
        "n_answerable": int(a.size),
        "n_unanswerable": int(u.size),
    }


def threshold_curve(answerable_scores: Sequence[float],
                    unanswerable_scores: Sequence[float],
                    n_points: int = 25) -> list[dict]:
    """The full trade-off, one row per candidate threshold.

    There is no optimum here without a cost ratio — only a curve. Reporting the
    curve rather than a single number is the point.
    """
    a = np.asarray([s for s in answerable_scores if s is not None], dtype=float)
    u = np.asarray([s for s in unanswerable_scores if s is not None], dtype=float)
    if a.size == 0 or u.size == 0:
        return []

    lo, hi = float(min(a.min(), u.min())), float(max(a.max(), u.max()))
    out = []
    for t in np.linspace(lo, hi, n_points):
        out.append({
            "threshold": float(t),
            # unanswerable questions that still get answered
            "false_answer_rate": float(np.mean(u >= t)),
            # answerable questions wrongly refused
            "over_abstention": float(np.mean(a < t)),
            "coverage": float(np.mean(a >= t)),
        })
    return out


def pick_threshold(answerable_scores: Sequence[float],
                   unanswerable_scores: Sequence[float],
                   cost_ratio: float = 5.0) -> dict:
    """Choose the threshold that minimises expected cost.

    cost = cost_ratio * false_answer_rate + 1.0 * over_abstention

    cost_ratio is the one judgement call: how many unnecessary refusals a single
    confident wrong answer is worth. Pick it deliberately and write it down —
    every threshold downstream inherits it.
    """
    curve = threshold_curve(answerable_scores, unanswerable_scores, n_points=101)
    if not curve:
        return {"usable": False}

    for row in curve:
        row["expected_cost"] = (cost_ratio * row["false_answer_rate"]
                                + row["over_abstention"])
    best = min(curve, key=lambda r: r["expected_cost"])
    return {
        "usable": True,
        "cost_ratio": cost_ratio,
        "threshold": best["threshold"],
        "false_answer_rate": best["false_answer_rate"],
        "over_abstention": best["over_abstention"],
        "coverage": best["coverage"],
        "expected_cost": best["expected_cost"],
    }


def tiers_from_calibration(cal: dict, picked: float) -> ConfidenceConfig:
    """Derive the three tier boundaries from the calibration.

    The cost-optimal threshold is the line for ASSERTING a claim, not the line
    for showing anything — so it becomes `t_mid`, not `t_low`:

        t_high  median of the answerable distribution; a confident answer is
                reserved for scores typical of questions the corpus answers
        t_mid   the cost-optimal threshold: below it, assert nothing
        t_low   the unanswerable median; between t_low and t_mid the passages
                are still worth showing without a conclusion

    Setting t_low equal to t_mid collapses the evidence_only band to nothing and
    turns the whole thing back into a binary refuse — which is the design S7
    exists to avoid.
    """
    t_mid = picked
    t_high = max(t_mid, cal.get("answerable_median", t_mid))
    # keep a real band below t_mid; fall back to a fraction of it when the
    # unanswerable distribution sits above t_mid (a badly separated score)
    t_low = cal.get("unanswerable_median", t_mid * 0.6)
    if t_low >= t_mid:
        t_low = t_mid * 0.6
    return ConfidenceConfig(t_low=float(t_low), t_mid=float(t_mid), t_high=float(t_high))
