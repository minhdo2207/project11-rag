"""S6 — Context selection.

S1–S5 decide WHICH chunks are available. S6 decides which of them actually
reach the model, and it is the group with the most direct effect on
hallucination — for a reason that is easy to get backwards.

The finding this scenario is built around
-----------------------------------------
Documents that are topically related but do NOT contain the answer are far more
damaging than documents that are simply irrelevant. A single such distractor has
been measured dropping answer accuracy by up to 25%, and the degradation
compounds as more are added, exceeding 67% in some settings. Meanwhile adding
genuinely *random* passages can improve accuracy.

So the quantity to minimise is not "irrelevant content" in general — it is the
NUMBER OF PLAUSIBLE-BUT-WRONG chunks in the final context. Every chunk added to
a prompt is a liability, not an asset. "Fewer and strongly relevant" beats "more
with some distractors".

Three mechanisms:

  adaptive_k   stop at the score cliff instead of padding to a fixed k. Directly
               reduces the distractor count, which is the whole point.
  mmr          drop near-duplicate evidence. Five paraphrases of one fact read
               to a model like five independent confirmations.
  u_shaped     put the strongest evidence at both ends of the context.

A warning about the third, which the runner checks for you: reordering gives
negligible improvement on small retrieved sets and only pays off when the set is
large. Since adaptive_k exists to make the set small, these two mechanisms work
against each other. Enable ordering only if your effective k stays high.

Dependency: numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "SelectionConfig", "SelectionReport",
    "adaptive_k", "mmr_select", "u_shaped", "select_context", "redundancy",
]


@dataclass
class SelectionConfig:
    k_min: int = 1
    k_max: int = 6

    use_adaptive_k: bool = True
    gap_ratio: float = 0.55        # cut where score drops below ratio * previous
    score_floor: float = 0.0       # absolute floor; chunks below never qualify

    use_mmr: bool = True
    mmr_lambda: float = 0.7        # 1.0 = pure relevance, 0.0 = pure diversity

    use_u_shape: bool = False      # off by default — see the module docstring
    u_shape_min_k: int = 6         # below this, ordering is not worth doing


@dataclass
class SelectionReport:
    k_selected: int
    k_before: int
    cut_reason: str                # "gap" | "floor" | "k_max" | "k_min"
    mmr_dropped: int
    reordered: bool
    redundancy_before: float
    redundancy_after: float


# --------------------------------------------------------------------------- #
# mechanisms
# --------------------------------------------------------------------------- #

def adaptive_k(scores: Sequence[float], *, k_min: int = 1, k_max: int = 6,
               gap_ratio: float = 0.55, score_floor: float = 0.0) -> tuple[int, str]:
    """Return (k, reason).

    Two independent cuts:
      * relative — the first position whose score falls below gap_ratio of the
        previous one. Catches the cliff after the genuinely relevant chunks.
      * absolute — score_floor. Catches the case where everything is weak and
        the cliff never appears, which is exactly when padding does most harm.

    Requires calibrated scores. On raw RRF scores, which are flat and not
    comparable across queries, the relative cut fires almost at random — this is
    why S6 depends on S3.
    """
    if not scores:
        return 0, "empty"

    n = min(len(scores), k_max)
    for i in range(n):
        if scores[i] < score_floor:
            return max(k_min, i), "floor"
        if i > 0 and scores[i - 1] > 0 and scores[i] < gap_ratio * scores[i - 1]:
            return max(k_min, i), "gap"
    return max(k_min, n), "k_max"


def mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int,
               lambda_mult: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance: λ·sim(q,d) − (1−λ)·max sim(d, already chosen).

    Greedy, order-preserving in intent: the first pick is always the most
    relevant chunk, so diversification never costs the top result.
    """
    n = len(doc_vecs)
    if n == 0:
        return []
    k = min(k, n)
    sim_q = doc_vecs @ query_vec.reshape(-1)
    selected = [int(np.argmax(sim_q))]
    while len(selected) < k:
        redundancy_term = (doc_vecs @ doc_vecs[selected].T).max(axis=1)
        mmr = lambda_mult * sim_q - (1 - lambda_mult) * redundancy_term
        mmr[selected] = -np.inf
        nxt = int(np.argmax(mmr))
        if not np.isfinite(mmr[nxt]):
            break
        selected.append(nxt)
    return selected


def u_shaped(items: list) -> list:
    """Strongest items at both ends, weakest in the middle.

    [1,2,3,4,5] -> [1,3,5,4,2]

    Exploits the position bias rather than absorbing it. Costs nothing — it is
    only the order in which chunks are concatenated. But it depends entirely on
    the ranking being right: if rank 1 is not actually the best chunk, this
    places a wrong answer in the most attended position.
    """
    front, back = [], []
    for i, it in enumerate(items):
        (front if i % 2 == 0 else back).append(it)
    return front + back[::-1]


def redundancy(doc_vecs: np.ndarray) -> float:
    """Mean pairwise cosine among selected chunks. High values mean the context
    is saying the same thing several times — the false-corroboration shape."""
    n = len(doc_vecs)
    if n < 2:
        return 0.0
    sim = doc_vecs @ doc_vecs.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sim[iu]))


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #

def select_context(hits: Sequence[Any], scores: Sequence[float],
                   query_vec: np.ndarray | None, doc_vecs: np.ndarray | None,
                   cfg: SelectionConfig) -> tuple[list[Any], SelectionReport]:
    """Apply adaptive-k, then MMR, then ordering. Returns the chosen hits in
    the order they should be concatenated, plus a report for diagnostics."""
    k_before = len(hits)
    if k_before == 0:
        return [], SelectionReport(0, 0, "empty", 0, False, 0.0, 0.0)

    if cfg.use_adaptive_k:
        k, reason = adaptive_k(scores, k_min=cfg.k_min, k_max=cfg.k_max,
                               gap_ratio=cfg.gap_ratio, score_floor=cfg.score_floor)
    else:
        k, reason = min(cfg.k_max, k_before), "k_max"

    pool = list(hits[: max(k, cfg.k_max)])
    pool_vecs = doc_vecs[: len(pool)] if doc_vecs is not None else None
    red_before = redundancy(pool_vecs[:k]) if pool_vecs is not None else 0.0

    dropped = 0
    if cfg.use_mmr and pool_vecs is not None and query_vec is not None and len(pool) > k:
        keep = mmr_select(query_vec, pool_vecs, k, cfg.mmr_lambda)
        dropped = sum(1 for i in range(k) if i not in keep)
        chosen = [pool[i] for i in keep]
        chosen_vecs = pool_vecs[keep]
    else:
        chosen = pool[:k]
        chosen_vecs = pool_vecs[:k] if pool_vecs is not None else None

    red_after = redundancy(chosen_vecs) if chosen_vecs is not None else 0.0

    reordered = False
    if cfg.use_u_shape and len(chosen) >= cfg.u_shape_min_k:
        chosen = u_shaped(chosen)
        reordered = True

    return chosen, SelectionReport(
        k_selected=len(chosen), k_before=k_before, cut_reason=reason,
        mmr_dropped=dropped, reordered=reordered,
        redundancy_before=red_before, redundancy_after=red_after,
    )
