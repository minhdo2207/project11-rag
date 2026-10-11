#!/usr/bin/env python3
"""S7 runner — calibrate the confidence thresholds, then measure the decision.

WHAT MAKES THIS SCENARIO DIFFERENT
----------------------------------
S1–S6 are measured on questions the corpus can answer. S7 is measured mostly on
the ones it cannot. The unanswerable group is not a robustness check here — it
is the half that carries the result.

Three kinds of unanswerable question, reported separately because they fail for
different reasons:

    absent      the topic is not in the corpus at all — the easy case
    adjacent    the topic IS present but the specific fact asked for is not.
                The hard case: retrieval returns good, on-topic, well-written
                chunks, and the only thing standing between that and a confident
                invention is the threshold.
    oos         the answer exists but in a scope the filter excludes — the gap
                S4 explicitly left open

THE PARAMETER YOU SET
---------------------
`--cost-ratio`: how many unnecessary refusals one confident wrong answer is
worth. There is no default that is right for every system; pick it deliberately,
write it in the report, and every threshold downstream follows from it.

    python run_s7.py --demo --cost-ratio 5
    python run_s7.py --docs corpus.jsonl --eval eval.jsonl --cost-ratio 3

    eval.jsonl  {"query": "...", "answer_spans": [...]}            answerable
                {"query": "...", "answerable": false, "kind": "adjacent"}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from chunking import ChunkConfig, chunk_documents
from confidence import (ConfidenceConfig, calibrate, pick_threshold,
                        threshold_curve, tiers_from_calibration)
from context_selection import SelectionConfig
from embedding import Embedder, EmbeddingConfig
from metadata import as_of_filter
from rerank import LexicalOverlapReranker, RerankConfig, Reranker
from retriever import HybridRetriever, RetrieverConfig, tokenize


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def has_span(text: str, spans) -> bool:
    return any(norm(sp) in norm(text) for sp in (spans or []))


# --------------------------------------------------------------------------- #
# scoring passes
# --------------------------------------------------------------------------- #

def top_scores(r, queries, where) -> list[float]:
    """Top reranker score per query, with no thresholding applied — the raw
    material calibration works on."""
    out = []
    for q in queries:
        res = r.retrieve(q, top_k=6, rerank=True, where=where)
        if not res.hits:
            out.append(0.0)
            continue
        h = res.hits[0]
        out.append(float(h.rerank_score if h.rerank_score is not None else h.score))
    return out


def evaluate_decisions(r, answerable, unanswerable, cfg, where) -> dict:
    """Measure the four-way decision, not just answer-or-not."""
    actions_a, actions_u = [], []
    correct_when_answered, by_kind = [], {}

    for row in answerable:
        res = r.retrieve(row["query"], top_k=6, rerank=True, where=where, confidence=cfg)
        a = res.decision.action
        actions_a.append(a)
        if res.may_assert:
            correct_when_answered.append(
                1.0 if any(has_span(h.doc.text, row["answer_spans"]) for h in res.hits)
                else 0.0)

    for row in unanswerable:
        res = r.retrieve(row["query"], top_k=6, rerank=True, where=where, confidence=cfg)
        a = res.decision.action
        actions_u.append(a)
        k = row.get("kind", "absent")
        by_kind.setdefault(k, []).append(1.0 if res.may_assert else 0.0)

    def share(acts, name):
        return sum(1 for x in acts if x == name) / max(len(acts), 1)

    asserted_u = sum(1 for x in actions_u if x in ("answer", "answer_conditional"))
    refused_a = sum(1 for x in actions_a if x in ("evidence_only", "abstain"))

    return {
        # the two headline numbers
        "false_answer_rate": asserted_u / max(len(actions_u), 1),
        "over_refusal": refused_a / max(len(actions_a), 1),
        # usefulness: how often a real question gets a real answer
        "coverage": share(actions_a, "answer") + share(actions_a, "answer_conditional"),
        "precision_when_answered": (float(np.mean(correct_when_answered))
                                    if correct_when_answered else float("nan")),
        # tier mix on answerable questions
        "a_answer": share(actions_a, "answer"),
        "a_conditional": share(actions_a, "answer_conditional"),
        "a_evidence": share(actions_a, "evidence_only"),
        "a_abstain": share(actions_a, "abstain"),
        # tier mix on unanswerable questions
        "u_answer": share(actions_u, "answer"),
        "u_conditional": share(actions_u, "answer_conditional"),
        "u_evidence": share(actions_u, "evidence_only"),
        "u_abstain": share(actions_u, "abstain"),
        "_by_kind": {k: float(np.mean(v)) for k, v in by_kind.items()},
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_calibration(cal: dict) -> bool:
    print("\ncalibration — can the score tell answerable from unanswerable?")
    if not cal.get("usable"):
        print(f"  {cal.get('note')}")
        return False
    print(f"  answerable    median={cal['answerable_median']:.3f}  "
          f"p25={cal['answerable_p25']:.3f}  p10={cal['answerable_p10']:.3f}  "
          f"n={cal['n_answerable']}")
    print(f"  unanswerable  median={cal['unanswerable_median']:.3f}  "
          f"p90={cal['unanswerable_p90']:.3f}  n={cal['n_unanswerable']}")
    print(f"  separation    {cal['separation']:.3f}")

    if cal["separation"] < 0.3:
        print("\n  WARNING separation below 0.3: the score carries almost no")
        print("          information about answerability. No threshold will fix")
        print("          this — the reranker is the problem. Revisit S3 (this is")
        print("          exactly what its `overlap` diagnostic was measuring).")
        return False
    if cal["separation"] < 0.6:
        print("\n  Separation is moderate. A threshold will work but the middle")
        print("  tiers will carry a lot of traffic — which is what they are for.")
    return True


def print_curve(curve: list[dict], picked: float) -> None:
    print("\ntrade-off curve")
    print(f"  {'threshold':>10}{'false_answer':>14}{'over_refusal':>14}{'coverage':>11}")
    step = max(1, len(curve) // 10)
    for row in curve[::step]:
        mark = " <-" if abs(row["threshold"] - picked) < 1e-9 else ""
        print(f"  {row['threshold']:>10.3f}{row['false_answer_rate']:>14.3f}"
              f"{row['over_abstention']:>14.3f}{row['coverage']:>11.3f}{mark}")
    print("  There is no optimum on this curve without a cost ratio — only a")
    print("  choice. That is why cost_ratio is an input, not a default.")


def print_decisions(m: dict, cfg: ConfidenceConfig) -> None:
    print(f"\ntiers  t_high={cfg.t_high:.3f}  t_mid={cfg.t_mid:.3f}  "
          f"t_low={cfg.t_low:.3f}  min_margin={cfg.min_margin}")
    print(f"\n  {'':<16}{'answer':>9}{'conditional':>13}{'evidence':>10}{'abstain':>9}")
    print(f"  {'answerable':<16}{m['a_answer']:>9.2f}{m['a_conditional']:>13.2f}"
          f"{m['a_evidence']:>10.2f}{m['a_abstain']:>9.2f}")
    print(f"  {'unanswerable':<16}{m['u_answer']:>9.2f}{m['u_conditional']:>13.2f}"
          f"{m['u_evidence']:>10.2f}{m['u_abstain']:>9.2f}")

    print(f"\n  false_answer_rate        {m['false_answer_rate']:.3f}   "
          "(asserted a claim with no basis)")
    print(f"  over_refusal             {m['over_refusal']:.3f}   "
          "(answerable question not answered)")
    print(f"  coverage                 {m['coverage']:.3f}   "
          "(real questions that got a real answer)")
    print(f"  precision_when_answered  {m['precision_when_answered']:.3f}")

    print("\n  by kind of unanswerable question (share that still got asserted)")
    for k in ("absent", "adjacent", "oos"):
        if k in m["_by_kind"]:
            print(f"    {k:<10}{m['_by_kind'][k]:>6.2f}")
    if m["_by_kind"].get("adjacent", 0) > m["_by_kind"].get("absent", 0):
        print("    As expected: 'adjacent' is the hard class. The corpus covers the")
        print("    topic, so retrieval returns confident-looking chunks that simply")
        print("    do not contain the fact asked for.")


def print_verdict(m: dict, baseline: dict) -> bool:
    print("\nverdict")
    print(f"  without S7  false_answer={baseline['false_answer_rate']:.3f}  "
          f"coverage={baseline['coverage']:.3f}")
    print(f"  with S7     false_answer={m['false_answer_rate']:.3f}  "
          f"coverage={m['coverage']:.3f}")

    ok = True
    d_false = baseline["false_answer_rate"] - m["false_answer_rate"]
    d_cov = baseline["coverage"] - m["coverage"]
    print(f"\n  bought {d_false:.3f} fewer unfounded answers for {d_cov:.3f} coverage")

    if d_false <= 0:
        print("  WARNING S7 did not reduce unfounded answers. Check the")
        print("          calibration separation above before tuning thresholds.")
        ok = False

    if m["a_evidence"] > 0:
        print(f"\n  {m['a_evidence']:.0%} of answerable questions landed in")
        print("  evidence_only — they still show passages, so this is degraded")
        print("  service rather than refusal. For a user-facing system that")
        print("  distinction is the point of having three tiers instead of two.")
    return ok


# --------------------------------------------------------------------------- #

class _HashingEmbedder:
    dim = 512

    def _v(self, t):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(t, fold_diacritics=False):
            v[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts, **kw):
        return np.vstack([self._v(t) for t in texts])

    def encode_queries(self, texts, **kw):
        return np.vstack([self._v(t) for t in texts])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--eval", dest="eval_path", type=Path)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--cost-ratio", type=float, default=5.0,
                    help="cost of one confident wrong answer / one needless refusal")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.demo:
        from demo_corpus import DEMO_ANSWERABLE, DEMO_DOCS, DEMO_UNANSWERABLE
        docs, answerable, unanswerable = DEMO_DOCS, DEMO_ANSWERABLE, DEMO_UNANSWERABLE
        embedder, reranker = _HashingEmbedder(), LexicalOverlapReranker()
        print("demo mode: hashing embedder + stub reranker — read the decision")
        print("           mechanics and the curve shape, not absolute quality")
    else:
        if not (args.docs and args.eval_path):
            ap.error("--docs and --eval are required (or use --demo)")
        docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [json.loads(l) for l in args.eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        answerable = [r for r in rows if r.get("answerable", True)]
        unanswerable = [r for r in rows if not r.get("answerable", True)]
        embedder = Embedder(EmbeddingConfig(model_name=args.model, device=args.device))
        reranker = Reranker(RerankConfig(device=args.device))

    if not unanswerable:
        print("ERROR no unanswerable questions in the eval set.")
        print("      S7 cannot be measured without them — false_answer_rate is")
        print("      undefined, and the threshold would be fitted to one class.")
        print("      Aim for 20-30% of the set.")
        return 2

    chunks = []
    for d in docs:
        part = chunk_documents([d], ChunkConfig(), strategy="structure")
        for c in part:
            c.metadata.update(d.get("metadata") or {})
        chunks.extend(part)

    print(f"corpus {len(docs)} docs -> {len(chunks)} chunks")
    print(f"eval   {len(answerable)} answerable + {len(unanswerable)} unanswerable "
          f"({len(unanswerable)/(len(answerable)+len(unanswerable)):.0%})")

    scope = as_of_filter(audience="retail")
    cfg = RetrieverConfig(mode="hybrid", rerank_candidates=20)
    sel = SelectionConfig(use_adaptive_k=True, use_mmr=False, use_u_shape=False)
    r = HybridRetriever(embedder, cfg, reranker=reranker, selection=sel).index(
        [c.to_dict() for c in chunks])

    a_scores = top_scores(r, [x["query"] for x in answerable], scope)
    u_scores = top_scores(r, [x["query"] for x in unanswerable], scope)

    cal = calibrate(a_scores, u_scores)
    usable = print_calibration(cal)

    picked = pick_threshold(a_scores, u_scores, cost_ratio=args.cost_ratio)
    print(f"\nthreshold at cost_ratio={args.cost_ratio:g}")
    print(f"  t={picked['threshold']:.3f}  false_answer={picked['false_answer_rate']:.3f}  "
          f"over_refusal={picked['over_abstention']:.3f}  coverage={picked['coverage']:.3f}")

    print_curve(threshold_curve(a_scores, u_scores), picked["threshold"])

    conf = tiers_from_calibration(cal, picked["threshold"])
    m = evaluate_decisions(r, answerable, unanswerable, conf, scope)

    # baseline: always answer whatever retrieval returned
    always = ConfidenceConfig(t_high=-1.0, t_mid=-1.0, t_low=-1.0, use_margin=False)
    base = evaluate_decisions(r, answerable, unanswerable, always, scope)

    print_decisions(m, conf)
    ok = print_verdict(m, base)
    return 0 if (ok and usable) else 1


if __name__ == "__main__":
    raise SystemExit(main())
