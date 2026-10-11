#!/usr/bin/env python3
"""S6 ablation runner — context selection.

THE QUANTITY THIS SCENARIO MINIMISES
------------------------------------
Not "irrelevant content" — `avg_distractors`, the mean NUMBER of on-topic but
answer-free chunks in the final context.

That specific count is what the evidence points at: a single such distractor has
been measured dropping answer accuracy by up to 25%, with the damage compounding
as more are added (past 67% in some settings), while genuinely random passages
can actually help. Topical near-misses are the dangerous ones, and they are
exactly what a good retriever surfaces.

So every chunk in the prompt is a liability. recall@k going up while
avg_distractors goes up too is not an improvement.

THE INTERACTION THE RUNNER CHECKS FOR YOU
-----------------------------------------
Reordering helps little on small retrieved sets and only pays off when the set
is large. adaptive_k exists to make the set small. The two mechanisms therefore
cancel, and the runner reports the effective k so you can see whether ordering
is worth enabling at all.

    python run_s6.py --demo
    python run_s6.py --docs corpus.jsonl --eval eval.jsonl --sweep
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from chunking import ChunkConfig, chunk_documents
from context_selection import SelectionConfig
from embedding import Embedder, EmbeddingConfig
from rerank import LexicalOverlapReranker, RerankConfig, Reranker
from retriever import HybridRetriever, RetrieverConfig, tokenize


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def has_span(text: str, spans) -> bool:
    return any(norm(sp) in norm(text) for sp in (spans or []))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def evaluate(r, dataset, selection) -> dict:
    recall, k_used, distractors, red_before, red_after = [], [], [], [], []
    gold_pos, cut_reasons, reordered = [], {}, []

    for row in dataset:
        res = r.retrieve(row["query"], top_k=6, rerank=True, selection=selection)
        flags = [has_span(h.doc.text, row["answer_spans"]) for h in res.hits]

        recall.append(1.0 if any(flags) else 0.0)
        k_used.append(len(res.hits))
        # the number that matters: on-topic chunks carrying no answer
        distractors.append(sum(1 for f in flags if not f))

        # Distance from the NEAREST EDGE, not raw position. U-shaped ordering
        # pushes strong evidence to both ends, so raw position rises by design
        # and would read as a regression. Edge distance is what the mechanism
        # actually targets: lower = the answer sits where the model attends.
        pos = next((i for i, f in enumerate(flags) if f), None)
        if pos is not None and res.hits:
            n = len(res.hits)
            gold_pos.append(min(pos, n - 1 - pos))

        sel = res.diagnostics.get("selection")
        if sel:
            cut_reasons[sel["cut_reason"]] = cut_reasons.get(sel["cut_reason"], 0) + 1
            red_before.append(sel["redundancy_before"])
            red_after.append(sel["redundancy_after"])
            reordered.append(1.0 if sel["reordered"] else 0.0)

    return {
        "recall": float(np.mean(recall)),
        "avg_k": float(np.mean(k_used)),
        "avg_distractors": float(np.mean(distractors)),
        "redundancy": float(np.mean(red_after)) if red_after else float("nan"),
        "redundancy_drop": (float(np.mean(red_before) - np.mean(red_after))
                            if red_after else float("nan")),
        "gold_pos": float(np.mean(gold_pos)) if gold_pos else float("nan"),
        "reorder_rate": float(np.mean(reordered)) if reordered else 0.0,
        "_cut_reasons": cut_reasons,
    }


VARIANTS = [
    ("V0 fixed k=6",    SelectionConfig(use_adaptive_k=False, use_mmr=False,
                                        use_u_shape=False, k_max=6)),
    ("V1 adaptive-k",   SelectionConfig(use_adaptive_k=True, use_mmr=False,
                                        use_u_shape=False, k_max=6)),
    ("V2 +MMR",         SelectionConfig(use_adaptive_k=True, use_mmr=True,
                                        use_u_shape=False, k_max=6)),
    ("V3 +U-shape",     SelectionConfig(use_adaptive_k=True, use_mmr=True,
                                        use_u_shape=True, k_max=6)),
]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_table(rows) -> None:
    cols = [("recall", "recall", 9), ("avg_k", "avg_k", 8),
            ("avg_distractors", "distract", 10), ("redundancy", "redund", 9),
            ("gold_pos", "edge_dist", 11), ("reorder_rate", "reorder", 9)]
    header = f"{'variant':<18}" + "".join(f"{h:>{w}}" for _, h, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for label, m in rows:
        print(f"{label:<18}" + "".join(f"{m[k]:>{w}.3f}" for k, _, w in cols))


def print_verdict(rows, u_shape_min_k: int) -> bool:
    d = dict(rows)
    base = d["V0 fixed k=6"]
    ok = True
    print("\nverdict")

    ad = d["V1 adaptive-k"]
    dd = base["avg_distractors"] - ad["avg_distractors"]
    dr = ad["recall"] - base["recall"]
    print(f"  fixed k=6     recall={base['recall']:.3f}  "
          f"distractors={base['avg_distractors']:.2f}  k={base['avg_k']:.1f}")
    print(f"  adaptive-k    recall={ad['recall']:.3f}  "
          f"distractors={ad['avg_distractors']:.2f}  k={ad['avg_k']:.1f}")

    if dd > 0 and dr >= -0.02:
        print(f"\n  adaptive-k removes {dd:.2f} distractors per query at "
              f"{dr:+.3f} recall — accept.")
    elif dd > 0:
        print(f"\n  adaptive-k removes {dd:.2f} distractors but costs {dr:.3f} recall.")
        print("  Loosen gap_ratio, or raise k_min. This is the trade-off curve to")
        print("  sweep, not a pass/fail.")
    else:
        print("\n  WARNING adaptive-k did not reduce the distractor count.")
        print("          Usually means the scores are not calibrated — adaptive-k")
        print("          needs reranker scores (S3), not raw RRF scores, which are")
        print("          flat and make the relative cut fire at random.")
        ok = False

    mm = d["V2 +MMR"]
    if not np.isnan(mm["redundancy_drop"]):
        print(f"\n  MMR cuts mean pairwise similarity by {mm['redundancy_drop']:+.3f} "
              f"(to {mm['redundancy']:.3f}),")
        print(f"  recall {mm['recall'] - ad['recall']:+.3f}.")
        if mm["redundancy_drop"] < 0.01:
            print("  Near zero: the context was not redundant to begin with. Either")
            print("  the corpus has little duplication, or chunking already merged")
            print("  the near-duplicates. Leave MMR off and save the complexity.")

    us, mm2 = d["V3 +U-shape"], d["V2 +MMR"]
    de = us["gold_pos"] - mm2["gold_pos"]
    print(f"\n  U-shape fired on {us['reorder_rate']:.0%} of queries "
          f"(needs k >= {u_shape_min_k}; effective k is {ad['avg_k']:.1f}).")
    if us["reorder_rate"] > 0:
        print(f"  gold edge-distance {mm2['gold_pos']:.2f} -> {us['gold_pos']:.2f} "
              f"({de:+.2f}; lower is better).")
        print("  Note this is a PROXY. Ordering cannot be scored from retrieval —")
        print("  its effect only appears in generation metrics, so confirm it with")
        print("  the generation team before keeping it.")
    if us["reorder_rate"] < 0.25:
        print("  Expected. adaptive-k deliberately keeps the context small, and")
        print("  reordering only pays off on large retrieved sets. These two")
        print("  mechanisms work against each other — pick one.")
        print("  If you want ordering, raise k_max and loosen adaptive-k; if you")
        print("  want a small clean context, drop ordering. Running both as")
        print("  configured mostly means ordering is inert.")
    return ok


def print_cut_reasons(rows) -> None:
    print("\nwhere adaptive-k cut")
    for label, m in rows:
        if m["_cut_reasons"]:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(m["_cut_reasons"].items()))
            print(f"  {label:<18} {parts}")
    print("  gap   = score cliff found (healthy)")
    print("  k_max = no cliff; the scores were flat, so it padded to the ceiling")
    print("  floor = everything was below the absolute floor")


def sweep_gap(r, dataset, values=(0.0, 0.3, 0.55, 0.75, 0.9)) -> None:
    print("\ngap_ratio sweep (adaptive-k only)")
    print(f"  {'gap_ratio':>10}{'recall':>9}{'avg_k':>8}{'distract':>10}")
    for g in values:
        sel = SelectionConfig(use_adaptive_k=True, use_mmr=False,
                              use_u_shape=False, gap_ratio=g, k_max=6)
        m = evaluate(r, dataset, sel)
        print(f"  {g:>10.2f}{m['recall']:>9.3f}{m['avg_k']:>8.2f}"
              f"{m['avg_distractors']:>10.2f}")
    print("  Read this as a curve, not a winner: pick the point where recall is")
    print("  still acceptable and the distractor count has dropped the most.")


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
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if args.demo:
        from demo_corpus import DEMO_DOCS, DEMO_EVAL
        docs, dataset = DEMO_DOCS, DEMO_EVAL
        embedder, reranker = _HashingEmbedder(), LexicalOverlapReranker()
        print("demo mode: hashing embedder + stub reranker — read avg_distractors "
              "and the k dynamics")
    else:
        if not (args.docs and args.eval_path):
            ap.error("--docs and --eval are required (or use --demo)")
        docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines() if l.strip()]
        dataset = [json.loads(l) for l in args.eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        embedder = Embedder(EmbeddingConfig(model_name=args.model, device=args.device))
        reranker = Reranker(RerankConfig(device=args.device))

    chunks = chunk_documents(docs, ChunkConfig(), strategy="structure")
    print(f"corpus {len(docs)} docs -> {len(chunks)} chunks | eval {len(dataset)} queries")

    cfg = RetrieverConfig(mode="hybrid", rerank_candidates=20)
    r = HybridRetriever(embedder, cfg, reranker=reranker).index([c.to_dict() for c in chunks])

    rows = [(label, evaluate(r, dataset, sel)) for label, sel in VARIANTS]

    print_table(rows)
    print_cut_reasons(rows)
    ok = print_verdict(rows, VARIANTS[3][1].u_shape_min_k)

    if args.sweep:
        sweep_gap(r, dataset)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
