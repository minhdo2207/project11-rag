#!/usr/bin/env python3
"""S3 ablation runner — reranking as a falsifiable experiment.

WHY THIS IS NOT A "TURN IT ON" SCENARIO
---------------------------------------
Reranking is usually assumed to help. Published results say otherwise often
enough that it has to be measured: a generic MS MARCO cross-encoder has been
observed cutting precision at every cutoff on an out-of-domain corpus (P@5 from
0.600 to 0.404), and a cross-encoder can bury a chunk the first stage already
had at rank 1.

So S3 reports **precision@k, not just recall@k**. Recall can hold steady while
precision collapses — and precision is what decides how much distracting
material reaches the model.

THREE OUTPUTS
-------------
  1. quality      recall@k / precision@k / nDCG@10, rerank vs no-rerank
  2. cost         latency p50/p95, swept over candidate pool size
  3. calibration  separation between gold and non-gold rerank scores — this is
                  what S7 needs. A reranker that reorders well but produces
                  overlapping score distributions is useless for thresholding.

    python run_s3.py --demo
    python run_s3.py --docs corpus.jsonl --eval eval.jsonl --sweep

    corpus.jsonl  {"source": "faq.md", "text": "<full document>"}
    eval.jsonl    {"query": "...", "answer_spans": ["..."]}
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from chunking import ChunkConfig, chunk_documents
from embedding import Embedder, EmbeddingConfig
from rerank import LexicalOverlapReranker, RerankConfig, Reranker
from retriever import HybridRetriever, RetrieverConfig, tokenize


# --------------------------------------------------------------------------- #
# span matching (same contract as S2 — labels survive chunking changes)
# --------------------------------------------------------------------------- #

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def is_gold(chunk_text: str, spans) -> bool:
    return any(norm(sp) in norm(chunk_text) for sp in spans)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def evaluate(r: HybridRetriever, dataset, *, rerank: bool, ks=(1, 3, 5, 10)) -> dict:
    rec = {k: [] for k in ks}
    prec = {k: [] for k in ks}
    ndcg, lat, churn = [], [], []
    gold_scores, other_scores = [], []

    for row in dataset:
        t0 = time.perf_counter()
        res = r.retrieve(row["query"], top_k=max(ks), rerank=rerank)
        lat.append((time.perf_counter() - t0) * 1000)
        churn.append(res.diagnostics.get("rank_churn", 0.0))

        flags = [is_gold(h.doc.text, row["answer_spans"]) for h in res.hits]
        for k in ks:
            top = flags[:k]
            rec[k].append(1.0 if any(top) else 0.0)
            # precision@k: share of returned chunks that actually carry an answer.
            # This is the number that catches a reranker promoting on-topic but
            # answer-free chunks — invisible to recall.
            prec[k].append(sum(top) / max(len(top), 1))

        g = [1.0 if f else 0.0 for f in flags[:10]]
        dcg = sum(v / math.log2(i + 1) for i, v in enumerate(g, 1))
        idcg = sum(1 / math.log2(i + 1) for i in range(1, max(int(sum(g)), 1) + 1))
        ndcg.append(dcg / idcg if idcg else 0.0)

        if rerank:
            for h, f in zip(res.hits, flags):
                if h.rerank_score is not None:
                    (gold_scores if f else other_scores).append(h.rerank_score)

    out = {f"recall@{k}": float(np.mean(rec[k])) for k in ks}
    out.update({f"precision@{k}": float(np.mean(prec[k])) for k in ks})
    out["ndcg@10"] = float(np.mean(ndcg))
    out["lat_p50"] = float(np.percentile(lat, 50))
    out["lat_p95"] = float(np.percentile(lat, 95))
    out["rank_churn"] = float(np.mean(churn))
    out["_gold_scores"] = gold_scores
    out["_other_scores"] = other_scores
    return out


def separation(gold, other) -> dict:
    """How well rerank scores separate answer-bearing chunks from the rest.

    This is the S7 readiness check. `overlap` is the share of non-gold chunks
    scoring above the 10th percentile of gold chunks — i.e. the leakage you
    would get from a threshold placed there.
    """
    if not gold or not other:
        return {"gold_median": float("nan"), "other_median": float("nan"),
                "gap": float("nan"), "overlap": float("nan")}
    gp10 = float(np.percentile(gold, 10))
    return {
        "gold_median": float(np.median(gold)),
        "other_median": float(np.median(other)),
        "gap": float(np.median(gold) - np.median(other)),
        "overlap": float(np.mean([o >= gp10 for o in other])),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_table(rows: list[tuple[str, dict]]) -> None:
    cols = [("recall@5", "rec@5", 9), ("precision@5", "prec@5", 9),
            ("ndcg@10", "ndcg@10", 9), ("rank_churn", "churn", 8),
            ("lat_p50", "p50 ms", 9), ("lat_p95", "p95 ms", 9)]
    header = f"{'variant':<26}" + "".join(f"{h:>{w}}" for _, h, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for label, m in rows:
        print(f"{label:<26}" + "".join(f"{m[k]:>{w}.3f}" for k, _, w in cols))


def print_verdict(base: dict, variants: list[tuple[str, dict]],
                  latency_budget_ms: float | None) -> bool:
    print("\nverdict")
    ok = True
    best_label, best = max(variants, key=lambda kv: kv[1]["precision@5"])

    dp = best["precision@5"] - base["precision@5"]
    dr = best["recall@5"] - base["recall@5"]
    print(f"  baseline (no rerank)   prec@5={base['precision@5']:.3f}  "
          f"rec@5={base['recall@5']:.3f}  p95={base['lat_p95']:.0f}ms")
    print(f"  best ({best_label})  prec@5={best['precision@5']:.3f}  "
          f"rec@5={best['recall@5']:.3f}  p95={best['lat_p95']:.0f}ms")

    if dp < 0:
        print(f"\n  REJECT reranking: precision@5 drops {dp:+.3f}.")
        print("         This is the documented out-of-domain failure — the")
        print("         cross-encoder is promoting on-topic but answer-free chunks.")
        print("         Options, in order: try fusion='rrf', try another reranker,")
        print("         or keep the reranker ONLY as a score source for S7.")
        ok = False
    elif dp < 0.02 and dr < 0.02:
        print(f"\n  MARGINAL precision {dp:+.3f}, recall {dr:+.3f} for "
              f"{best['lat_p95'] - base['lat_p95']:+.0f}ms at p95.")
        print("           Probably not worth the latency on quality grounds alone.")
        print("           It may still be worth keeping for the S7 score — see below.")
    else:
        print(f"\n  ACCEPT precision {dp:+.3f}, recall {dr:+.3f}, "
              f"+{best['lat_p95'] - base['lat_p95']:.0f}ms at p95.")

    if latency_budget_ms and best["lat_p95"] > latency_budget_ms:
        print(f"\n  WARNING p95 {best['lat_p95']:.0f}ms exceeds the budget "
              f"({latency_budget_ms:.0f}ms).")
        print("          Reduce rerank_candidates — the quality curve usually")
        print("          flattens around 50.")
        ok = False

    if best["rank_churn"] < 0.5:
        print(f"\n  NOTE rank_churn is {best['rank_churn']:.2f} — the reranker barely")
        print("       changes the order. The first stage was already doing the work.")
    return ok


def print_calibration(variants: list[tuple[str, dict]]) -> None:
    print("\nS7 readiness — rerank score separation")
    print(f"  {'variant':<26}{'gold med':>10}{'other med':>11}{'gap':>8}{'overlap':>10}")
    for label, m in variants:
        s = separation(m["_gold_scores"], m["_other_scores"])
        print(f"  {label:<26}{s['gold_median']:>10.3f}{s['other_median']:>11.3f}"
              f"{s['gap']:>8.3f}{s['overlap']:>10.3f}")
    print("  overlap = share of non-answer chunks above the 10th percentile of")
    print("  answer chunks; that is the leakage a threshold placed there would")
    print("  admit. Above ~0.3, the score is too noisy to carry S7's abstention.")


def sweep_candidates(r, dataset, values=(10, 20, 50, 100)) -> None:
    print("\ncandidate pool sweep (fusion=replace)")
    print(f"  {'candidates':>11}{'prec@5':>9}{'rec@5':>8}{'p95 ms':>9}")
    original = r.config.rerank_candidates
    for n in values:
        r.config.rerank_candidates = n
        m = evaluate(r, dataset, rerank=True)
        print(f"  {n:>11}{m['precision@5']:>9.3f}{m['recall@5']:>8.3f}{m['lat_p95']:>9.0f}")
    r.config.rerank_candidates = original
    print("  Look for the knee: quality usually flattens well before latency does.")


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #

class _HashingEmbedder:
    dim = 512

    def _v(self, t):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(t):
            v[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts, **kw):
        return np.vstack([self._v(t) for t in texts])

    def encode_queries(self, texts, **kw):
        return np.vstack([self._v(t) for t in texts])


def _load_demo():
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_s2demo", Path(__file__).parent / "demo_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DEMO_DOCS, mod.DEMO_EVAL


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--eval", dest="eval_path", type=Path)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--rerank-kind", default="seq-cls", choices=["seq-cls", "qwen3"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--latency-budget", type=float, default=None,
                    help="p95 ms budget; the verdict fails if exceeded")
    args = ap.parse_args()

    if args.demo:
        docs, dataset = _load_demo()
        embedder = _HashingEmbedder()
        reranker = LexicalOverlapReranker()
        print("demo mode: hashing embedder + lexical-overlap stub reranker.")
        print("           Read the machinery, not the quality numbers.")
    else:
        if not (args.docs and args.eval_path):
            ap.error("--docs and --eval are required (or use --demo)")
        docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines() if l.strip()]
        dataset = [json.loads(l) for l in args.eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        embedder = Embedder(EmbeddingConfig(model_name=args.model, device=args.device))
        reranker = Reranker(RerankConfig(model_name=args.reranker, kind=args.rerank_kind,
                                         device=args.device))

    if any(not row.get("answer_spans") for row in dataset):
        print("ERROR every eval row needs 'answer_spans' (same contract as S2).")
        return 2

    chunks = chunk_documents(docs, ChunkConfig(), strategy="structure")
    print(f"corpus {len(docs)} docs -> {len(chunks)} chunks | eval {len(dataset)} queries")

    cfg = RetrieverConfig(mode="hybrid", rerank_candidates=50)
    r = HybridRetriever(embedder, cfg, reranker=reranker).index([c.to_dict() for c in chunks])

    base = evaluate(r, dataset, rerank=False)

    variants = []
    for label, fusion in (("rerank replace", "replace"), ("rerank rrf-blend", "rrf")):
        r.config.rerank_fusion = fusion
        variants.append((label, evaluate(r, dataset, rerank=True)))

    print_table([("no rerank (S1+S2)", base)] + variants)
    ok = print_verdict(base, variants, args.latency_budget)
    print_calibration(variants)

    if args.sweep:
        r.config.rerank_fusion = "replace"
        sweep_candidates(r, dataset)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
