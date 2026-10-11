#!/usr/bin/env python3
"""S5 ablation runner — query understanding.

S1–S4 all fix the corpus side. S5 is the first scenario on the *input* side,
and in a user-facing Vietnamese Q&A system that is where a large share of the
remaining misses live: the index is right, and retrieval still fails because the
question arrived as "lam sao doi mat khau ko".

RESULTS ARE REPORTED PER QUERY CATEGORY
---------------------------------------
An aggregate number hides everything here. A mechanism that lifts unaccented
queries by 40 points and does nothing elsewhere shows up as "+6 overall" and
gets dismissed. So every variant is scored separately on:

    plain       well-formed, accented, single-intent
    noaccent    no diacritics — normal input, not an edge case
    abbrev      domain abbreviations and colloquial wording
    multipart   two questions in one sentence

THE SAFETY RULE
---------------
A rewritten query is always FUSED with the original, never substituted. The
runner reports `broken` — queries that a variant made worse — because a rewriter
that drifts off-target is invisible in an aggregate that improved.

    python run_s5.py --demo
    python run_s5.py --docs corpus.jsonl --eval eval.jsonl
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
from embedding import Embedder, EmbeddingConfig
from query_understanding import (DEFAULT_GLOSSARY, QueryConfig, QueryUnderstanding,
                                 RuleRewriter, classify_query, fold_diacritics)
from rerank import LexicalOverlapReranker
from retriever import HybridRetriever, RetrieverConfig, tokenize


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def has_span(text: str, spans) -> bool:
    return any(norm(sp) in norm(text) for sp in (spans or []))


CATEGORIES = ("plain", "noaccent", "abbrev", "multipart")


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def evaluate(r, qu, dataset, ks=(1, 5)) -> dict:
    """Returns aggregate + per-category recall, plus per-query detail so the
    caller can diff variants and find what each one broke."""
    per_query: dict[str, float] = {}
    by_cat = {c: [] for c in CATEGORIES}
    rec = {k: [] for k in ks}
    n_variants = []

    for row in dataset:
        plan = qu.plan(row["query"]) if qu else row["query"]
        res = r.retrieve(plan, top_k=max(ks))
        n_variants.append(res.diagnostics.get("n_variants", 1))

        gold = [has_span(h.doc.text, row["answer_spans"]) for h in res.hits]
        for k in ks:
            rec[k].append(1.0 if any(gold[:k]) else 0.0)

        # Per-category scoring uses recall@1, not @5. On a corpus of a few dozen
        # chunks, top-5 is a large slice of the whole index and saturates — every
        # variant looks identical. recall@1 still discriminates.
        hit1 = 1.0 if (gold and gold[0]) else 0.0
        per_query[row["query"]] = hit1
        cat = row.get("category", "plain")
        if cat in by_cat:
            by_cat[cat].append(hit1)

    out = {f"recall@{k}": float(np.mean(rec[k])) for k in ks}
    for c in CATEGORIES:
        out[c] = float(np.mean(by_cat[c])) if by_cat[c] else float("nan")
    out["avg_variants"] = float(np.mean(n_variants))
    out["_per_query"] = per_query
    return out


# Each variant carries BOTH a query-side config and the index-side fold flag.
# Diacritic folding is not a query-side toggle: the BM25 index has to store the
# folded forms too, so the index is rebuilt per variant. Holding the index fixed
# while toggling only the query side makes folding look useless — the index was
# already doing the work.
VARIANTS = [
    ("V0 raw query",        False, QueryConfig(normalize=False, fold_diacritics=False,
                                               use_glossary=False, decompose=False)),
    ("V1 +normalise",       False, QueryConfig(normalize=True, fold_diacritics=False,
                                               use_glossary=False, decompose=False)),
    ("V2 +fold diacritics", True,  QueryConfig(normalize=True, fold_diacritics=True,
                                               use_glossary=False, decompose=False)),
    ("V3 +glossary",        True,  QueryConfig(normalize=True, fold_diacritics=True,
                                               use_glossary=True, decompose=False)),
    ("V4 +decompose",       True,  QueryConfig(normalize=True, fold_diacritics=True,
                                               use_glossary=True, decompose=True)),
]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_table(rows) -> None:
    # aggregate at @5, per-category at @1 (see evaluate())
    cols = [("recall@5", "rec@5", 9), ("recall@1", "rec@1", 8)] + \
           [(c, c + "@1", 11) for c in CATEGORIES] + \
           [("avg_variants", "variants", 10)]
    header = f"{'variant':<22}" + "".join(f"{h:>{w}}" for _, h, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for label, m in rows:
        print(f"{label:<22}" + "".join(f"{m[k]:>{w}.3f}" for k, _, w in cols))


def print_deltas(rows) -> None:
    print("\nmechanism contribution (overall recall@5; per category recall@1)")
    for i in range(1, len(rows)):
        (pl, prev), (cl, cur) = rows[i - 1], rows[i]
        name = cl.split("+", 1)[-1] if "+" in cl else cl
        bits = "  ".join(
            f"{c}={cur[c] - prev[c]:+.2f}" for c in CATEGORIES
            if not (np.isnan(cur[c]) or np.isnan(prev[c])))
        print(f"  {name:<18} overall {cur['recall@5'] - prev['recall@5']:+.3f}   {bits}")


def print_broken(rows) -> None:
    """Queries each mechanism made worse. The whole reason the original query is
    never dropped from the fusion."""
    print("\nqueries broken by a mechanism")
    any_broken = False
    for i in range(1, len(rows)):
        (pl, prev), (cl, cur) = rows[i - 1], rows[i]
        broken = [q for q, v in cur["_per_query"].items()
                  if v < prev["_per_query"].get(q, 0.0)]
        if broken:
            any_broken = True
            print(f"  {cl}: {len(broken)}")
            for q in broken[:4]:
                print(f"      {q[:62]}")
    if not any_broken:
        print("  none — every mechanism was monotone on this eval set")


def print_verdict(rows) -> bool:
    base, best = rows[0][1], max(rows, key=lambda kv: kv[1]["recall@5"])[1]
    best_label = max(rows, key=lambda kv: kv[1]["recall@5"])[0]
    print("\nverdict")
    print(f"  baseline {rows[0][0]}: recall@5={base['recall@5']:.3f}")
    print(f"  best     {best_label}: recall@5={best['recall@5']:.3f}")

    ok = True
    if best["recall@1"] <= base["recall@1"] and best["recall@5"] <= base["recall@5"]:
        print("\n  WARNING no mechanism beats the raw query.")
        print("          Likely the eval set is all well-formed queries. Collect")
        print("          real user input — unaccented, abbreviated, multi-part —")
        print("          before concluding query understanding is unnecessary.")
        ok = False

    v1, v2 = rows[1][1], rows[2][1]
    na = v2["noaccent"] - v1["noaccent"]
    if not np.isnan(na) and na > 0.05:
        print(f"\n  Diacritic folding carries the unaccented category "
              f"({v1['noaccent']:.2f} -> {v2['noaccent']:.2f}).")
        print("  This is an INDEX-side change: the BM25 index must store both the")
        print("  accented and the folded form, so adopting it means a rebuild, not")
        print("  a config flip at query time.")

    if best["avg_variants"] > 3:
        print(f"\n  NOTE {best['avg_variants']:.1f} variants per query on average.")
        print("       Each one costs an embedding call and a BM25 pass. Cap")
        print("       max_variants if latency matters.")
    return ok


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #

class _HashingEmbedder:
    """Stub dense encoder, deliberately diacritic-SENSITIVE.

    `tokenize()` folds by default, which would hand the dense branch unaccented
    matching for free and make the folding ablation look pointless. A real
    embedding model has no such guarantee on Vietnamese, so the stub is built
    without folding — otherwise the demo would understate the mechanism.
    """
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
    args = ap.parse_args()

    if args.demo:
        from demo_corpus import DEMO_DOCS, DEMO_EVAL
        docs, dataset = DEMO_DOCS, DEMO_EVAL
        embedder = _HashingEmbedder()
        print("demo mode: hashing embedder — the per-category pattern is the "
              "signal, not the absolute numbers")
    else:
        if not (args.docs and args.eval_path):
            ap.error("--docs and --eval are required (or use --demo)")
        docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines() if l.strip()]
        dataset = [json.loads(l) for l in args.eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        embedder = Embedder(EmbeddingConfig(model_name=args.model, device=args.device))

    chunks = chunk_documents(docs, ChunkConfig(), strategy="structure")
    print(f"corpus {len(docs)} docs -> {len(chunks)} chunks | eval {len(dataset)} queries")
    if len(chunks) < 40:
        print(f"  WARNING only {len(chunks)} chunks — top-5 covers a large share of")
        print("          the index, so recall@5 saturates. Per-category numbers below")
        print("          use recall@1. On a real corpus, read recall@5 too.")

    counts = {}
    for row in dataset:
        counts[row.get("category", "plain")] = counts.get(row.get("category", "plain"), 0) + 1
    print("  categories: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\nquery classification")
    for row in dataset[:6]:
        print(f"  {row['query'][:50]:<52} -> {classify_query(row['query'])}")

    payload = [c.to_dict() for c in chunks]
    index_cache: dict[bool, HybridRetriever] = {}

    def retriever_for(fold: bool) -> HybridRetriever:
        if fold not in index_cache:
            cfg = RetrieverConfig(mode="hybrid", fold_diacritics=fold)
            index_cache[fold] = HybridRetriever(embedder, cfg).index(payload)
        return index_cache[fold]

    rows = [(label, evaluate(retriever_for(fold), QueryUnderstanding(qc), dataset))
            for label, fold, qc in VARIANTS]

    print_table(rows)
    print_deltas(rows)
    print_broken(rows)
    ok = print_verdict(rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
