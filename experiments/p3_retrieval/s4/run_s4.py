#!/usr/bin/env python3
"""S4 ablation runner — scope filtering, and why it must come before scoring.

THE METRIC THIS SCENARIO EXISTS FOR
-----------------------------------
`stale_rate@k` — the share of queries where a SUPERSEDED chunk reaches the
context. Recall and precision are both blind to it: the superseded chunk is
on-topic, well-worded and a perfectly good semantic match. It is simply the
answer to a question nobody asked any more.

This is the single most dangerous retrieval error in a user-facing Q&A system,
because the resulting answer is fluent, specific, confidently wrong, and looks
exactly like a correct one.

FIVE VARIANTS
-------------
  V0  no filter                 the baseline; measures how often stale wins
  V1  no filter + rerank        tests whether reranking rescues it (it does not)
  V2  post-filter               the common mistake: k silently shrinks
  V3  pre-filter                the fix
  V4  pre-filter + rerank       does reranking still help once scope is clean?

    python run_s4.py --demo
    python run_s4.py --docs corpus.jsonl --eval eval.jsonl --as-of 2026-10-07

    corpus.jsonl  {"source": "...", "text": "...", "metadata": {"status": "current", ...}}
    eval.jsonl    {"query": "...", "answer_spans": [...], "stale_spans": [...]}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from chunking import ChunkConfig, chunk_documents
from embedding import Embedder, EmbeddingConfig
from metadata import (as_of_filter, coverage_report, extract_metadata,
                      validate_metadata)
from rerank import LexicalOverlapReranker, RerankConfig, Reranker
from retriever import HybridRetriever, RetrieverConfig, tokenize


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def has_span(text: str, spans) -> bool:
    return any(norm(sp) in norm(text) for sp in (spans or []))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def evaluate(r, dataset, *, where, filter_mode, rerank, ks=(1, 5)) -> dict:
    rec = {k: [] for k in ks}
    stale = {k: [] for k in ks}
    stale_wins, short, returned = [], [], []

    for row in dataset:
        res = r.retrieve(row["query"], top_k=max(ks), where=where,
                         filter_mode=filter_mode, rerank=rerank)
        returned.append(len(res.hits))
        short.append(res.diagnostics.get("short_of_k", 0))

        gold = [has_span(h.doc.text, row.get("answer_spans")) for h in res.hits]
        bad = [has_span(h.doc.text, row.get("stale_spans")) for h in res.hits]

        for k in ks:
            rec[k].append(1.0 if any(gold[:k]) else 0.0)
            stale[k].append(1.0 if any(bad[:k]) else 0.0)

        # the worst case: a superseded chunk outranks the current one
        gi = next((i for i, v in enumerate(gold) if v), None)
        bi = next((i for i, v in enumerate(bad) if v), None)
        stale_wins.append(1.0 if (bi is not None and (gi is None or bi < gi)) else 0.0)

    out = {f"recall@{k}": float(np.mean(rec[k])) for k in ks}
    out.update({f"stale_rate@{k}": float(np.mean(stale[k])) for k in ks})
    out["stale_outranks_current"] = float(np.mean(stale_wins))
    out["avg_returned"] = float(np.mean(returned))
    out["short_of_k"] = float(np.mean(short))
    return out


def abstention_check(r, queries, where, filter_mode) -> dict:
    """What scope filtering does NOT solve.

    A filter restricts the *universe*; it cannot tell that the *question*
    belongs to a different scope. Ask an enterprise question against a retail
    filter and the retriever happily returns retail chunks — they match, they
    are in scope, they are simply about the wrong customer segment.

    Closing this gap needs either query-side scope detection (S5) or a
    confidence threshold that notices the weak match (S7). S4 only abstains
    when the filter matches nothing at all.
    """
    empty, reasons = 0, []
    for q in queries:
        res = r.retrieve(q, top_k=5, where=where, filter_mode=filter_mode)
        if not res.hits:
            empty += 1
            reasons.append(res.empty_reason)
    return {"abstained": empty, "total": len(queries), "reasons": reasons}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

VARIANTS = [
    ("V0 no filter",            dict(where=None, filter_mode="off",  rerank=False)),
    ("V1 no filter + rerank",   dict(where=None, filter_mode="off",  rerank=True)),
    ("V2 post-filter",          dict(where="AS_OF", filter_mode="post", rerank=False)),
    ("V3 pre-filter",           dict(where="AS_OF", filter_mode="pre",  rerank=False)),
    ("V4 pre-filter + rerank",  dict(where="AS_OF", filter_mode="pre",  rerank=True)),
]


def print_table(rows) -> None:
    cols = [("recall@5", "rec@5", 9), ("stale_rate@5", "stale@5", 10),
            ("stale_outranks_current", "stale>cur", 11),
            ("avg_returned", "returned", 10), ("short_of_k", "short", 8)]
    header = f"{'variant':<24}" + "".join(f"{h:>{w}}" for _, h, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for label, m in rows:
        print(f"{label:<24}" + "".join(f"{m[k]:>{w}.3f}" for k, _, w in cols))


def print_verdict(rows) -> bool:
    d = dict(rows)
    ok = True
    print("\nverdict")

    base, pre = d["V0 no filter"], d["V3 pre-filter"]
    print(f"  no filter   stale@5={base['stale_rate@5']:.3f}  "
          f"stale outranks current in {base['stale_outranks_current']:.0%} of queries")
    print(f"  pre-filter  stale@5={pre['stale_rate@5']:.3f}  "
          f"rec@5={pre['recall@5']:.3f}")

    if pre["stale_rate@5"] > 0:
        print(f"\n  WARNING superseded chunks still reach the context ({pre['stale_rate@5']:.3f}).")
        print("          Metadata coverage is incomplete — check the coverage report")
        print("          above. A chunk with no `status` cannot be filtered out.")
        ok = False

    rr = d["V1 no filter + rerank"]
    if rr["stale_rate@5"] >= base["stale_rate@5"]:
        print(f"\n  Reranking did NOT reduce stale content "
              f"({base['stale_rate@5']:.3f} -> {rr['stale_rate@5']:.3f}).")
        print("  Expected: a cross-encoder has no way to know a chunk is out of date.")
        print("  Scope is a metadata problem and only metadata solves it.")

    post = d["V2 post-filter"]
    if post["short_of_k"] > 0:
        print(f"\n  post-filter returns {post['avg_returned']:.1f} chunks on average "
              f"instead of 5")
        print(f"  (short by {post['short_of_k']:.1f}). The caller asked for k and got")
        print("  less, with nothing in the response saying so. This is why the filter")
        print("  belongs before scoring, not after.")

    v4 = d["V4 pre-filter + rerank"]
    dr = v4["recall@5"] - pre["recall@5"]
    print(f"\n  rerank on top of a clean scope: recall@5 {dr:+.3f}")
    print("  Decide S3 and S4 together — reranking is worth far more once the")
    print("  candidate pool no longer contains plausible wrong-version chunks.")
    return ok


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--eval", dest="eval_path", type=Path)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--as-of", default=None, help="ISO date; default = today")
    ap.add_argument("--audience", default="retail")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.demo:
        from demo_corpus import DEMO_DOCS, DEMO_EVAL, DEMO_OUT_OF_SCOPE
        docs, dataset, oos = DEMO_DOCS, DEMO_EVAL, DEMO_OUT_OF_SCOPE
        embedder, reranker = _HashingEmbedder(), LexicalOverlapReranker()
        print("demo mode: hashing embedder + stub reranker — read stale_rate, "
              "not absolute quality")
    else:
        if not (args.docs and args.eval_path):
            ap.error("--docs and --eval are required (or use --demo)")
        docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines() if l.strip()]
        dataset = [json.loads(l) for l in args.eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        oos = []
        embedder = Embedder(EmbeddingConfig(model_name=args.model, device=args.device))
        reranker = Reranker(RerankConfig(device=args.device))

    # chunk, carrying document metadata down to every chunk and filling gaps
    # from the document body where it states its own effective dates
    chunks = []
    for d in docs:
        base = dict(d.get("metadata") or {})
        base = extract_metadata(d["text"], base)
        part = chunk_documents([d], ChunkConfig(), strategy="structure")
        for c in part:
            c.metadata.update(base)
        chunks.extend(part)

    print(f"corpus {len(docs)} docs -> {len(chunks)} chunks | eval {len(dataset)} queries")

    cov = validate_metadata(chunks)
    print("\nmetadata coverage")
    for f in ("status", "effective_from", "effective_to", "audience", "doc_type", "version"):
        print(f"  {f:<16}{cov[f]:>6.0%}")
    if cov["_missing_required"]:
        print(f"  ERROR missing required field(s): {cov['_missing_required']}")
        print("        Chunks without `status` cannot be excluded — fix the index first.")
        return 2

    as_of = args.as_of or date.today().isoformat()
    scope = as_of_filter(as_of, audience=args.audience)
    kept = coverage_report(chunks, scope)
    print(f"\nscope filter  as_of={as_of}  audience={args.audience}")
    print(f"  keeps {kept['kept']}/{kept['total']} chunks ({kept['ratio']:.0%})")
    if kept["ratio"] < 0.05:
        print("  WARNING the filter keeps under 5% of the index; expect mostly abstentions")

    cfg = RetrieverConfig(mode="hybrid", rerank_candidates=50)
    r = HybridRetriever(embedder, cfg, reranker=reranker).index([c.to_dict() for c in chunks])

    rows = []
    for label, kw in VARIANTS:
        kw = dict(kw)
        kw["where"] = scope if kw["where"] == "AS_OF" else None
        rows.append((label, evaluate(r, dataset, **kw)))

    print_table(rows)
    ok = print_verdict(rows)

    if oos:
        res = abstention_check(r, oos, scope, "pre")
        print(f"\nknown gap — questions belonging to another scope")
        print(f"  {res['abstained']}/{res['total']} abstained")
        for q, reason in zip(oos, res["reasons"]):
            print(f"  {q[:52]:<54} {reason or 'answered from the in-scope corpus'}")
        if res["abstained"] < res["total"]:
            print("  Expected. A scope filter restricts WHICH documents are eligible;")
            print("  it cannot detect that the QUESTION is about a different scope.")
            print("  That needs query-side scope detection (S5) or a confidence")
            print("  threshold on the weak match (S7). Note it and carry it forward.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
