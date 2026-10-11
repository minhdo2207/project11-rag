"""S1 — Hybrid retrieval baseline.

Three components, nothing more:

    dense bi-encoder  +  BM25 (code-aware tokenizer)  →  RRF (k=60)

The `mode` switch ("dense" | "bm25" | "hybrid") is the whole point of S1. The
deliverable of this scenario is not the retriever — it is the three-way
comparison that tells you whether fusion is actually earning its keep, and
whether your embedding model is contributing at all.

Why RRF and not a weighted score blend: BM25 produces unbounded positive scores
while cosine is bounded in [-1, 1]. Mixing the raw numbers hands BM25 dominant
weight by accident. RRF operates only on rank positions, so no normalisation
assumption is needed.

S3 ADDS a second stage: an optional cross-encoder reranker over the top-N fused
candidates. Two things make this more than a wrapper call:

  * `fusion="rrf"` blends the cross-encoder ranking with the first-stage ranking
    rather than replacing it, so a reranker cannot overrule a first stage that
    was already right.
  * every Hit carries both `fused_score` and `rerank_score`, so the ablation can
    attribute any change to the right stage, and S7 can calibrate a threshold on
    the reranker score.

S4 ADDS scope filtering. `retrieve(..., where={...})` restricts the candidate
pool BEFORE either branch scores anything. `filter_mode` exists only so the
ablation can demonstrate why:

  "pre"   restrict the pool, then score  -- correct; top-k stays top-k
  "post"  score, then drop non-matching  -- the common mistake; silently
          returns fewer than k, so the model gets thinner evidence than
          configured and nobody notices
  "off"   no filtering

When a filter matches nothing the retriever returns empty with a reason rather
than falling back to an unfiltered search. A silent fallback is how a system
ends up answering a 2026 question with a 2025 rule, fluently.

S5 ADDS query understanding. `retrieve()` accepts a QueryPlan instead of a bare
string: several query variants are retrieved in parallel and fused, and the
fusion weights come from the query class rather than being fixed.

Two retrieval-side changes S5 requires:
  * `tokenize(..., fold_diacritics=True)` indexes both "mật" and "mat", so an
    unaccented query still has a lexical branch to work with.
  * variant fusion: every variant contributes a ranked list to the same RRF,
    with the original always weighted highest.

Still NOT here (later scenarios):
    S6 adaptive-k / MMR · S7 abstention · S8 index-time enrichment

Dependencies: numpy, scipy.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import sparse

__all__ = [
    "RetrieverConfig", "Document", "Hit", "RetrievalResult",
    "HybridRetriever", "BM25Index", "tokenize", "rrf",
]


# =========================================================================== #
# config & records
# =========================================================================== #

@dataclass
class RetrieverConfig:
    mode: str = "hybrid"            # "dense" | "bm25" | "hybrid"

    dense_top_k: int = 60           # candidates from each branch before fusion
    lexical_top_k: int = 60
    rrf_k: int = 60                 # RRF smoothing constant
    weight_dense: float = 1.0       # S1 keeps these equal; S5 makes them query-dependent
    weight_lexical: float = 1.0

    top_k: int = 5                  # fixed k — adaptive-k is S6

    # --- S4: scope filtering ---
    filter_mode: str = "pre"        # "pre" | "post" | "off"

    # --- S3: reranking ---
    rerank_enabled: bool = False    # off by default: S3 must prove it helps
    rerank_candidates: int = 50     # first-stage hits sent to the cross-encoder
    rerank_fusion: str = "replace"  # "replace" = pure CE order
                                    # "rrf"     = fuse CE rank with first-stage rank
    rerank_fusion_k: int = 60
    rerank_weight_ce: float = 1.0
    rerank_weight_first: float = 1.0

    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    split_identifiers: bool = True
    fold_diacritics: bool = True    # S5: index accented and unaccented forms

    # --- S5: multi-variant fusion ---
    variant_decay: float = 0.6      # weight of variant i = decay**i; the original
                                    # (i=0) always carries weight 1.0


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embed_text: str | None = None

    @property
    def to_embed(self) -> str:
        return self.embed_text or self.text


@dataclass
class Hit:
    doc: Document
    score: float                    # final score after whatever stages ran
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    # --- S3 ---
    fused_score: float | None = None    # first-stage score, kept for attribution
    fused_rank: int | None = None
    rerank_score: float | None = None   # cross-encoder score in [0,1]


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    mode: str = "hybrid"
    diagnostics: dict = field(default_factory=dict)
    # --- S4 ---
    empty_reason: str = ""          # non-empty when the filter excluded everything

    def ids(self) -> list[str]:
        return [h.doc.id for h in self.hits]

    def context(self, max_chars: int = 12000, separator: str = "\n---\n") -> str:
        parts, used = [], 0
        for h in self.hits:
            src = h.doc.metadata.get("source") or h.doc.metadata.get("path") or ""
            block = f"[{h.doc.id}] {src}".rstrip() + f"\n{h.doc.text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block) + len(separator)
        return separator.join(parts)


def _coerce(obj: Any) -> Document:
    """Accept Document, dict, or any object with id/text — so this drops into an
    existing codebase without forcing a schema change."""
    if isinstance(obj, Document):
        return obj
    if isinstance(obj, dict):
        return Document(
            id=str(obj.get("id") or obj.get("chunk_id") or obj.get("doc_id")),
            text=obj.get("text") or obj.get("content") or "",
            metadata=obj.get("metadata") or obj.get("meta") or {},
            embed_text=obj.get("embed_text"),
        )
    return Document(
        id=str(getattr(obj, "id", None) or getattr(obj, "chunk_id")),
        text=getattr(obj, "text", "") or getattr(obj, "content", ""),
        metadata=getattr(obj, "metadata", None) or getattr(obj, "meta", None) or {},
        embed_text=getattr(obj, "embed_text", None),
    )


# =========================================================================== #
# lexical branch
# =========================================================================== #

# Unicode-aware. The ASCII-only class [^0-9A-Za-z_] shreds Vietnamese text:
# "mật khẩu" tokenises to ['m','t','kh','u'] because every accented character is
# treated as a separator, which silently disables the lexical branch for the
# entire corpus. \W respects Unicode letters in Python 3 str patterns.
_NON_WORD = re.compile(r"\W+", re.UNICODE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _fold(s: str) -> str:
    """Strip Vietnamese tone marks; đ is a separate letter, not a combining mark."""
    import unicodedata
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(c))


def tokenize(text: str, split_identifiers: bool = True,
             fold_diacritics: bool = True) -> list[str]:
    """Keeps each identifier whole AND emits its subtokens.

    A generic word tokenizer turns `build_comm_edges` into one opaque token that
    never matches "build communication edges"; keeping only subtokens loses
    exact-symbol matching. Emitting both wins on either phrasing, at the cost of
    a slightly larger vocabulary.
    """
    out: list[str] = []
    for raw in _NON_WORD.split(text):
        if not raw:
            continue
        low = raw.lower()
        out.append(low)
        if fold_diacritics:
            f = _fold(low)
            if f != low:
                # emit BOTH forms so an unaccented query matches an accented
                # document without losing the accented match
                out.append(f)
        if not split_identifiers:
            continue
        parts = [p for chunk in raw.split("_") if chunk for p in _CAMEL.split(chunk)]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts if len(p) > 1)
    return out


class BM25Index:
    """BM25 on a precomputed sparse weight matrix: scoring a query is a column
    gather plus a row sum. No external retrieval library to install."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, split_identifiers: bool = True,
                 fold_diacritics: bool = True):
        self.k1, self.b, self.split_identifiers = k1, b, split_identifiers
        self.fold_diacritics = fold_diacritics
        self.vocab: dict[str, int] = {}
        self.W: sparse.csc_matrix | None = None
        self.n_docs = 0

    def build(self, texts: Sequence[str]) -> "BM25Index":
        docs = [tokenize(t, self.split_identifiers, self.fold_diacritics) for t in texts]
        self.n_docs = len(docs)
        if not self.n_docs:
            return self

        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        for d, toks in enumerate(docs):
            counts: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(d); cols.append(j); tfs.append(c)
        self.vocab = vocab

        tf = sparse.csr_matrix((tfs, (rows, cols)),
                               shape=(self.n_docs, len(vocab)), dtype=np.float32).tocoo()
        dl = np.asarray([len(d) for d in docs], dtype=np.float32)
        avgdl = max(float(dl.mean()), 1e-9)
        df = np.bincount(tf.col, minlength=len(vocab)).astype(np.float32)
        idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        denom = self.k1 * (1 - self.b + self.b * dl / avgdl)
        w = (tf.data * (self.k1 + 1)) / (tf.data + denom[tf.row]) * idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)),
                                   shape=(self.n_docs, len(vocab)), dtype=np.float32).tocsc()
        return self

    def search(self, query: str, top_k: int = 60,
               mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        if self.W is None or not self.n_docs:
            return []
        cols = [self.vocab[t] for t in tokenize(query, self.split_identifiers,
                                                self.fold_diacritics)
                if t in self.vocab]
        if not cols:
            return []
        scores = np.asarray(self.W[:, cols].sum(axis=1)).ravel()
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        k = min(top_k, self.n_docs)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


# =========================================================================== #
# fusion
# =========================================================================== #

def rrf(ranked_lists: Sequence[tuple[Sequence[int], float]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = Σ w_i / (k + rank_i(d)).

    k controls how sharply top ranks are favoured — smaller k gives the top of
    each list more influence. k=60 is the value from Cormack et al. (2009) and a
    reasonable default; sweep it once you have an eval set.
    """
    out: dict[int, float] = {}
    for ids, w in ranked_lists:
        if w <= 0:
            continue
        for rank, doc_id in enumerate(ids, start=1):
            out[doc_id] = out.get(doc_id, 0.0) + w / (k + rank)
    return out


# =========================================================================== #
# retriever
# =========================================================================== #

class HybridRetriever:
    def __init__(self, embedder, config: RetrieverConfig | None = None,
                 reranker=None):
        self.embedder = embedder
        self.config = config or RetrieverConfig()
        self.reranker = reranker        # any object with .score(query, docs) -> ndarray
        self.docs: list[Document] = []
        self.vectors: np.ndarray | None = None
        self.bm25: BM25Index | None = None
        self._row: dict[str, int] = {}

    # -- indexing ----------------------------------------------------------- #

    def index(self, documents: Iterable[Any], *, vectors: np.ndarray | None = None,
              show_progress: bool = False) -> "HybridRetriever":
        docs = [_coerce(d) for d in documents]
        if not docs:
            raise ValueError("no documents to index")

        if vectors is None:
            try:
                vectors = self.embedder.encode_documents(
                    [d.to_embed for d in docs], show_progress=show_progress)
            except TypeError:                      # embedder without show_progress
                vectors = self.embedder.encode_documents([d.to_embed for d in docs])
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vectors) != len(docs):
            raise ValueError(f"vectors({len(vectors)}) != documents({len(docs)})")

        self.docs, self.vectors = docs, vectors
        self._row = {d.id: i for i, d in enumerate(docs)}
        self.bm25 = BM25Index(self.config.bm25_k1, self.config.bm25_b,
                              self.config.split_identifiers,
                              self.config.fold_diacritics).build([d.to_embed for d in docs])
        return self

    def _row_of(self, doc_id: str) -> int:
        return self._row[doc_id]

    # -- S4: scope mask ----------------------------------------------------- #

    def build_mask(self, where) -> np.ndarray | None:
        """Boolean mask over the index. Built once per query, used by both
        branches, so dense and BM25 always see the same universe."""
        from metadata import compile_filter
        pred = compile_filter(where)
        if pred is None:
            return None
        return np.fromiter((bool(pred(d.metadata or {})) for d in self.docs),
                           bool, len(self.docs))

    # -- branches ----------------------------------------------------------- #

    def _dense(self, qvec: np.ndarray, top_k: int,
               mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        scores = self.vectors @ qvec.reshape(-1)        # cosine; vectors normalised
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(top_k, len(self.docs))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if np.isfinite(scores[i])]

    # -- main --------------------------------------------------------------- #

    def retrieve(self, query, *, top_k: int | None = None,
                 mode: str | None = None,
                 rerank: bool | None = None,
                 where=None,
                 filter_mode: str | None = None) -> RetrievalResult:
        """`query` is a string, or a QueryPlan from S5 (duck-typed: anything with
        .original, .variants, .weight_dense, .weight_lexical)."""
        if self.vectors is None or self.bm25 is None:
            raise RuntimeError("call index() before retrieve()")

        c = self.config
        mode = mode or c.mode

        # --- S5: unpack the plan ------------------------------------------- #
        if isinstance(query, str):
            variants = [query]
            q_text = query
            w_dense, w_lex = c.weight_dense, c.weight_lexical
            plan_notes: list[str] = []
        else:
            q_text = query.original
            variants = list(query.variants) or [q_text]
            w_dense, w_lex = query.weight_dense, query.weight_lexical
            plan_notes = list(getattr(query, "notes", []))
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown mode {mode!r}")
        k_final = top_k or c.top_k

        fmode = filter_mode or c.filter_mode
        mask = self.build_mask(where) if (where is not None and fmode != "off") else None

        if mask is not None and not mask.any():
            return RetrievalResult(
                query=q_text, hits=[], mode=mode,
                empty_reason="no document matches the scope filter",
                diagnostics={"filter_mode": fmode, "n_in_scope": 0},
            )

        pre_mask = mask if fmode == "pre" else None

        # --- retrieve every variant, keep per-variant ranked lists ---------- #
        dense_lists: list[tuple[list[int], float]] = []
        lex_lists: list[tuple[list[int], float]] = []
        dense = lexical = []

        if mode in {"dense", "hybrid"}:
            qvecs = self.embedder.encode_queries(variants)
            for i, qv in enumerate(qvecs):
                hits_i = self._dense(qv, c.dense_top_k, pre_mask)
                if i == 0:
                    dense = hits_i
                    qvec = qv
                dense_lists.append(([j for j, _ in hits_i],
                                    w_dense * (c.variant_decay ** i)))
        if mode in {"bm25", "hybrid"}:
            for i, v in enumerate(variants):
                hits_i = self.bm25.search(v, c.lexical_top_k, pre_mask)
                if i == 0:
                    lexical = hits_i
                lex_lists.append(([j for j, _ in hits_i],
                                  w_lex * (c.variant_decay ** i)))

        dmap = {i: (r, s) for r, (i, s) in enumerate(dense, 1)}
        lmap = {i: (r, s) for r, (i, s) in enumerate(lexical, 1)}

        if mode == "dense":
            ordered = [(i, s) for i, s in dense]
        elif mode == "bm25":
            ordered = [(i, s) for i, s in lexical]
        else:
            # one RRF over every (branch, variant) ranked list. Variant weight
            # decays with position, so a drifting rewrite can nudge the ranking
            # but never outvote the original.
            fused = rrf(dense_lists + lex_lists, k=c.rrf_k)
            ordered = sorted(fused.items(), key=lambda kv: -kv[1])

        # --- S3: rerank the shortlist ------------------------------------- #
        use_rerank = (rerank if rerank is not None else c.rerank_enabled) and \
            self.reranker is not None
        pool_n = c.rerank_candidates if use_rerank else k_final
        pool = ordered[:pool_n]

        hits = [
            Hit(doc=self.docs[i], score=float(sc),
                fused_score=float(sc), fused_rank=rank,
                dense_rank=dmap[i][0] if i in dmap else None,
                lexical_rank=lmap[i][0] if i in lmap else None,
                dense_score=dmap[i][1] if i in dmap else None,
                lexical_score=lmap[i][1] if i in lmap else None)
            for rank, (i, sc) in enumerate(pool, start=1)
        ]

        rerank_ms = 0.0
        if use_rerank and hits:
            scores = self.reranker.score(q_text, [h.doc.text for h in hits])
            rerank_ms = getattr(self.reranker, "last_latency_ms", 0.0)
            for h, sc in zip(hits, scores):
                h.rerank_score = float(sc)

            if c.rerank_fusion == "rrf":
                # Blend CE rank with first-stage rank. Guards against the known
                # failure where a cross-encoder buries a chunk the first stage
                # had correctly at #1 (typically an exact lexical match it
                # reweights away).
                ce_order = sorted(range(len(hits)), key=lambda j: -hits[j].rerank_score)
                ce_rank = {j: r for r, j in enumerate(ce_order, start=1)}
                blended = {
                    j: c.rerank_weight_ce / (c.rerank_fusion_k + ce_rank[j])
                       + c.rerank_weight_first / (c.rerank_fusion_k + hits[j].fused_rank)
                    for j in range(len(hits))
                }
                for j, h in enumerate(hits):
                    h.score = blended[j]
            else:
                for h in hits:
                    h.score = h.rerank_score

            hits.sort(key=lambda h: -h.score)

        if fmode == "post" and mask is not None:
            # The mistake this mode exists to demonstrate: filtering after
            # ranking removes results from an already-truncated list, so the
            # caller asked for k and gets fewer, with no signal that it happened.
            hits = [h for h in hits if mask[self._row_of(h.doc.id)]]

        hits = hits[:k_final]

        return RetrievalResult(
            query=q_text, hits=hits, mode=mode,
            diagnostics={
                # --- S5 ---
                "n_variants": len(variants),
                "variants": variants,
                "plan_notes": plan_notes,
                "weight_dense": w_dense, "weight_lexical": w_lex,
                "n_dense": len(dense), "n_lexical": len(lexical),
                # how much each branch contributed to the final list
                "from_dense_only": sum(1 for h in hits if h.lexical_rank is None),
                "from_lexical_only": sum(1 for h in hits if h.dense_rank is None),
                "from_both": sum(1 for h in hits
                                 if h.dense_rank is not None and h.lexical_rank is not None),
                # --- S3 ---
                "reranked": use_rerank,
                "n_candidates": len(pool),
                "rerank_ms": rerank_ms,
                "best_rerank_score": hits[0].rerank_score if (use_rerank and hits) else None,
                # how far the reranker moved things: 0 means it changed nothing
                "rank_churn": (sum(abs(h.fused_rank - r) for r, h in enumerate(hits, 1))
                               / max(len(hits), 1)) if use_rerank else 0.0,
                # --- S4 ---
                "filter_mode": fmode if where is not None else "off",
                "n_in_scope": int(mask.sum()) if mask is not None else len(self.docs),
                "returned": len(hits),
                # post-filtering shows up here as returned < k_final
                "short_of_k": max(0, k_final - len(hits)),
            },
        )

    # -- persistence -------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        d = Path(path); d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with open(d / "documents.jsonl", "w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(json.dumps({"id": doc.id, "text": doc.text,
                                    "metadata": doc.metadata,
                                    "embed_text": doc.embed_text},
                                   ensure_ascii=False) + "\n")
        with open(d / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

    @classmethod
    def load(cls, path: str | Path, embedder,
             config: RetrieverConfig | None = None) -> "HybridRetriever":
        d = Path(path)
        r = cls(embedder, config)
        r.vectors = np.load(d / "vectors.npy")
        with open(d / "documents.jsonl", encoding="utf-8") as f:
            r.docs = [Document(**json.loads(line)) for line in f if line.strip()]
        with open(d / "bm25.pkl", "rb") as f:
            r.bm25 = pickle.load(f)
        return r
