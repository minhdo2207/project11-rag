"""S3 — Cross-encoder reranking.

Reranking is usually presented as a free win. It is not, and S3 is built to find
that out on YOUR corpus rather than assume it:

  * A generic MS MARCO-trained cross-encoder has been measured *reducing*
    precision at every cutoff on an out-of-domain corpus (P@5 from 0.600 to
    0.404). Domain fit has to be tested before deployment.
  * A cross-encoder can also override a first stage that was already right —
    the classic symptom is hybrid retrieval ranking the correct chunk at #1 and
    the reranker burying it, because BM25 matched an exact term the
    cross-encoder reweights away.

Hence two things this module does that a plain reranker wrapper does not:

  1. `fusion="rrf"` — blend the cross-encoder ranking with the first-stage
     ranking instead of replacing it. Keeps the reranker's precision gain
     without letting it overrule a strong first stage.
  2. sigmoid-normalised scores in [0,1] — reranking's real contribution to
     hallucination work is not the reordering but producing a quantity that can
     carry a threshold across queries. S7 depends on this.

Model families:
  "seq-cls"  AutoModelForSequenceClassification — bge-reranker-v2-m3, mxbai, jina
  "qwen3"    Qwen3-Reranker, scored by the yes/no logit gap. Verify the prompt
             template against the model card before trusting numbers here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["RerankConfig", "Reranker", "LexicalOverlapReranker", "RerankerProtocol"]


@dataclass
class RerankConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    kind: str = "seq-cls"               # "seq-cls" | "qwen3"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 1024

    candidates: int = 50                # how many first-stage hits to rescore
    fusion: str = "replace"             # "replace" = pure CE order
                                        # "rrf"     = fuse CE rank with first-stage rank
    fusion_k: int = 60
    weight_ce: float = 1.0
    weight_first_stage: float = 1.0

    instruction: str = "Retrieve the passage that answers the query"


class RerankerProtocol:
    """score(query, docs) -> np.ndarray in [0,1], one per doc."""
    def score(self, query: str, documents: Sequence[str]) -> np.ndarray: ...


_QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    "on the Query and the Instruct provided. Note that the answer can only be "
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    def __init__(self, config: RerankConfig | None = None):
        self.config = config or RerankConfig()
        self._model = None
        self._tok = None
        self.last_latency_ms: float = 0.0

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (AutoModelForCausalLM,
                                  AutoModelForSequenceClassification, AutoTokenizer)
        c = self.config
        self._tok = AutoTokenizer.from_pretrained(c.model_name)
        if c.kind == "qwen3":
            self._tok.padding_side = "left"
            self._model = AutoModelForCausalLM.from_pretrained(
                c.model_name, torch_dtype=torch.float16).to(c.device).eval()
            self._yes = self._tok.convert_tokens_to_ids("yes")
            self._no = self._tok.convert_tokens_to_ids("no")
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                c.model_name, torch_dtype=torch.float16).to(c.device).eval()

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.zeros(0, dtype=np.float32)
        self._load()
        import torch

        c, out = self.config, []
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(documents), c.batch_size):
                batch = list(documents[i: i + c.batch_size])
                if c.kind == "qwen3":
                    prompts = [
                        f"{_QWEN_PREFIX}<Instruct>: {c.instruction}\n<Query>: {query}\n"
                        f"<Document>: {d}{_QWEN_SUFFIX}" for d in batch
                    ]
                    enc = self._tok(prompts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=c.max_length).to(c.device)
                    logits = self._model(**enc).logits[:, -1, :]
                    pair = torch.stack([logits[:, self._no], logits[:, self._yes]], dim=1)
                    out.extend(torch.log_softmax(pair.float(), 1)[:, 1].exp().cpu().tolist())
                else:
                    enc = self._tok([query] * len(batch), batch, return_tensors="pt",
                                    padding=True, truncation=True,
                                    max_length=c.max_length).to(c.device)
                    logits = self._model(**enc).logits.view(-1).float()
                    # sigmoid, not softmax: scores must be comparable ACROSS queries
                    # for S7 thresholding, and softmax would normalise that away
                    out.extend(torch.sigmoid(logits).cpu().tolist())
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return np.asarray(out, dtype=np.float32)


class LexicalOverlapReranker:
    """Offline stand-in so S3's machinery is runnable with no model download.

    Scores by token overlap with a length penalty. Not a cross-encoder — it has
    no notion of meaning. Use it to verify the pipeline and the latency
    bookkeeping, never to decide whether reranking helps.
    """

    def __init__(self, config: RerankConfig | None = None):
        self.config = config or RerankConfig()
        self.last_latency_ms = 0.0

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        from retriever import tokenize
        t0 = time.perf_counter()
        q = set(tokenize(query))
        out = []
        for d in documents:
            toks = tokenize(d)
            if not toks or not q:
                out.append(0.0)
                continue
            overlap = len(q & set(toks)) / len(q)
            density = len(q & set(toks)) / (len(set(toks)) ** 0.5 + 1e-9)
            out.append(float(1 / (1 + np.exp(-(4 * overlap + 2 * density - 3)))))
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        return np.asarray(out, dtype=np.float32)
