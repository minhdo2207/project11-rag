"""S1 — Dense embedding.

Scope is deliberately narrow. S1 establishes the hybrid baseline, so this file
holds only what the baseline needs:

  * asymmetric encoding    instruction on the query side, raw text on documents
  * L2 normalisation       the retriever scores by dot product = cosine
  * content-hash cache     re-indexing after a chunking change only pays for
                           the chunks that actually changed

Deliberately NOT here (they belong to later scenarios):
  - Matryoshka truncation          -> a cost-tuning experiment, not a baseline
  - reranking                      -> S3
  - query rewriting                -> S5

Only numpy is a hard dependency; sentence-transformers / torch load lazily.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ["EmbeddingConfig", "Embedder", "EmbedderProtocol", "format_query"]

QUERY_TEMPLATE = "Instruct: {task}\nQuery: {query}"

DEFAULT_TASK = "Given a user question, retrieve the passage that answers it"


def format_query(text: str, task: str = DEFAULT_TASK) -> str:
    """Qwen3-Embedding query format. Documents must NOT be wrapped this way —
    that asymmetry is the point."""
    return QUERY_TEMPLATE.format(task=task, query=text)


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Implement this to plug in a different embedder; the retriever only needs
    these two methods."""
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class EmbeddingConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"

    backend: str = "local"                 # "local" = sentence-transformers, "http" = TEI/vLLM
    http_url: str = "http://127.0.0.1:8080/embed"
    http_timeout: int = 120

    device: str = "cuda"
    batch_size: int = 16
    max_seq_length: int = 2048

    query_instruction: str = DEFAULT_TASK
    use_query_instruction: bool = True      # set False to measure its contribution

    cache_dir: str | None = ".cache/embeddings"


class Embedder:
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._lock = threading.Lock()
        self._cache = Path(self.config.cache_dir) if self.config.cache_dir else None
        if self._cache:
            self._cache.mkdir(parents=True, exist_ok=True)

    # -- model -------------------------------------------------------------- #

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(self.config.model_name, device=self.config.device)
                    m.max_seq_length = self.config.max_seq_length
                    self._model = m
        return self._model

    def warmup(self) -> None:
        """Pay the load cost at service start, not on the first real query."""
        self.encode_queries(["warmup"])

    # -- public ------------------------------------------------------------- #

    def encode_documents(self, texts: Sequence[str], *, use_cache: bool = True,
                         show_progress: bool = False) -> np.ndarray:
        return self._encode(list(texts),
                            use_cache=use_cache and self._cache is not None,
                            show_progress=show_progress)

    def encode_queries(self, texts: Sequence[str], *, task: str | None = None) -> np.ndarray:
        if self.config.use_query_instruction:
            prepared = [format_query(t, task or self.config.query_instruction) for t in texts]
        else:
            prepared = list(texts)
        # queries are not cached: they rarely repeat, and a stale query cache
        # would hide prompt bugs
        return self._encode(prepared, use_cache=False, show_progress=False)

    # -- internals ---------------------------------------------------------- #

    def _key(self, text: str) -> str:
        sig = f"{self.config.model_name}|{self.config.max_seq_length}|{text}"
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()

    def _encode(self, texts: list[str], *, use_cache: bool, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        vecs: list[np.ndarray | None] = [None] * len(texts)
        todo: list[int] = []

        if use_cache and self._cache is not None:
            for i, t in enumerate(texts):
                f = self._cache / f"{self._key(t)}.npy"
                if f.exists():
                    try:
                        vecs[i] = np.load(f)
                        continue
                    except Exception:
                        f.unlink(missing_ok=True)
                todo.append(i)
        else:
            todo = list(range(len(texts)))

        if todo:
            batch = [texts[i] for i in todo]
            raw = (self._http(batch) if self.config.backend == "http"
                   else self._local(batch, show_progress))
            fresh = self._normalize(raw)
            for pos, i in enumerate(todo):
                vecs[i] = fresh[pos]
                if use_cache and self._cache is not None:
                    np.save(self._cache / f"{self._key(texts[i])}.npy", fresh[pos])

        return np.vstack([v for v in vecs if v is not None]).astype(np.float32)

    def _local(self, texts: list[str], show_progress: bool) -> np.ndarray:
        return np.asarray(self.model.encode(
            texts, batch_size=self.config.batch_size,
            show_progress_bar=show_progress, convert_to_numpy=True,
            normalize_embeddings=False,
        ))

    def _http(self, texts: list[str]) -> np.ndarray:
        import urllib.error
        import urllib.request
        out: list[list[float]] = []
        for i in range(0, len(texts), self.config.batch_size):
            payload = json.dumps({"inputs": texts[i: i + self.config.batch_size]}).encode()
            req = urllib.request.Request(self.config.http_url, data=payload,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.config.http_timeout) as r:
                    body = json.loads(r.read().decode())
            except urllib.error.URLError as e:
                raise RuntimeError(f"embedding endpoint unreachable: {e}") from e
            out.extend(item["embedding"] for item in body["data"]) if (
                isinstance(body, dict) and "data" in body) else out.extend(body)
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return np.ascontiguousarray(mat / np.clip(norms, 1e-12, None), dtype=np.float32)
