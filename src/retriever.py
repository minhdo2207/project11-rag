"""P3 — Truy xuất tri thức liên quan cho RAG.

``Retriever`` là interface. Trong ``src`` hiện có ``VectorRetriever`` (nhúng truy vấn rồi
tìm gần nhất qua ``KnowledgeRepository.search``). Hybrid BM25 + vector, rerank và các thí
nghiệm khác của P3 nằm ở ``experiments/p3_retrieval`` và chưa nối vào đây.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import Config
from .embedding import Embedder
from .models import Document
from .repository._similarity import cosine_similarity
from .repository.base import KnowledgeRepository


class Retriever(ABC):
    """Lấy các tài liệu liên quan tới một truy vấn (đoạn code cần kiểm tra)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Trả về danh sách tài liệu liên quan nhất."""


class VectorRetriever(Retriever):
    """Truy xuất thuần vector. Tài liệu dưới ``score_threshold`` (cosine) bị bỏ."""

    def __init__(
        self, repository: KnowledgeRepository, embedder: Embedder, score_threshold: float = 0.0
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._threshold = score_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        query_vec = self._embedder.embed(query)
        docs = self._repo.search(query_vec, top_k)
        if self._threshold <= 0:
            return docs
        return [d for d in docs if cosine_similarity(query_vec, d.embedding) >= self._threshold]


def embed_missing(repo: KnowledgeRepository, embedder: Embedder, batch_size: int = 32) -> int:
    """Nhúng các tài liệu trong ``repo`` chưa có vector (vd: vừa ingest xong). Trả về số đã nhúng."""
    pending = [d for d in repo.all() if not d.embedding]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        for doc, vec in zip(batch, embedder.embed_batch([d.text for d in batch]), strict=True):
            doc.embedding = vec
    repo.add_many(pending)
    return len(pending)


def build_retriever(cfg: Config, repo: KnowledgeRepository, embedder: Embedder) -> Retriever:
    """Dựng retriever theo ``pipeline.retrieval``; hiện chỉ có ``vector``."""
    kind = cfg.get("pipeline.retrieval", "vector")
    if kind != "vector":
        raise NotImplementedError(
            f"pipeline.retrieval={kind!r} chưa nối vào src; "
            "bản hybrid/rerank nằm ở experiments/p3_retrieval."
        )
    return VectorRetriever(repo, embedder, cfg.get("retrieval.score_threshold", 0.0))
