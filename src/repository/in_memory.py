"""Backend lưu trữ trong bộ nhớ (RAM).

Nhanh, không cần I/O đĩa; dữ liệu mất khi tiến trình kết thúc. Phù hợp thí
nghiệm nhanh và unit test.
"""

from __future__ import annotations

from ..models import Document
from ._similarity import rank_by_similarity
from .base import KnowledgeRepository


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Lưu tài liệu trong một ``dict`` theo ``doc_id``."""

    def __init__(self) -> None:
        self._store: dict[str, Document] = {}

    def add(self, document: Document) -> None:
        self._store[document.doc_id] = document

    def get(self, doc_id: str) -> Document | None:
        return self._store.get(doc_id)

    def all(self) -> list[Document]:
        return list(self._store.values())

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        return rank_by_similarity(self.all(), query_embedding, top_k)

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
