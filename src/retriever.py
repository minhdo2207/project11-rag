"""P3 — Truy xuất tri thức liên quan cho RAG.

STUB: định nghĩa interface ``Retriever``. P3 sẽ hiện thực:
  - vector retrieval (qua KnowledgeRepository.search),
  - hybrid (BM25 + vector),
  - re-ranking bằng cross-encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Document


class Retriever(ABC):
    """Lấy các tài liệu liên quan tới một truy vấn (đoạn code cần kiểm tra)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Trả về danh sách tài liệu liên quan nhất."""
