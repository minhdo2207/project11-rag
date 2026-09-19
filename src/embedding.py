"""P3 — Nhúng văn bản/mã nguồn thành vector.

STUB: định nghĩa interface ``Embedder``. P3 sẽ hiện thực bằng một mô hình nhúng
thực tế (vd: sentence-transformers, hoặc embedding của Ollama).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Chuyển một đoạn text thành vector số."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Nhúng một đoạn text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Nhúng nhiều đoạn (mặc định gọi lần lượt; P3 có thể tối ưu)."""
        return [self.embed(t) for t in texts]
