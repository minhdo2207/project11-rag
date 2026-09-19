"""Hàm tiện ích tính độ tương đồng, dùng chung cho các repository.

Tách riêng để hai backend (memory/file) chia sẻ cùng một logic xếp hạng,
đảm bảo chúng cho kết quả ``search`` giống hệt nhau.
"""

from __future__ import annotations

import math

from ..models import Document


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Độ tương đồng cosine giữa hai vector; trả 0.0 nếu vector rỗng/không hợp lệ."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_by_similarity(
    documents: list[Document], query_embedding: list[float], top_k: int
) -> list[Document]:
    """Xếp hạng ``documents`` theo cosine với ``query_embedding``, lấy ``top_k``."""
    scored = [
        (cosine_similarity(query_embedding, doc.embedding), doc) for doc in documents
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
