"""Interface trừu tượng cho lớp lưu trữ knowledge base.

Đây là điểm mấu chốt đáp ứng yêu cầu non-functional của đề: business logic chỉ
phụ thuộc vào ``KnowledgeRepository`` (trừu tượng), không biết dữ liệu nằm trong
RAM hay trên file. Nhờ vậy đổi backend chỉ cần thay lớp khởi tạo, các module
khác giữ nguyên (dependency inversion).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Document


class KnowledgeRepository(ABC):
    """Hợp đồng chung cho mọi cách lưu trữ tri thức (memory hoặc file)."""

    @abstractmethod
    def add(self, document: Document) -> None:
        """Thêm (hoặc ghi đè nếu trùng ``doc_id``) một tài liệu."""

    @abstractmethod
    def get(self, doc_id: str) -> Document | None:
        """Lấy tài liệu theo id, trả về None nếu không tồn tại."""

    @abstractmethod
    def all(self) -> list[Document]:
        """Trả về toàn bộ tài liệu hiện có."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        """Trả về ``top_k`` tài liệu gần nhất theo độ tương đồng cosine."""

    @abstractmethod
    def count(self) -> int:
        """Số tài liệu đang lưu."""

    @abstractmethod
    def clear(self) -> None:
        """Xoá toàn bộ tài liệu."""
