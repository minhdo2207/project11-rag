"""Cấu trúc dữ liệu dùng chung cho toàn pipeline.

Định nghĩa ``Document`` — đơn vị tri thức được lưu trong knowledge base và
được trả về khi truy xuất. Mọi module (repository, retriever, detector) đều
trao đổi qua kiểu này để giữ interface ổn định.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """Một mẩu tri thức trong knowledge base.

    Attributes:
        doc_id: Định danh duy nhất của tài liệu.
        text: Nội dung văn bản/mã nguồn.
        embedding: Vector nhúng của ``text`` (rỗng nếu chưa nhúng).
        metadata: Thông tin phụ (nguồn, loại: YARA/GHSA/sample, nhãn...).
        label: Nhãn nếu có ("malicious" | "benign" | None).
    """

    doc_id: str
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    label: str | None = None
