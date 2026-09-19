"""P4 — Business logic phán quyết một gói/đoạn code là độc hay lành.

Đây là ví dụ minh hoạ cách business logic chỉ phụ thuộc vào các *interface*
(Retriever, LLMClient) — không hề biết knowledge base nằm ở memory hay file.
Nhờ đó đổi backend persistence không cần sửa lớp này.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm_client import LLMClient
from .retriever import Retriever


@dataclass
class Verdict:
    """Kết quả phán quyết."""

    label: str  # "malicious" | "benign"
    reason: str
    evidence: list[str]  # các mẩu tri thức/đoạn code làm căn cứ


class MaliciousCodeDetector:
    """Phán quyết bằng RAG: truy xuất bằng chứng rồi hỏi LLM.

    Chế độ ``use_rag=False`` cho phép so sánh baseline no-RAG.
    """

    def __init__(self, retriever: Retriever, llm: LLMClient, use_rag: bool = True):
        self._retriever = retriever
        self._llm = llm
        self._use_rag = use_rag

    def _build_prompt(self, code: str) -> str:
        context = ""
        if self._use_rag:
            docs = self._retriever.retrieve(code, top_k=5)
            context = "\n\n".join(f"- {d.text}" for d in docs)
        # P4 sẽ hoàn thiện prompt engineering ở đây.
        return (
            "Bạn là chuyên gia bảo mật. Xác định đoạn code sau là 'malicious' hay "
            "'benign', nêu lý do và trích dòng đáng ngờ.\n"
            f"Bằng chứng liên quan:\n{context}\n\n"
            f"Code cần kiểm tra:\n{code}"
        )

    def detect(self, code: str) -> Verdict:
        """STUB: P4 phân tích đầu ra LLM để trả về Verdict có cấu trúc."""
        raise NotImplementedError("P4 sẽ hiện thực phần phân tích đầu ra LLM.")
