"""Khâu từ chối: quyết định trả "I don't know" thay cho câu trả lời của model.

Hai điểm quyết định, gọi từ ``src.crag.generate``:

    decide_before   trước khi sinh, chỉ nhìn ngữ cảnh — cổng B5: đoạn tốt nhất quá yếu
                    thì từ chối luôn, khỏi tốn một lượt sinh
    decide_after    sau khi sinh, nhìn được cả câu trả lời — chỗ dành cho các chính sách
                    cần đáp án (so với đáp án closed-book kiểu P3, bộ ước lượng kiểu SABER).
                    Hiện chưa bậc nào dùng nên luôn trả False.

``ctx`` là dict do ``src.crag.context.build_context`` trả về.
"""

from __future__ import annotations

# Bậc có cổng từ chối trước khi sinh
GATE_CONFIGS = ("B5",)


def decide_before(config: str, ctx: dict, gate_threshold: float) -> bool:
    """True nếu nên từ chối mà không gọi model."""
    return config in GATE_CONFIGS and ctx["best"] is not None and ctx["best"] < gate_threshold


def decide_after(config: str, ctx: dict, answer: str) -> bool:
    """True nếu nên thay câu trả lời đã sinh bằng "I don't know"."""
    return False
