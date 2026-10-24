"""Khâu dựng ngữ cảnh: chọn tài liệu nào đưa vào prompt, trong ngân sách 4.000 token.

Tách khỏi ``src.crag.generate`` để ba khâu truy hồi → sinh → từ chối có ranh giới rõ:

    src.crag.retrieve   chia đoạn, tìm, xếp hạng lại  →  retrieval.jsonl
    src.crag.context    chọn đoạn/trang, cắt theo ngân sách  →  chuỗi tài liệu + điểm
    src.crag.abstain    quyết định có từ chối không (trước và sau khi sinh)
    src.crag.generate   dựng prompt, gọi model, ghi file
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Bài dùng cửa sổ 4K token cho họ Llama 3 Instruct và GPT-4 Turbo
REF_TOKEN_BUDGET = 4000
# Bậc dùng tài liệu; B0 là bậc duy nhất không dùng
RAG_CONFIGS = ("B1", "B1.1", "B1.2", "B2", "B3", "B4", "B5")
# B1.1 chỉ là tên khác của B1; B1.2 dùng dữ liệu 50 trang
PAGE_CONFIGS = ("B1", "B1.1", "B1.2")
# Bậc đọc file truy hồi thay vì nối thẳng cả trang
CHUNK_CONFIGS = ("B2", "B3", "B4", "B5")
# Bậc đánh số tài liệu và bắt trích dẫn
CITE_CONFIGS = ("B4", "B5")
# Bản sao tokenizer Llama 3 không bị khoá tải (trọng số gốc của Meta cần xin quyền)
TOKENIZER_REPO = "unsloth/llama-3-8b-Instruct"


class Budget:
    """Cắt phần tài liệu tham chiếu theo đúng số token, như bài mô tả."""

    def __init__(self, repo: str = TOKENIZER_REPO) -> None:
        from transformers import AutoTokenizer  # nhập tại chỗ: chỉ B1 mới cần

        self.tok = AutoTokenizer.from_pretrained(repo)

    def clip(self, text: str, budget: int) -> tuple[str, int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return text, len(ids)
        return self.tok.decode(ids[:budget]), budget


def build_references(item: dict, budget: Budget, source: str) -> tuple[str, int]:
    """Nối các trang theo THỨ TỰ GỐC cho tới khi đầy cửa sổ token.

    ``source='text'``: dùng nội dung trang đã tách khỏi HTML.
    ``source='snippet'``: dùng đoạn tóm tắt ngắn mà công cụ tìm kiếm trả về.
    Bài viết "webpage snippets" nên hai cách đều có thể là cách bài đã dùng — chạy cả hai
    rồi xem cách nào ra gần Bảng 5 hơn (xem docs/crag_quy_trinh_danh_gia.md §1).
    """
    parts, used = [], 0
    for page in item["pages"]:
        raw = page["text"] if source == "text" else page["page_snippet"]
        if not raw.strip():
            continue
        clipped, n = budget.clip(raw, REF_TOKEN_BUDGET - used)
        if n == 0:
            break
        parts.append(f"- {clipped.strip()}")
        used += n
        if used >= REF_TOKEN_BUDGET:
            break
    return "\n".join(parts), used


def build_from_chunks(
    chunks: list[dict], budget: Budget, config: str, keep_threshold: float
) -> tuple[str, int, int]:
    """Lấp ngân sách bằng các đoạn đã xếp hạng lại, tốt nhất trước.

    Trả về (chuỗi tài liệu, số token đã dùng, số đoạn đã dùng). B3 trở lên bỏ những
    đoạn có điểm dưới ngưỡng — đây chính là cơ chế được kỳ vọng kéo tỷ lệ bịa xuống,
    và nó CÓ THỂ trả về chuỗi rỗng khi không đoạn nào đủ điểm.
    """
    if config == "B2":
        ranked = sorted(chunks, key=lambda c: -c["rrf"])
    else:
        ranked = [
            c for c in sorted(chunks, key=lambda c: -c["rerank"]) if c["rerank"] >= keep_threshold
        ]

    parts, used = [], 0
    for c in ranked:
        clipped, n = budget.clip(c["text"], REF_TOKEN_BUDGET - used)
        if n == 0:
            break
        label = f"[{len(parts) + 1}] " if config in CITE_CONFIGS else "- "
        parts.append(label + clipped.strip())
        used += n
        if used >= REF_TOKEN_BUDGET:
            break
    return "\n".join(parts), used, len(parts)


def build_context(
    item: dict,
    config: str,
    budget: Budget | None,
    retrieval: dict[str, list[dict]],
    ref_source: str,
    keep_threshold: float,
) -> dict:
    """Ngữ cảnh cho một câu ở một bậc.

    Returns:
        ``refs`` (chuỗi tài liệu, rỗng ở B0), ``n_ref`` (số token), ``n_used`` (số đoạn,
        0 ở bậc dùng trang), ``best`` (điểm xếp hạng lại cao nhất, None ở bậc không chia đoạn).
    """
    if config not in RAG_CONFIGS:
        return {"refs": "", "n_ref": 0, "n_used": 0, "best": None}
    if config in PAGE_CONFIGS:
        refs, n_ref = build_references(item, budget, ref_source)
        return {"refs": refs, "n_ref": n_ref, "n_used": 0, "best": None}
    chunks = retrieval.get(item["interaction_id"], [])
    best = max((c["rerank"] for c in chunks), default=-1.0)
    refs, n_ref, n_used = build_from_chunks(chunks, budget, config, keep_threshold)
    return {"refs": refs, "n_ref": n_ref, "n_used": n_used, "best": best}
