"""Chọn ngưỡng cho B3 và B5 bằng số liệu, không chọn bằng cảm tính.

Ý tưởng: với mỗi câu hỏi, kiểm tra xem đáp án chuẩn có THỰC SỰ nằm trong các đoạn đã
truy hồi hay không. Đó là thước đo "tài liệu có đủ để trả lời" mà không cần chấm tay.
Sau đó xem xác suất đó biến thiên thế nào theo điểm xếp hạng lại cao nhất của câu.

Ngưỡng cổng từ chối của B5 nên đặt ở chỗ xác suất này sụp xuống: dưới ngưỡng đó, gọi
model gần như chỉ sinh ra câu bịa.

    python -m src.crag.calibrate
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "crag_public_test"

_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def norm(text) -> str:
    text = str(text).lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def main() -> None:
    questions = {
        json.loads(line)["interaction_id"]: json.loads(line)
        for line in (DATA / "questions.jsonl").open(encoding="utf-8")
    }
    rows = []
    for line in (DATA / "retrieval.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        item = questions[r["interaction_id"]]
        golds = [str(item["answer"])] + [str(a) for a in item["alt_ans"] if a != ""]
        golds = [norm(g) for g in golds if norm(g) and norm(g) != "invalid question"]
        best = max((c["rerank"] for c in r["chunks"]), default=-1.0)
        blob = norm(" ".join(c["text"] for c in r["chunks"]))
        rows.append(
            {
                "best": best,
                "n_chunks": len(r["chunks"]),
                "has_answer": bool(golds) and any(g in blob for g in golds),
                "scorable": bool(golds),
                "type": item["question_type"],
            }
        )

    print(f"{len(rows)} câu · {sum(r['scorable'] for r in rows)} câu có đáp án so khớp được\n")

    edges = [-1.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.70, 0.90, 1.01]
    print(f"{'khoảng điểm cao nhất':<26}{'số câu':>8}{'có đáp án trong tài liệu':>28}")
    for lo, hi in zip(edges, edges[1:], strict=False):
        sub = [r for r in rows if lo <= r["best"] < hi and r["scorable"]]
        if not sub:
            continue
        pct = 100 * sum(r["has_answer"] for r in sub) / len(sub)
        bar = "█" * round(pct / 3)
        print(f"  [{lo:>5.2f} – {hi:<5.2f})       {len(sub):>8}{pct:>22.1f}%  {bar}")

    print(
        f"\n{'ngưỡng':<10}{'số câu bị chặn':>16}{'% toàn tập':>13}"
        f"{'trong số bị chặn: có đáp án':>30}"
    )
    for tau in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        blocked = [r for r in rows if r["best"] < tau]
        sc = [r for r in blocked if r["scorable"]]
        pct_has = 100 * sum(r["has_answer"] for r in sc) / len(sc) if sc else 0.0
        print(
            f"  {tau:<8.2f}{len(blocked):>16}{100 * len(blocked) / len(rows):>12.1f}%{pct_has:>29.1f}%"
        )


if __name__ == "__main__":
    main()
