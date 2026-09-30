"""Nạp CRAG task 1 & 2: lọc tập public test, tách chữ khỏi HTML, lưu vào DocumentStore.

Chạy một lần; các bước sinh câu trả lời sau đó chỉ đọc lại kết quả, không phải phân tích
lại HTML (mỗi câu có 5 trang, có trang tới hơn 250.000 ký tự).

    python -m src.crag.ingest --split 1 --out data/crag_public_test

Đầu ra:
    <out>/questions.jsonl   một dòng một câu hỏi, kèm văn bản đã tách của 5 trang
    <out>/stats.json        phân bố theo domain / question_type / static_or_dynamic
"""

from __future__ import annotations

import argparse
import bz2
import collections
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "data" / "raw" / "crag" / "crag_task_1_and_2_dev_v4.jsonl.bz2"

# Các thẻ chỉ chứa mã hoặc trang trí, bỏ trước khi lấy chữ
DROP_TAGS = ("script", "style", "noscript", "svg", "canvas", "iframe", "form")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


def html_to_text(html: str, max_chars: int = 400_000) -> str:
    """Lấy phần chữ người đọc được từ một trang HTML.

    ``max_chars`` chặn các trang khổng lồ: cắt phần HTML thô trước khi phân tích, vì
    BeautifulSoup trên trang vài MB rất chậm mà phần đuôi hầu như luôn là chân trang.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html[:max_chars], "lxml")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NL.sub("\n\n", text).strip()


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Đọc luồng file .jsonl.bz2 — không giải nén toàn bộ ra đĩa."""
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def convert(record: dict[str, Any]) -> dict[str, Any]:
    """Giữ các trường cần cho việc sinh và chấm; thay HTML bằng chữ đã tách."""
    pages = []
    for page in record.get("search_results", []):
        pages.append(
            {
                "page_name": page.get("page_name", ""),
                "page_url": page.get("page_url", ""),
                "page_snippet": page.get("page_snippet", ""),
                "page_last_modified": page.get("page_last_modified", ""),
                "text": html_to_text(page.get("page_result", "")),
            }
        )
    # alt_ans có khi là chuỗi JSON, có khi là list — chuẩn hoá về list
    alt = record.get("alt_ans", record.get("alternative_answers", []))
    if isinstance(alt, str):
        try:
            alt = json.loads(alt)
        except json.JSONDecodeError:
            alt = [alt] if alt.strip() else []
    return {
        "interaction_id": record["interaction_id"],
        "query": record["query"],
        "query_time": record["query_time"],
        "answer": record["answer"],
        "alt_ans": alt if isinstance(alt, list) else [],
        "domain": record.get("domain", ""),
        "question_type": record.get("question_type", ""),
        "static_or_dynamic": record.get("static_or_dynamic", ""),
        "popularity": record.get("popularity", ""),
        "split": record.get("split"),
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="1 = public test (bài báo dùng 1.335 câu), 0 = validation",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "crag_public_test")
    parser.add_argument("--limit", type=int, default=0, help="chỉ lấy N câu đầu, để chạy thử")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "questions.jsonl"
    counters = {
        k: collections.Counter()
        for k in ("domain", "question_type", "static_or_dynamic", "popularity")
    }
    n_kept = n_seen = n_empty_pages = 0
    page_chars = []

    with out_file.open("w", encoding="utf-8") as sink:
        for record in iter_records(args.src):
            n_seen += 1
            if record.get("split") != args.split:
                continue
            item = convert(record)
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_kept += 1
            for key, counter in counters.items():
                counter[item[key] or "(trống)"] += 1
            page_chars.extend(len(p["text"]) for p in item["pages"])
            n_empty_pages += sum(1 for p in item["pages"] if not p["text"])
            if n_kept % 100 == 0:
                print(f"đã nạp {n_kept} câu", flush=True)
            if args.limit and n_kept >= args.limit:
                break

    page_chars.sort()
    stats = {
        "source": str(args.src.relative_to(ROOT))
        if args.src.is_relative_to(ROOT)
        else str(args.src),
        "split": args.split,
        "n_records_scanned": n_seen,
        "n_questions": n_kept,
        "n_pages": len(page_chars),
        "n_pages_without_text": n_empty_pages,
        "page_text_chars_p50_p95_max": (
            [
                page_chars[len(page_chars) // 2],
                page_chars[int(len(page_chars) * 0.95)],
                page_chars[-1],
            ]
            if page_chars
            else None
        ),
        **{k: dict(c.most_common()) for k, c in counters.items()},
    }
    (args.out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
