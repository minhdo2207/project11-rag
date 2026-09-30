"""Nạp CRAG task 3 — 50 trang mỗi câu — cho bậc B1.2.

Khác task 1&2 ở hai chỗ, nên phải viết riêng:
  · dữ liệu nằm trong .tar.bz2 chia làm 4 phần, phải đọc luồng (giải nén ra đĩa tốn ~40 GB)
  · bản ghi KHÔNG có trường `split`, nên lọc tập public test bằng interaction_id của
    1.335 câu đã nạp từ task 1&2. Nhờ vậy hai bậc dùng đúng cùng bộ câu hỏi, so theo cặp được.

Mặc định chỉ giữ page_snippet: đó là thứ baseline của bài dùng, và nó tránh phải phân tích
66.750 trang HTML. Thêm --with-text nếu sau này muốn chạy B2/B3 trên 50 trang.

    python -m src.crag.ingest_task3

Đầu ra: data/crag_task3_public_test/questions.jsonl + stats.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import time
from pathlib import Path

from .ingest import html_to_text

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "crag"
REF = ROOT / "data" / "crag_public_test" / "questions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parts", default="crag_task_3_dev_v5.tar.bz2.part*")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "crag_task3_public_test")
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="tách cả chữ khỏi HTML (chậm, ~580 MB); mặc định chỉ giữ snippet",
    )
    args = parser.parse_args()

    ref = {}
    for line in REF.open(encoding="utf-8"):
        r = json.loads(line)
        ref[r["interaction_id"]] = r
    print(f"cần tìm {len(ref)} câu của tập public test")

    parts = sorted(RAW.glob(args.parts))
    if not parts:
        raise SystemExit(f"không thấy phần nào khớp {args.parts} trong {RAW}")
    print(f"đọc luồng {len(parts)} phần: {', '.join(p.name for p in parts)}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "questions.jsonl"
    n_seen = n_kept = 0
    page_counts, empty_snip = [], 0
    started = time.time()

    # nối 4 phần bằng cat rồi đưa vào tarfile ở chế độ luồng
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    with (
        tarfile.open(fileobj=cat.stdout, mode="r|bz2") as tar,
        out_file.open("w", encoding="utf-8") as sink,
    ):
        for member in tar:
            if not member.name.endswith(".jsonl"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            print(f"  · {member.name} ({member.size / 1e9:.2f} GB)", flush=True)
            for raw in handle:
                n_seen += 1
                rec = json.loads(raw)
                base = ref.get(rec["interaction_id"])
                if base is None:
                    continue
                pages = []
                for pg in rec.get("search_results", []):
                    item = {
                        "page_name": pg.get("page_name", ""),
                        "page_url": pg.get("page_url", ""),
                        "page_snippet": pg.get("page_snippet", "") or "",
                        "page_last_modified": pg.get("page_last_modified", ""),
                    }
                    if args.with_text:
                        item["text"] = html_to_text(pg.get("page_result", ""))
                    pages.append(item)
                empty_snip += sum(1 for p in pages if not p["page_snippet"].strip())
                page_counts.append(len(pages))
                # đáp án lấy từ bản task 1&2 để chắc chắn hai bậc chấm trên cùng chuẩn
                sink.write(
                    json.dumps(
                        {
                            "interaction_id": rec["interaction_id"],
                            "query": base["query"],
                            "query_time": base["query_time"],
                            "answer": base["answer"],
                            "alt_ans": base["alt_ans"],
                            "domain": base["domain"],
                            "question_type": base["question_type"],
                            "static_or_dynamic": base["static_or_dynamic"],
                            "popularity": base["popularity"],
                            "pages": pages,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_kept += 1
                if n_kept % 200 == 0:
                    print(
                        f"    giữ {n_kept}/{len(ref)} · quét {n_seen} · "
                        f"{(time.time() - started) / 60:.0f} phút",
                        flush=True,
                    )
            if n_kept >= len(ref):
                break
    cat.stdout.close()
    cat.terminate()

    stats = {
        "n_questions": n_kept,
        "n_records_scanned": n_seen,
        "n_pages_total": sum(page_counts),
        "pages_per_question_min_max": [min(page_counts), max(page_counts)] if page_counts else None,
        "n_snippets_empty": empty_snip,
        "with_text": args.with_text,
        "minutes": round((time.time() - started) / 60, 1),
    }
    (args.out / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if n_kept < len(ref):
        print(f"CẢNH BÁO: chỉ tìm thấy {n_kept}/{len(ref)} câu")


if __name__ == "__main__":
    main()
