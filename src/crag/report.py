"""Dựng bảng kết quả từ các file đã chấm.

Số đo duy nhất: **bộ ba nhãn của CRAG** — accuracy / hallucination / missing / truthfulness.
   Đây là số đo gốc của benchmark, so thẳng được với Bảng 11 của bài (Bảng 5 chỉ có
   hai model lớn; dòng Llama 3 8B nằm ở Bảng 11, Phụ lục A.3.2).

    python -m src.crag.report

Đầu ra: results/crag/metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRAG = ROOT / "results" / "crag"

# Tất cả đều là bản chạy với trần đầu ra 75 token — đúng bằng cửa sổ chấm của CRAG.
# Bản 50 token cũ vẫn còn trong results/crag/ nhưng KHÔNG dùng nữa: nó cắt thấp hơn
# cửa sổ chấm nên thổi phồng tỷ lệ bịa, và thổi phồng không đều giữa các bậc.
STEM = {
    "B0": "gen_B0_llama3-8b-instruct-fp16_none_seed1_p75",
    "B1": "gen_B1_llama3-8b-instruct-fp16_text_seed1_p75",
    "B1.1": "gen_B1.1_llama3-8b-instruct-fp16_snippet_seed1_p75",
    "B1.2": "gen_B1.2_llama3-8b-instruct-fp16_snippet_seed1_p75",
    "B2": "gen_B2_llama3-8b-instruct-fp16_chunk_seed1_p75",
    "B3": "gen_B3_llama3-8b-instruct-fp16_chunk_seed1_p75",
    "B4": "gen_B4_llama3-8b-instruct-fp16_chunk_seed1_p75",
    "B5": "gen_B5_llama3-8b-instruct-fp16_chunk_seed1_p75",
}
RUNS = {k: CRAG / f"{v}.scored_crag.jsonl" for k, v in STEM.items()}
RUNS_THREE = {k: CRAG / f"{v}.scored_three.jsonl" for k, v in STEM.items()}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def crag_metrics(rows: list[dict]) -> dict:
    """Ba nhãn gốc của benchmark."""
    n = len(rows)
    acc = sum(r["label"] == "accurate" for r in rows) / n
    hal = sum(r["label"] == "hallucinated" for r in rows) / n
    mis = sum(r["label"] == "missing" for r in rows) / n
    return {
        "n": n,
        "accuracy": 100 * acc,
        "hallucination": 100 * hal,
        "missing": 100 * mis,
        "truthfulness": 100 * (acc - hal),
    }


def main() -> None:
    rows = {tag: load(path) for tag, path in RUNS.items()}
    three = {tag: crag_metrics(r) for tag, r in rows.items()}

    three_mode = {k: crag_metrics(load(p)) for k, p in RUNS_THREE.items()}
    out = {"crag_three_label": three, "doi_chung_3_nhan": three_mode}
    (CRAG / "metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
