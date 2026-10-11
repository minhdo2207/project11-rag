"""Cấu hình và hàm dùng chung cho luồng SQuAD 2.0 + guardrail (src/squad)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "squad" / "raw"
PROC = ROOT / "data" / "squad" / "processed"
INDEX = ROOT / "data" / "squad" / "index"
OUT = ROOT / "results" / "squad"

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_FILE = RAW / "dev-v2.0.json"

SEED = 42
N_ARTICLES = 10  # số bài lấy từ 35 bài của SQuAD 2.0 dev
N_DEV = 3  # 3 bài dò ngưỡng, 7 bài còn lại để báo cáo

EMBED_MODEL = "BAAI/bge-m3"
TOP_K = 10  # số đoạn lưu lại cho mỗi câu hỏi
CAND_K = 50  # số ứng viên mỗi retriever đưa vào RRF
RRF_K = 60

# Tên ngắn dùng trong tên file kết quả → tên model trên Ollama
MODELS = {
    "phi3": "phi3:3.8b-mini-4k-instruct-q4_K_M",
    "llama3": "llama3:8b-instruct-q4_K_M",
}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def tokenize(text: str) -> list[str]:
    """Tách từ cho BM25: chữ thường, chỉ giữ ký tự chữ và số."""
    return re.findall(r"\w+", text.lower())


def pick_gpu() -> None:
    """Chọn GPU đang dùng ít bộ nhớ nhất, trừ khi CUDA_VISIBLE_DEVICES đã được đặt."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = [tuple(int(x) for x in line.split(",")) for line in out.strip().splitlines()]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(min(used, key=lambda t: t[1])[0])
    except Exception:
        pass
