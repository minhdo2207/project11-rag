"""Chạy detector trên một file code: ingest -> nhúng -> truy xuất -> hỏi LLM -> in verdict.

Ví dụ:
    python scripts/run_detector.py suspicious_setup.py --mode no_rag
    python scripts/run_detector.py suspicious_setup.py --mode rag --ingest
    python scripts/run_detector.py setup.py --config config.yaml --model qwen2.5:7b

Cần Ollama đang chạy và đã pull model trong config (llm.model, embedding.model).
Chế độ rag cần knowledge base có dữ liệu: tải bằng scripts/fetch_data.py rồi thêm --ingest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.detector import MaliciousCodeDetector  # noqa: E402
from src.embedding import build_embedder  # noqa: E402
from src.ingest import ingest_all  # noqa: E402
from src.llm_client import build_llm_client  # noqa: E402
from src.repository.factory import build_repository  # noqa: E402
from src.retriever import build_retriever, embed_missing  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", type=Path, help="file code cần kiểm tra")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["no_rag", "rag", "rag_fewshot"], help="ghi đè pipeline.mode"
    )
    parser.add_argument("--model", help="ghi đè llm.model")
    parser.add_argument("--data-dir", default="data", help="thư mục data cho --ingest")
    parser.add_argument("--ingest", action="store_true", help="nạp data vào KB trước khi chạy")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg.data.setdefault("pipeline", {})["mode"] = args.mode
    if args.model:
        cfg.data.setdefault("llm", {})["model"] = args.model

    retriever = None
    if cfg.mode != "no_rag":
        repo = build_repository(cfg)
        if args.ingest:
            print(f"ingest: {ingest_all(repo, args.data_dir)}")
        if repo.count() == 0:
            print("Knowledge base đang trống: chạy scripts/fetch_data.py rồi thêm --ingest.")
            return 1
        embedder = build_embedder(cfg)
        print(f"nhúng {embed_missing(repo, embedder)} tài liệu mới ({repo.count()} tổng)")
        retriever = build_retriever(cfg, repo, embedder)

    detector = MaliciousCodeDetector.from_config(cfg, retriever, build_llm_client(cfg))
    verdict = detector.detect(args.file.read_text(encoding="utf-8", errors="replace"))

    print(f"\nmode:   {cfg.mode}")
    print(f"label:  {verdict.label}")
    print(f"reason: {verdict.reason}")
    for line in verdict.suspicious_lines:
        print(f"  suspicious> {line}")
    for item in verdict.evidence:
        print(f"  evidence>   {item}")
    return 0 if verdict.label != "unknown" else 2


if __name__ == "__main__":
    raise SystemExit(main())
