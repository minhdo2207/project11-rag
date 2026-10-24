"""Sinh câu trả lời cho CRAG theo đúng cấu hình trong bài báo (Phụ lục A.3.1).

Bậc thang, mỗi bậc thêm ĐÚNG MỘT thứ so với bậc dưới:
    B0 — chỉ hỏi model, không đưa tài liệu (dòng "LLM only" của Bảng 5)
    B1.1 — Task 1: 5 trang/câu, nối theo thứ tự gốc tới khi đầy cửa sổ token
    B1.2 — Task 3: 50 trang/câu, mọi thứ khác giữ y hệt B1.1 (bỏ phần KG của bài)
    B2 — thay cách chọn: chia đoạn, tìm bằng từ khoá + vector, gộp bằng RRF
    B3 — thêm xếp hạng lại và cắt ngưỡng, bỏ đoạn lạc đề
    B4 — thêm ràng buộc trích dẫn: không dẫn được nguồn thì phải nói không biết
    B5 — thêm cổng từ chối ở mức hệ thống: tài liệu quá yếu thì không gọi model

B2–B5 đọc trước kết quả của ``src.crag.retrieve``; B0/B1 không cần.
Ngân sách tài liệu giữ nguyên 4.000 token ở mọi bậc, nên thứ duy nhất thay đổi là
NỘI DUNG lấp đầy ngân sách đó — trừ B3 trở lên, nơi phép cắt ngưỡng có thể dùng ít hơn.

    python -m src.crag.generate --config B0 --model llama3:8b-instruct-fp16
    python -m src.crag.generate --config B3 --model llama3:8b-instruct-fp16 --limit 50

Đầu ra: results/crag/gen_<config>_<model>_<ref>_seed<seed>.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import ollama

from src.crag.abstain import decide_after, decide_before
from src.crag.context import (
    CHUNK_CONFIGS,
    CITE_CONFIGS,
    PAGE_CONFIGS,
    RAG_CONFIGS,
    Budget,
    build_context,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "crag_public_test" / "questions.jsonl"
OUT_DIR = ROOT / "results" / "crag"

# Nguyên văn Phụ lục A.3.1 của bài CRAG (arXiv 2406.04744)
PROMPT_B0 = """You are given a Question and the time when it was asked in the Pacific Time Zone (PT), referred to as "Query Time". The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".

### Question
{query}

### Query Time
{query_time}

### Answer
"""

PROMPT_B1 = """You are given a Question, References and the time when it was asked in the Pacific Time Zone (PT), referred to as "Query Time". The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".

### Question
{query}

### Query Time
{query_time}

### References
{references}

### Answer
"""

# Bậc B4/B5: giữ nguyên hai quy tắc gốc, thêm ràng buộc phải dẫn nguồn.
PROMPT_B4 = """You are given a Question, numbered References and the time when it was asked in the Pacific Time Zone (PT), referred to as "Query Time". The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". The references may or may not help answer the question. Your task is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer "invalid question".
2. If you are uncertain or don't know the answer, respond with "I don't know".
3. Answer only from the references. End your answer with the number of the reference you used, in square brackets, for example: Paris [2].
4. If no reference supports an answer, respond with "I don't know".

### Question
{query}

### Query Time
{query_time}

### References
{references}

### Answer
"""

DATA_TASK3 = ROOT / "data" / "crag_task3_public_test" / "questions.jsonl"
_CITATION = re.compile(r"\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", choices=["B0", "B1", "B1.1", "B1.2", "B2", "B3", "B4", "B5"], required=True
    )
    parser.add_argument("--model", default="llama3:8b-instruct-fp16")
    parser.add_argument("--ref-source", choices=["text", "snippet"], default="text")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="bài KHÔNG công bố giá trị này; 0,1 lấy theo mã mẫu của repo",
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--num-predict",
        type=int,
        default=50,
        help="số token tối đa sinh ra; bài chấm trên 75 token đầu nên "
        "đặt 75 thì phép cắt của bộ chấm mới có nghĩa",
    )
    parser.add_argument(
        "--raw", action="store_true", help="gửi prompt thẳng, KHÔNG bọc template chat của Llama 3"
    )
    parser.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu")
    parser.add_argument("--host", default=None, help="vd. http://127.0.0.1:11435")
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=0.10,
        help="B3+: bỏ đoạn có điểm xếp hạng lại thấp hơn ngưỡng này",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.10,
        help="B5: đoạn tốt nhất dưới ngưỡng này thì từ chối, không gọi model",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="file câu hỏi khác mặc định, vd. data/crag_canary/questions.jsonl; "
        "tên thư mục chứa nó được gắn vào tên file đầu ra",
    )
    parser.add_argument(
        "--retrieval", type=Path, default=ROOT / "data" / "crag_public_test" / "retrieval.jsonl"
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="phải đủ chứa 4000 token tài liệu + prompt, nếu không Ollama tự cắt",
    )
    args = parser.parse_args()

    data_file = args.data or (DATA_TASK3 if args.config == "B1.2" else DATA)
    if not data_file.exists():
        raise SystemExit(f"thiếu {data_file} — chạy bước nạp tương ứng trước")
    # bản nạp task 3 mặc định chỉ giữ snippet, không có chữ tách từ HTML
    if args.config == "B1.2" and args.ref_source == "text":
        raise SystemExit(
            "B1.2 chỉ có snippet; chạy lại với --ref-source snippet "
            "hoặc nạp lại task 3 kèm --with-text"
        )
    items = [json.loads(line) for line in data_file.open(encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]
    if args.config in PAGE_CONFIGS and items and "pages" not in items[0]:
        raise SystemExit(f"{data_file} không có trang gốc (vd. bộ canary) — chỉ chạy B0, B2–B5")

    budget = Budget() if args.config in RAG_CONFIGS else None
    retrieval = {}
    if args.config in CHUNK_CONFIGS:
        if not args.retrieval.exists():
            raise SystemExit(f"thiếu {args.retrieval} — chạy python -m src.crag.retrieve trước")
        retrieval = {
            r["interaction_id"]: r["chunks"]
            for r in map(json.loads, args.retrieval.open(encoding="utf-8"))
        }
        print(f"nạp truy hồi cho {len(retrieval)} câu")
    client = ollama.Client(host=args.host) if args.host else ollama
    options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
        "seed": args.seed,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace(":", "-")
    ref_tag = (
        args.ref_source
        if args.config in PAGE_CONFIGS
        else ("chunk" if args.config in CHUNK_CONFIGS else "none")
    )
    variant = ""
    if args.temperature != 0.1:
        variant += f"_t{args.temperature:g}"
    if args.raw:
        variant += "_raw"
    if args.num_predict != 50:
        variant += f"_p{args.num_predict}"
    if args.data and args.data.parent.name != DATA.parent.name:
        variant += f"_{args.data.parent.name}"
    out_file = OUT_DIR / f"gen_{args.config}_{tag}_{ref_tag}_seed{args.seed}{variant}.jsonl"

    done = set()
    if out_file.exists():  # chạy tiếp nếu lần trước bị ngắt
        done = {
            json.loads(line)["interaction_id"]
            for line in out_file.open(encoding="utf-8")
            if line.strip()
        }
        print(f"đã có {len(done)} câu, bỏ qua")

    started = time.time()
    with out_file.open("a", encoding="utf-8") as sink:
        for i, item in enumerate(items, 1):
            if item["interaction_id"] in done:
                continue
            ctx = build_context(
                item, args.config, budget, retrieval, args.ref_source, args.keep_threshold
            )
            if args.config == "B0":
                prompt = PROMPT_B0.format(query=item["query"], query_time=item["query_time"])
            else:
                template = PROMPT_B4 if args.config in CITE_CONFIGS else PROMPT_B1
                prompt = template.format(
                    query=item["query"], query_time=item["query_time"], references=ctx["refs"]
                )
            n_ref, n_used, best = ctx["n_ref"], ctx["n_used"], ctx["best"]
            gated = decide_before(args.config, ctx, args.gate_threshold)

            t0 = time.time()
            if gated:
                resp = {"response": "I don't know", "eval_count": 0}
            else:
                resp = client.generate(
                    model=args.model, prompt=prompt, options=options, raw=args.raw
                )
                if decide_after(args.config, ctx, resp["response"]):
                    resp = {**resp, "response": "I don't know"}
                    gated = True
            raw = resp["response"].strip()
            pred = _CITATION.sub("", raw).strip() if args.config in CITE_CONFIGS else raw
            sink.write(
                json.dumps(
                    {
                        "interaction_id": item["interaction_id"],
                        "query": item["query"],
                        "answer": item["answer"],
                        "alt_ans": item["alt_ans"],
                        "domain": item["domain"],
                        "question_type": item["question_type"],
                        "static_or_dynamic": item["static_or_dynamic"],
                        **({"canary_type": item["canary_type"]} if "canary_type" in item else {}),
                        "prediction": pred,
                        "prediction_raw": raw,
                        "n_ref_tokens": n_ref,
                        "n_ref_chunks": n_used,
                        "best_rerank": best,
                        "gated": gated,
                        "eval_count": resp.get("eval_count"),
                        "seconds": round(time.time() - t0, 2),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()
            if i % 25 == 0:
                rate = (time.time() - started) / i
                print(
                    f"{i}/{len(items)} · {rate:.1f} s/câu · còn ~{rate * (len(items) - i) / 60:.0f} phút",
                    flush=True,
                )

    print(f"xong: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
