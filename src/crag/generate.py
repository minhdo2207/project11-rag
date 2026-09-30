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

# Bài dùng cửa sổ 4K token cho họ Llama 3 Instruct và GPT-4 Turbo
REF_TOKEN_BUDGET = 4000
# Bậc dùng tài liệu; B0 là bậc duy nhất không dùng
RAG_CONFIGS = ("B1", "B1.1", "B1.2", "B2", "B3", "B4", "B5")
# B1.1 chỉ là tên khác của B1; B1.2 dùng dữ liệu 50 trang
PAGE_CONFIGS = ("B1", "B1.1", "B1.2")
DATA_TASK3 = ROOT / "data" / "crag_task3_public_test" / "questions.jsonl"
# Bậc đọc file truy hồi thay vì nối thẳng cả trang
CHUNK_CONFIGS = ("B2", "B3", "B4", "B5")
# Bậc đánh số tài liệu và bắt trích dẫn
CITE_CONFIGS = ("B4", "B5")
_CITATION = re.compile(r"\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
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
        "--retrieval", type=Path, default=ROOT / "data" / "crag_public_test" / "retrieval.jsonl"
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="phải đủ chứa 4000 token tài liệu + prompt, nếu không Ollama tự cắt",
    )
    args = parser.parse_args()

    data_file = DATA_TASK3 if args.config == "B1.2" else DATA
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
            n_used, gated, best = 0, False, None
            if args.config == "B0":
                prompt = PROMPT_B0.format(query=item["query"], query_time=item["query_time"])
                n_ref = 0
            elif args.config in PAGE_CONFIGS:
                refs, n_ref = build_references(item, budget, args.ref_source)
                prompt = PROMPT_B1.format(
                    query=item["query"], query_time=item["query_time"], references=refs
                )
            else:
                chunks = retrieval.get(item["interaction_id"], [])
                best = max((c["rerank"] for c in chunks), default=-1.0)
                refs, n_ref, n_used = build_from_chunks(
                    chunks, budget, args.config, args.keep_threshold
                )
                template = PROMPT_B4 if args.config in CITE_CONFIGS else PROMPT_B1
                prompt = template.format(
                    query=item["query"], query_time=item["query_time"], references=refs
                )
                # B5: tài liệu quá yếu thì từ chối luôn, khỏi tốn một lượt sinh
                gated = args.config == "B5" and best < args.gate_threshold

            t0 = time.time()
            if gated:
                resp = {"response": "I don't know", "eval_count": 0}
            else:
                resp = client.generate(
                    model=args.model, prompt=prompt, options=options, raw=args.raw
                )
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
