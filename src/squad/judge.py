"""Giám khảo LLM cho các câu "unmatched": answerable, không từ chối, F1 với đáp án dưới ngưỡng.

Giám khảo chỉ thấy câu hỏi, các đáp án chuẩn và câu trả lời (giống bộ chấm tự động của CRAG),
trả về JSON {"correct": true|false}.

    python -m src.squad.judge --split dev --judge llama3.3:70b --limit 40

Chạy lại lệnh cũ sẽ tiếp tục từ chỗ dừng.
Đầu ra: results/squad/judge_<judge>_<model>_<config>_<split>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from src.squad.common import OUT, PROC, SEED, read_jsonl
from src.squad.label import judge_tag

SYSTEM = (
    "You are grading answers to reading-comprehension questions. You are given a question, "
    "one or more reference answers (all acceptable), and a prediction. "
    "Decide whether the prediction is correct.\n"
    "- Correct: the prediction states the same answer as at least one reference answer. "
    "Paraphrases, abbreviations, a more specific or less specific wording of the same fact, "
    "and extra supporting detail are fine. A prediction cut off mid-sentence is fine if the "
    "answer was already stated.\n"
    "- Incorrect: the prediction gives a different answer, contradicts the reference, lists "
    "several different candidate answers, or does not actually answer the question.\n"
    'Reply with JSON only: {"correct": true} or {"correct": false}'
)


def build_user_message(question: str, golds: list[str], prediction: str) -> str:
    refs = "\n".join(f"Reference answer {i}: {g}" for i, g in enumerate(golds, 1))
    return f"Question: {question}\n{refs}\nPrediction: {prediction}"


def parse_verdict(raw: str) -> bool | None:
    """Đọc {"correct": ...}; giám khảo trả sai định dạng thì để trống, không đoán."""
    try:
        return bool(json.loads(raw)["correct"])
    except Exception:
        return None


def judge_one(model: str, host: str, question: str, golds: list[str], prediction: str):
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_message(question, golds, prediction)},
            ],
            "options": {"temperature": 0, "seed": SEED, "num_predict": 16, "num_ctx": 2048},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.loads(r.read())["message"]["content"]
    return parse_verdict(raw), raw.strip(), round(time.perf_counter() - t0, 2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--judge", required=True, help="tên model trên Ollama")
    ap.add_argument("--runs", default="", help="lọc theo tên run, ví dụ 'c1' hoặc 'phi3_c1'")
    ap.add_argument("--limit", type=int, default=0, help="số câu mỗi run (sau khi xáo theo seed)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    tag = judge_tag(args.judge)
    qs = {q["qid"]: q for q in read_jsonl(PROC / f"questions_{args.split}.jsonl")}

    def run(r: dict) -> dict:
        q = qs[r["qid"]]
        verdict, raw, secs = judge_one(
            args.judge, args.host, q["question"], q["gold_answers"], r["answer"]
        )
        return {
            "qid": r["qid"],
            "correct": verdict,
            "raw": raw,
            "judge_s": secs,
            "best_f1": r["best_f1"],
        }

    for path in sorted(OUT.glob(f"labeled_*_{args.split}.jsonl")):
        name = path.stem.replace("labeled_", "")
        if args.runs not in name:
            continue
        rows = [r for r in read_jsonl(path) if r["label"] == "unmatched"]
        random.Random(SEED).shuffle(rows)
        if args.limit:
            rows = rows[: args.limit]
        out_path = OUT / f"judge_{tag}_{name}.jsonl"
        done = {r["qid"] for r in read_jsonl(out_path)} if out_path.exists() else set()
        todo = [r for r in rows if r["qid"] not in done]
        print(f"{out_path.name}: đã có {len(done)}, còn {len(todo)}", flush=True)

        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
            for n, row in enumerate(ex.map(run, todo), 1):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if n % 50 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.1f} s/câu", flush=True)


if __name__ == "__main__":
    main()
