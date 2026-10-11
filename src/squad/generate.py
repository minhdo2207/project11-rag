"""Bước 3: sinh câu trả lời bằng LLM nhỏ qua Ollama.

    c0 = không RAG, c1 = RAG với 5 đoạn đứng đầu của bước 2.

C2–C6 dùng lại đúng câu trả lời của c1, chỉ khác bước quyết định, nên không cần sinh lại.

    python -m src.squad.generate --model phi3 --config c1 --split dev --limit 200

Chạy lại lệnh cũ sẽ tiếp tục từ chỗ dừng (bỏ qua các qid đã có).
Đầu ra: results/squad/gen_<model>_<config>_<split>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from src.squad.common import MODELS, OUT, PROC, SEED, read_jsonl

N_CTX = 5  # số đoạn đưa vào prompt
IDK = "I don't know"

SYSTEM_C0 = (
    "You are a question answering assistant. Reply with the answer only: a short phrase of "
    "at most a few words. Never write a full sentence and never explain. "
    f"If you do not know the answer, reply exactly: {IDK}"
)
SYSTEM_C1 = (
    "You are a question answering assistant. Answer the question using only the provided "
    "context. Reply with the answer only: a short phrase copied from the context, at most a "
    "few words. Never write a full sentence and never explain. "
    f"If the context does not contain the answer, reply exactly: {IDK}"
)

# Ví dụ mẫu tự viết, không lấy từ SQuAD, dùng chung cho mọi model và mọi câu hỏi.
SHOT_CTX = (
    "[1] Lake Baikal\nLake Baikal is a rift lake in southern Siberia, Russia. With a maximum "
    "depth of 1,642 metres it is the deepest lake in the world, and it holds about 23% of the "
    "world's fresh surface water.\n\n"
    "[2] Lake Baikal\nThe lake is home to the Baikal seal, the only exclusively freshwater seal "
    "species. Russian explorer Kurbat Ivanov reached the lake in 1643."
)
SHOTS_C1 = [
    ("How deep is Lake Baikal?", "1,642 metres"),
    ("Who reached Lake Tanganyika in 1643?", IDK),
    ("What seal species lives in the lake?", "Baikal seal"),
    ("How much of the world's salt water does Lake Baikal hold?", IDK),
]
SHOTS_C0 = [
    ("What is the capital of Australia?", "Canberra"),
    ("What was the name of Isaac Newton's third cousin's dog?", IDK),
    ("Who wrote the novel Pride and Prejudice?", "Jane Austen"),
]


def build_messages(config: str, question: str, passages: list[str]) -> list[dict]:
    """Dựng hội thoại: system prompt, các ví dụ mẫu theo lượt user/assistant, rồi câu hỏi thật."""
    if config == "c0":
        msgs = [{"role": "system", "content": SYSTEM_C0}]
        for q, a in SHOTS_C0:
            msgs += [
                {"role": "user", "content": f"Question: {q}"},
                {"role": "assistant", "content": a},
            ]
        return msgs + [{"role": "user", "content": f"Question: {question}"}]
    msgs = [{"role": "system", "content": SYSTEM_C1}]
    for q, a in SHOTS_C1:
        msgs += [
            {"role": "user", "content": f"Context:\n{SHOT_CTX}\n\nQuestion: {q}"},
            {"role": "assistant", "content": a},
        ]
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return msgs + [{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}]


def chat(model: str, messages: list[dict], host: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": SEED, "num_predict": 64, "num_ctx": 4096},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.loads(r.read())
    return {
        "answer": res["message"]["content"].strip(),
        "wall_s": round(time.perf_counter() - t0, 3),
        # Thời gian do Ollama tự đo cho riêng request này (ns -> s).
        "gen_s": round((res.get("prompt_eval_duration", 0) + res.get("eval_duration", 0)) / 1e9, 3),
        "prompt_tokens": res.get("prompt_eval_count"),
        "output_tokens": res.get("eval_count"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--config", choices=["c0", "c1"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu sau khi xáo theo seed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    qs = read_jsonl(PROC / f"questions_{args.split}.jsonl")
    # thứ tự cố định, nên --limit N luôn là tập con của --limit M > N
    random.Random(SEED).shuffle(qs)
    if args.limit:
        qs = qs[: args.limit]
    kb = {p["para_id"]: p["text"] for p in read_jsonl(PROC / "kb.jsonl")}
    hits = {
        r["qid"]: [h["para_id"] for h in r["hits"][:N_CTX]]
        for r in read_jsonl(OUT / f"retrieval_{args.split}.jsonl")
    }

    out_path = OUT / f"gen_{args.model}_{args.config}_{args.split}.jsonl"
    done = {r["qid"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [q for q in qs if q["qid"] not in done]
    print(f"{out_path.name}: đã có {len(done)}, còn {len(todo)}")

    def run(q: dict) -> dict:
        ctx_ids = hits[q["qid"]] if args.config == "c1" else []
        messages = build_messages(args.config, q["question"], [kb[i] for i in ctx_ids])
        res = chat(MODELS[args.model], messages, args.host)
        return {
            "qid": q["qid"],
            "model": args.model,
            "config": args.config,
            "ctx_ids": ctx_ids,
            **res,
        }

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        for n, row in enumerate(ex.map(run, todo), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {n / (time.time() - t0):.1f} câu/s", flush=True)


if __name__ == "__main__":
    main()
