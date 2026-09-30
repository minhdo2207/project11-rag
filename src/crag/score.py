"""Chấm câu trả lời CRAG. Tự viết vì mã chính thức không chạy được nguyên trạng.

Hai chế độ, chạy trên cùng một file sinh:

``--mode crag`` (tầng 1) — đúng cách của bài, để so với Bảng 5:
    dò chuỗi "i don't know" → khớp chính xác → giám khảo trả lời True/False.

``--mode three`` (tầng 2) — giám khảo tự chọn một trong ba nhãn, theo định nghĩa "missing"
    mở rộng của ban tổ chức cuộc thi cộng luật xử lý câu rào đón của SimpleQA.

    python -m src.crag.score --gen results/crag/gen_B0_....jsonl --mode crag --judge gemma4:31b

Đầu ra: <gen>.scored_<mode>.jsonl (từng câu) và một dòng tổng hợp trong results/crag/runs.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import string
import time
from pathlib import Path

import ollama

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "crag" / "runs.jsonl"

ACCURATE, HALLUCINATED, MISSING, UNPARSED = "accurate", "hallucinated", "missing", "unparsed"

# Prompt giám khảo nhị phân — Phụ lục A.2.2 của bài CRAG (bài chỉ công bố bản "tương tự")
JUDGE_BINARY = """# Task: You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement.
1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False".
3. If the Ground Truth is "invalid question", "Accuracy" is "True" only if the model prediction is exactly "invalid question".
# Output: Respond with only a single JSON string with an "Accuracy" field which is "True" or "False".

Question: {query}
Ground truth: {truths}
Prediction: {prediction}
"""

# Prompt ba nhãn — định nghĩa "missing" theo báo cáo tổng kết KDD Cup CRAG,
# luật xử lý câu rào đón và con số theo bộ chấm của SimpleQA.
JUDGE_THREE = """# Task: You are given a Question, a list of Ground Truth answers, and a model Prediction. Classify the Prediction into exactly one of three labels.

Labels:
- "missing": the Prediction declines to answer. This includes "I don't know", "I'm sorry I can't find...", an empty response, a system error, or the model asking the user to clarify the question.
- "accurate": the Prediction gives an answer that matches any of the Ground Truth answers. Hedging is allowed as long as the correct answer is fully stated. If the Ground Truth is "invalid question", the Prediction is accurate only if it also says the question is invalid.
- "hallucinated": the Prediction gives an answer that contradicts the Ground Truth, or answers something else. A wrong answer that is hedged (for example "I'm not sure, maybe X") is still hallucinated.

Rules:
- Only semantic meaning matters. Capitalization, punctuation and word order do not matter.
- For numbers, the Prediction must match the Ground Truth to the last significant figure.
- Do not punish typos in a person's name if it is clearly the same name.

# Output: Respond with only a single JSON object with one field "label" whose value is "accurate", "hallucinated" or "missing".

Question: {query}
Ground truth: {truths}
Prediction: {prediction}
"""

_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def norm(text) -> str:
    """Chuẩn hoá kiểu SQuAD: chữ thường, bỏ dấu câu, bỏ mạo từ, gộp khoảng trắng.

    Nhận cả số: 29/1335 câu của CRAG có đáp án là int hoặc float (vd. answer = 43).
    """
    text = str(text).lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def trim_75(text: str, tok) -> str:
    """Bài chấm trên 75 token đầu của câu trả lời."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:75]) if len(ids) > 75 else text


def ask_judge(client, model: str, prompt: str, field: str, allowed: set[str]) -> str | None:
    """Gọi giám khảo, đọc JSON trả về. Không đọc được thì trả None (ghi riêng, không tính là sai)."""
    resp = client.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0, "num_predict": 40, "num_ctx": 4096},
        format="json",
    )
    raw = resp["response"]
    value = ""
    try:
        data = json.loads(raw)
        # model hay đổi hoa thường của khoá ("Accuracy" thay vì "accuracy")
        value = next((str(v) for k, v in data.items() if k.lower() == field.lower()), "")
        value = value.strip().lower()
    except (json.JSONDecodeError, AttributeError):
        pass
    if value not in allowed:  # vớt vát khi model trả về văn xuôi lẫn JSON
        match = re.search(rf'"?{field}"?\s*:\s*"?(\w+)"?', raw, re.I)
        value = match.group(1).lower() if match else value
    return value if value in allowed else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gen", type=Path, required=True, help="file kết quả sinh")
    parser.add_argument("--mode", choices=["crag", "three"], default="crag")
    parser.add_argument("--judge", default="gemma4:31b", help="phải khác model đã sinh")
    parser.add_argument("--host", default=None)
    parser.add_argument("--tokenizer", default="unsloth/llama-3-8b-Instruct")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    client = ollama.Client(host=args.host) if args.host else ollama

    rows = [json.loads(line) for line in args.gen.open(encoding="utf-8") if line.strip()]
    out_file = args.gen.with_suffix(f".scored_{args.mode}.jsonl")
    n_judge_calls = 0
    started = time.time()

    with out_file.open("w", encoding="utf-8") as sink:
        for i, row in enumerate(rows, 1):
            pred = trim_75(row["prediction"], tok).strip()
            # ép về chuỗi: một số đáp án chuẩn của CRAG là số
            truths = [str(row["answer"])] + [str(a) for a in row.get("alt_ans", []) if a != ""]
            label, how = None, ""

            if args.mode == "crag":
                if "i don't know" in pred.lower():
                    label, how = MISSING, "chuỗi cố định"
                elif any(norm(pred) == norm(t) for t in truths):
                    label, how = ACCURATE, "khớp chính xác"
                else:
                    n_judge_calls += 1
                    verdict = ask_judge(
                        client,
                        args.judge,
                        JUDGE_BINARY.format(query=row["query"], truths=truths, prediction=pred),
                        "accuracy",
                        {"true", "false"},
                    )
                    label = (
                        UNPARSED
                        if verdict is None
                        else (ACCURATE if verdict == "true" else HALLUCINATED)
                    )
                    how = "giám khảo nhị phân"
            else:
                if any(norm(pred) == norm(t) for t in truths):
                    label, how = ACCURATE, "khớp chính xác"
                else:
                    n_judge_calls += 1
                    verdict = ask_judge(
                        client,
                        args.judge,
                        JUDGE_THREE.format(query=row["query"], truths=truths, prediction=pred),
                        "label",
                        {ACCURATE, HALLUCINATED, MISSING},
                    )
                    label = verdict or UNPARSED
                    how = "giám khảo ba nhãn"

            sink.write(
                json.dumps(
                    {**row, "prediction_trimmed": pred, "label": label, "decided_by": how},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if i % 100 == 0:
                print(f"{i}/{len(rows)} · {(time.time() - started) / i:.1f} s/câu", flush=True)

    labels = [json.loads(line)["label"] for line in out_file.open(encoding="utf-8")]
    n = len(labels)
    count = collections.Counter(labels)
    acc, hall, miss = count[ACCURATE] / n, count[HALLUCINATED] / n, count[MISSING] / n
    summary = {
        "gen_file": str(args.gen.resolve().relative_to(ROOT)),
        "mode": args.mode,
        "judge": args.judge,
        "n": n,
        "accuracy": round(100 * acc, 1),
        "hallucination": round(100 * hall, 1),
        "missing": round(100 * miss, 1),
        "truthfulness": round(100 * (acc - hall), 1),
        "unparsed": count[UNPARSED],
        "n_judge_calls": n_judge_calls,
        "scored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with RUNS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
