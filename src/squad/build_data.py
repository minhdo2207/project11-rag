"""Bước 1: dựng knowledge base và bộ câu hỏi từ SQuAD 2.0 dev.

    python -m src.squad.build_data

Đầu ra (trong data/squad/processed):
    kb.jsonl              mỗi dòng là một đoạn (chunk)
    questions_dev.jsonl   câu hỏi của 3 bài dev, dùng để dò ngưỡng
    questions_test.jsonl  câu hỏi của 7 bài test, chỉ dùng để báo cáo
    stats.json            số bài, số đoạn, số câu hỏi của từng tập
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request

from src.squad.common import (
    N_ARTICLES,
    N_DEV,
    PROC,
    RAW,
    SEED,
    SQUAD_FILE,
    SQUAD_URL,
    write_json,
    write_jsonl,
)


def build(articles: list[dict], n_articles: int = N_ARTICLES, n_dev: int = N_DEV, seed: int = SEED):
    """Chọn bài theo seed, chia dev/test theo bài, khử đoạn trùng. Trả về (kb, questions, chosen)."""
    # Lấy mẫu trên danh sách tên bài đã sắp xếp để kết quả không phụ thuộc thứ tự trong file.
    titles = sorted(a["title"] for a in articles)
    chosen = random.Random(seed).sample(titles, n_articles)
    split_of = {t: ("dev" if i < n_dev else "test") for i, t in enumerate(chosen)}
    by_title = {a["title"]: a for a in articles}

    kb: list[dict] = []
    questions: dict[str, list[dict]] = {"dev": [], "test": []}
    seen: dict[str, str] = {}
    for a_idx, title in enumerate(chosen):
        split = split_of[title]
        name = title.replace("_", " ")
        n_para = 0
        for para in by_title[title]["paragraphs"]:
            ctx = para["context"].strip()
            h = hashlib.sha1(ctx.encode("utf-8")).hexdigest()
            if h not in seen:
                seen[h] = f"a{a_idx:02d}_p{n_para:03d}"
                n_para += 1
                kb.append(
                    {
                        "para_id": seen[h],
                        "title": title,
                        "split": split,
                        "context": ctx,
                        # Chèn tên bài vào đầu chunk để cả BM25 lẫn bge-m3 đều thấy tên thực thể.
                        "text": f"{name}\n{ctx}",
                    }
                )
            for qa in para["qas"]:
                golds = list(dict.fromkeys(x["text"] for x in qa["answers"]))
                questions[split].append(
                    {
                        "qid": qa["id"],
                        "split": split,
                        "title": title,
                        "question": qa["question"].strip(),
                        "answerable": not qa["is_impossible"],
                        # chỉ dùng để chấm, không bao giờ vào index hay prompt
                        "gold_answers": golds,
                        "gold_para_id": seen[h],
                    }
                )
    return kb, questions, chosen


def validate(kb: list[dict], questions: dict[str, list[dict]]) -> None:
    """Các kiểm tra ở mục "Validate" của proposal."""
    para_by_id = {p["para_id"]: p for p in kb}
    assert len(para_by_id) == len(kb), "para_id bị trùng"
    dev_titles = {p["title"] for p in kb if p["split"] == "dev"}
    test_titles = {p["title"] for p in kb if p["split"] == "test"}
    assert not dev_titles & test_titles, "dev và test dùng chung bài"
    for qs in questions.values():
        for q in qs:
            assert q["gold_para_id"] in para_by_id
            assert q["answerable"] == bool(q["gold_answers"])
            ctx = para_by_id[q["gold_para_id"]]["context"]
            for g in q["gold_answers"]:
                assert g in ctx, f"gold answer không nằm trong đoạn: {q['qid']}"


def main() -> None:
    if not SQUAD_FILE.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, SQUAD_FILE)
    with open(SQUAD_FILE, encoding="utf-8") as f:
        articles = json.load(f)["data"]

    kb, questions, chosen = build(articles)
    validate(kb, questions)

    write_jsonl(PROC / "kb.jsonl", kb)
    stats: dict = {"seed": SEED, "kb_paragraphs": len(kb), "splits": {}}
    for split, qs in questions.items():
        write_jsonl(PROC / f"questions_{split}.jsonl", qs)
        n_ans = sum(q["answerable"] for q in qs)
        stats["splits"][split] = {
            "articles": sorted({q["title"] for q in qs}, key=chosen.index),
            "paragraphs": sum(p["split"] == split for p in kb),
            "questions": len(qs),
            "answerable": n_ans,
            "unanswerable": len(qs) - n_ans,
        }
    words = sorted(len(p["context"].split()) for p in kb)
    stats["words_per_paragraph"] = {
        "mean": round(sum(words) / len(words), 1),
        "median": words[len(words) // 2],
        "max": words[-1],
    }
    stats["total_questions"] = sum(s["questions"] for s in stats["splits"].values())
    stats["total_answerable"] = sum(s["answerable"] for s in stats["splits"].values())
    stats["total_unanswerable"] = sum(s["unanswerable"] for s in stats["splits"].values())
    write_json(PROC / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
