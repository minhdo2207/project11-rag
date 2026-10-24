"""Dựng bộ câu canary: câu mà hành vi đúng là KHÔNG trả lời theo tài liệu.

Hai loại, cùng ghi vào một bộ dữ liệu riêng có trường ``canary_type``:

``no_evidence`` — câu CRAG thật, nhưng bỏ mọi đoạn truy hồi có chứa đáp án chuẩn.
    Thử cổng từ chối khi truy hồi trượt. Giữ nguyên đáp án chuẩn và chấm như thường:
    model vẫn có thể trả lời đúng bằng trí nhớ, theo luật CRAG như vậy không sai, nên
    số cần nhìn là tỷ lệ bịa. Chỉ lấy câu có đáp án ngắn (≤ ``--max-answer-words`` từ),
    vì đáp án dài kiểu cả câu văn thì không đoạn nào chứa nguyên văn, lọc sẽ không bỏ
    được đoạn nào dù đoạn đó có đáp án. Một đoạn bị bỏ nếu chứa nguyên đáp án (sau
    chuẩn hoá, như ``src.crag.calibrate``) HOẶC bất kỳ từ đặc trưng nào của đáp án —
    đoạn hay nhắc tắt ("the Predators", chỉ họ "Hanks"). Lọc thừa thì không sao, lọc sót
    thì canary hỏng.

``fictional`` — câu hỏi theo mẫu về một thực thể không tồn tại, ghép với các đoạn của
    một câu thật cùng lĩnh vực (lạc đề nhưng cùng chủ đề). Không có đáp án: từ chối hoặc
    nói "invalid question" là đúng, trả lời gì khác là bịa (``src.crag.score`` chấm
    riêng nhánh này). Điểm xếp hạng lại được tính lại cho câu hỏi mới, vì cổng B5 đọc
    điểm này.

    python -m src.crag.make_canary --src data/crag_public_test --out data/crag_canary

Đầu ra: <out>/questions.jsonl và <out>/retrieval.jsonl, cùng định dạng với bộ gốc nhưng
không có ``pages`` — chỉ chạy được B0 và B2–B5.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.crag.calibrate import norm
from src.crag.retrieve import RERANK_REPO

ROOT = Path(__file__).resolve().parents[2]

# Mẫu câu theo lĩnh vực của CRAG; {name} là thực thể bịa
TEMPLATES = {
    "movie": [
        "what year was the movie {name} released?",
        "who directed the film {name}?",
        "how much did the movie {name} gross at the box office?",
    ],
    "music": [
        "how many albums has the band {name} released?",
        "what was the first single by the singer {name}?",
        "which record label is the rapper {name} signed to?",
    ],
    "sports": [
        "which team does the basketball player {name} play for?",
        "how many goals did the footballer {name} score last season?",
        "when did the tennis player {name} win their first grand slam?",
    ],
    "finance": [
        "what is the stock ticker of {name} inc?",
        "what was the closing price of {name} corp stock yesterday?",
        "who is the ceo of {name} holdings?",
    ],
    "open": [
        "what is the population of the town of {name}?",
        "who founded the company {name}?",
        "when was the {name} bridge built?",
    ],
}
_SYLLABLES = (
    "vel or ith kan mere zor qua bri tal un dro fen sha lix mo rav tess yul gor nim".split()
)


def fake_name(rng: random.Random, corpus_words: set[str]) -> str:
    """Tên ghép từ âm tiết vô nghĩa, kiểm tra không trùng từ nào trong toàn bộ đoạn truy hồi."""
    while True:
        parts = [
            "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3))) for _ in range(2)
        ]
        if not any(p in corpus_words for p in parts):
            return " ".join(p.capitalize() for p in parts)


def golds_of(item: dict) -> list[str]:
    golds = [str(item["answer"])] + [str(a) for a in item["alt_ans"] if a != ""]
    return [g for g in map(norm, golds) if g and g not in ("invalid question", "yes", "no")]


# Từ quá phổ biến để coi là dấu vết của đáp án
_COMMON = set(
    "that this with from have were what when which their there about would after first "
    "year years team club city state university company movie film album song band".split()
)


def key_words(gold: str) -> set[str]:
    """Từ đặc trưng của một đáp án đã chuẩn hoá: số, hoặc chữ ≥ 4 ký tự không phổ biến."""
    return {w for w in gold.split() if w.isdigit() or (len(w) >= 4 and w not in _COMMON)}


def mentions(text: str, golds: list[str]) -> bool:
    text = norm(text)
    words = set(text.split())
    return any(g in text or key_words(g) & words for g in golds)


def no_evidence(items: list[dict], retrieval: dict, max_words: int) -> list[tuple[dict, dict]]:
    out = []
    for item in items:
        golds = golds_of(item)
        if not golds or any(len(g.split()) > max_words for g in golds):
            continue
        chunks = retrieval.get(item["interaction_id"], [])
        kept = [c for c in chunks if not mentions(c["text"], golds)]
        if not kept or len(kept) == len(chunks):  # hết sạch đoạn, hoặc vốn không đoạn nào có đáp án
            continue
        cid = f"canary-ne-{item['interaction_id']}"
        out.append(
            (
                {**item, "interaction_id": cid, "canary_type": "no_evidence"},
                {"interaction_id": cid, "n_chunks_total": len(kept), "chunks": kept},
            )
        )
    return out


def fictional(
    items: list[dict], retrieval: dict, n: int, rng: random.Random, rerank
) -> list[tuple[dict, dict]]:
    corpus_words = {
        w for chunks in retrieval.values() for c in chunks for w in norm(c["text"]).split()
    }
    by_domain: dict[str, list[dict]] = {}
    for item in items:
        if retrieval.get(item["interaction_id"]) and item["domain"] in TEMPLATES:
            by_domain.setdefault(item["domain"], []).append(item)
    domains = sorted(by_domain)
    out = []
    for i in range(n):
        domain = domains[i % len(domains)]
        host = rng.choice(by_domain[domain])
        query = rng.choice(TEMPLATES[domain]).format(name=fake_name(rng, corpus_words))
        chunks = retrieval[host["interaction_id"]]
        scores = rerank(query, [c["text"] for c in chunks])
        chunks = sorted(
            ({**c, "rerank": round(s, 6)} for c, s in zip(chunks, scores, strict=True)),
            key=lambda c: -c["rerank"],
        )
        cid = f"canary-fx-{i:04d}"
        out.append(
            (
                {
                    "interaction_id": cid,
                    "query": query,
                    "query_time": host["query_time"],
                    "answer": "invalid question",
                    "alt_ans": [],
                    "domain": domain,
                    "question_type": "canary",
                    "static_or_dynamic": "static",
                    "canary_type": "fictional",
                    "context_from": host["interaction_id"],
                },
                {"interaction_id": cid, "n_chunks_total": len(chunks), "chunks": chunks},
            )
        )
    return out


def make_reranker(batch: int = 64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(RERANK_REPO)
    model = (
        AutoModelForSequenceClassification.from_pretrained(RERANK_REPO, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    def rerank(query: str, texts: list[str]) -> list[float]:
        probs = []
        for start in range(0, len(texts), batch):
            pairs = [[query, t] for t in texts[start : start + batch]]
            with torch.no_grad():
                enc = tok(
                    pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(device)
                logits = model(**enc).logits.view(-1).float()
            probs += torch.sigmoid(logits).cpu().tolist()
        return probs

    return rerank


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "crag_public_test")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "crag_canary")
    parser.add_argument("--n-fictional", type=int, default=150)
    parser.add_argument("--max-answer-words", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    items = [json.loads(line) for line in (args.src / "questions.jsonl").open(encoding="utf-8")]
    for item in items:
        item.pop("pages", None)
    retrieval = {
        r["interaction_id"]: r["chunks"]
        for r in map(json.loads, (args.src / "retrieval.jsonl").open(encoding="utf-8"))
    }
    rng = random.Random(args.seed)

    ne = no_evidence(items, retrieval, args.max_answer_words)
    fx = fictional(items, retrieval, args.n_fictional, rng, make_reranker())

    args.out.mkdir(parents=True, exist_ok=True)
    with (
        (args.out / "questions.jsonl").open("w", encoding="utf-8") as fq,
        (args.out / "retrieval.jsonl").open("w", encoding="utf-8") as fr,
    ):
        for question, retr in ne + fx:
            fq.write(json.dumps(question, ensure_ascii=False) + "\n")
            fr.write(json.dumps(retr, ensure_ascii=False) + "\n")
    print(f"no_evidence: {len(ne)} câu · fictional: {len(fx)} câu → {args.out}")


if __name__ == "__main__":
    main()
