"""Truy hồi cho các bậc B2–B5: chia đoạn, tìm bằng từ khoá + vector, rồi xếp hạng lại.

Chạy MỘT lần, ghi ra một file dùng chung cho cả bốn bậc — nếu không thì mỗi bậc lại
phải nhúng lại toàn bộ 5 trang của 1.335 câu, tốn gấp bốn lần.

    python -m src.crag.retrieve --top-k 40

Đầu ra: data/crag_public_test/retrieval.jsonl
    mỗi dòng một câu hỏi, kèm top-k đoạn cùng bốn loại điểm
    (bm25, dense, rrf để chọn ứng viên; rerank để lọc ở B3 trở lên)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "crag_public_test"

EMBED_REPO = "BAAI/bge-m3"
RERANK_REPO = "BAAI/bge-reranker-v2-m3"

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def chunk_pages(
    item: dict, size: int = 220, overlap: int = 40, max_per_page: int = 60
) -> list[dict]:
    """Cắt chữ của từng trang thành đoạn chồng lấn, tính theo TỪ cho nhanh.

    ``max_per_page`` chặn các trang khổng lồ: 95% trang dưới 54.000 ký tự nhưng có
    trang tới hơn 250.000, để nguyên thì một câu hỏi sinh ra vài trăm đoạn.
    """
    out = []
    for p_idx, page in enumerate(item["pages"]):
        toks = page["text"].split()
        if not toks:
            continue
        step = size - overlap
        for c_idx, start in enumerate(range(0, len(toks), step)):
            if c_idx >= max_per_page:
                break
            piece = toks[start : start + size]
            if len(piece) < 20 and c_idx:  # đuôi vụn, bỏ
                break
            out.append(
                {
                    "page": p_idx,
                    "chunk": c_idx,
                    "page_name": page["page_name"],
                    "text": " ".join(piece),
                }
            )
    return out


class BM25:
    """Okapi BM25. Tự cài vì rank_bm25 không có sẵn và thuật toán chỉ vài dòng."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in corpus]
        self.lens = [len(d) for d in corpus]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0.0
        n_doc = len(corpus)
        df = Counter()
        for d in corpus:
            df.update(set(d))
        self.idf = {t: math.log(1 + (n_doc - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.docs, self.lens, strict=False):
            s = 0.0
            for term in query:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
                s += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            out.append(s)
        return out


def rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion — gộp hai bảng xếp hạng mà không cần chuẩn hoá điểm.

    Điểm của hai bộ tìm kiếm không cùng thang (BM25 không chặn trên, cosine trong
    [-1,1]), nên cộng thẳng là sai. RRF chỉ dùng THỨ HẠNG.
    """
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return fused


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--top-k", type=int, default=40, help="số đoạn giữ lại mỗi câu")
    parser.add_argument("--rerank-k", type=int, default=40, help="số đoạn đưa qua bộ xếp hạng lại")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", type=Path, default=DATA / "retrieval.jsonl")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_REPO, device=device)
    embedder.max_seq_length = 512
    embedder.half()

    rr_tok = AutoTokenizer.from_pretrained(RERANK_REPO)
    rr_model = (
        AutoModelForSequenceClassification.from_pretrained(RERANK_REPO, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    items = [json.loads(line) for line in (DATA / "questions.jsonl").open(encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]

    done = set()
    if args.out.exists():
        done = {
            json.loads(line)["interaction_id"]
            for line in args.out.open(encoding="utf-8")
            if line.strip()
        }
        print(f"đã có {len(done)} câu, bỏ qua")

    started, n_chunks = time.time(), 0
    with args.out.open("a", encoding="utf-8") as sink:
        for i, item in enumerate(items, 1):
            if item["interaction_id"] in done:
                continue
            chunks = chunk_pages(item)
            if not chunks:
                sink.write(
                    json.dumps(
                        {"interaction_id": item["interaction_id"], "chunks": []}, ensure_ascii=False
                    )
                    + "\n"
                )
                continue
            n_chunks += len(chunks)
            texts = [c["text"] for c in chunks]
            query = item["query"]

            bm25 = BM25([words(t) for t in texts]).scores(words(query))
            vecs = embedder.encode(
                texts,
                batch_size=args.batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            qvec = embedder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )[0]
            dense = (vecs @ qvec).tolist()

            order_b = sorted(range(len(texts)), key=lambda j: -bm25[j])
            order_d = sorted(range(len(texts)), key=lambda j: -dense[j])
            fused = rrf([order_b, order_d])
            cand = sorted(fused, key=lambda j: -fused[j])[: max(args.top_k, args.rerank_k)]

            pairs = [[query, texts[j]] for j in cand[: args.rerank_k]]
            with torch.no_grad():
                enc = rr_tok(
                    pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(device)
                logits = rr_model(**enc).logits.view(-1).float()
                probs = torch.sigmoid(logits).cpu().tolist()
            score_rr = dict(zip(cand[: args.rerank_k], probs, strict=False))

            keep = sorted(cand, key=lambda j: -score_rr.get(j, -1.0))[: args.top_k]
            sink.write(
                json.dumps(
                    {
                        "interaction_id": item["interaction_id"],
                        "n_chunks_total": len(chunks),
                        "chunks": [
                            {
                                **chunks[j],
                                "bm25": round(bm25[j], 4),
                                "dense": round(dense[j], 4),
                                "rrf": round(fused[j], 6),
                                "rerank": round(score_rr.get(j, -1.0), 6),
                            }
                            for j in keep
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()
            if i % 25 == 0:
                rate = (time.time() - started) / i
                print(
                    f"{i}/{len(items)} · {rate:.2f} s/câu · còn ~{rate * (len(items) - i) / 60:.0f} phút "
                    f"· tb {n_chunks / i:.0f} đoạn/câu",
                    flush=True,
                )

    out = args.out.resolve()
    print(f"xong: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
