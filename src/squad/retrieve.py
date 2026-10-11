"""Bước 2: dựng index BM25 + bge-m3/FAISS, truy hồi hybrid (RRF) và đo recall@k, MRR.

    python -m src.squad.retrieve

Index được dựng MỘT lần cho cả kho 350 đoạn rồi dùng chung cho mọi câu hỏi.

Đầu ra:
    data/squad/index/kb.faiss, kb_ids.json
    results/squad/retrieval_{dev,test}.jsonl   top-k đoạn cho từng câu hỏi
    results/squad/retrieval_metrics.json
"""

from __future__ import annotations

import json

from src.squad.common import (
    CAND_K,
    EMBED_MODEL,
    INDEX,
    OUT,
    PROC,
    RRF_K,
    TOP_K,
    pick_gpu,
    read_jsonl,
    tokenize,
    write_json,
    write_jsonl,
)

KS = (1, 3, 5, 10)


def rank_metrics(rank_lists: list[list[str]], golds: list[str]) -> dict[str, float]:
    """rank_lists[i] là danh sách para_id đã xếp hạng, golds[i] là para_id đúng."""
    out = {f"recall@{k}": 0.0 for k in KS}
    mrr = 0.0
    for ranked, gold in zip(rank_lists, golds, strict=True):
        pos = ranked.index(gold) + 1 if gold in ranked else None
        if pos:
            mrr += 1.0 / pos
            for k in KS:
                out[f"recall@{k}"] += pos <= k
    n = len(golds)
    out = {k: round(v / n, 4) for k, v in out.items()}
    out["mrr@10"] = round(mrr / n, 4)
    return out


def rrf_fuse(rank_maps: list[dict[int, int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: cộng 1/(k + thứ hạng) của mỗi tài liệu trên từng bảng xếp hạng."""
    fused: dict[int, float] = {}
    for ranks in rank_maps:
        for j, r in ranks.items():
            fused[j] = fused.get(j, 0.0) + 1.0 / (k + r)
    return fused


def main() -> None:
    pick_gpu()
    import faiss
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    kb = read_jsonl(PROC / "kb.jsonl")
    ids = [p["para_id"] for p in kb]
    texts = [p["text"] for p in kb]

    bm25 = BM25Okapi([tokenize(t) for t in texts])

    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 512
    emb = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    INDEX.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX / "kb.faiss"))
    write_json(INDEX / "kb_ids.json", ids)

    metrics: dict = {}
    for split in ("dev", "test"):
        qs = read_jsonl(PROC / f"questions_{split}.jsonl")
        q_emb = model.encode(
            [q["question"] for q in qs],
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        d_scores, d_idx = index.search(q_emb, CAND_K)

        rows = []
        ranked: dict[str, list[list[str]]] = {"bm25": [], "dense": [], "hybrid": []}
        for i, q in enumerate(qs):
            b_scores = bm25.get_scores(tokenize(q["question"]))
            b_idx = np.argsort(-b_scores)[:CAND_K]
            b_rank = {int(j): r for r, j in enumerate(b_idx, 1)}
            d_rank = {int(j): r for r, j in enumerate(d_idx[i], 1)}
            d_cos = {int(j): float(s) for j, s in zip(d_idx[i], d_scores[i], strict=True)}

            rrf = rrf_fuse([b_rank, d_rank])
            h_idx = sorted(rrf, key=lambda j: (-rrf[j], j))[:TOP_K]

            ranked["bm25"].append([ids[j] for j in b_idx[:TOP_K]])
            ranked["dense"].append([ids[int(j)] for j in d_idx[i][:TOP_K]])
            ranked["hybrid"].append([ids[j] for j in h_idx])
            rows.append(
                {
                    "qid": q["qid"],
                    # Điểm cho context check: cosine bge-m3 cao nhất trên toàn KB.
                    # Điểm RRF chỉ phụ thuộc thứ hạng nên không đặt ngưỡng được.
                    "top1_dense_cos": round(float(d_scores[i][0]), 4),
                    "hits": [
                        {
                            "para_id": ids[j],
                            "rrf": round(rrf[j], 5),
                            "bm25_rank": b_rank.get(j),
                            "dense_rank": d_rank.get(j),
                            "dense_cos": round(d_cos[j], 4)
                            if j in d_cos
                            else round(float(emb[j] @ q_emb[i]), 4),
                            "bm25_score": round(float(b_scores[j]), 3),
                        }
                        for j in h_idx
                    ],
                }
            )
        write_jsonl(OUT / f"retrieval_{split}.jsonl", rows)

        golds = [q["gold_para_id"] for q in qs]
        ans = [i for i, q in enumerate(qs) if q["answerable"]]
        metrics[split] = {"n_all": len(qs), "n_answerable": len(ans)}
        for name, lists in ranked.items():
            metrics[split][name] = {
                "all": rank_metrics(lists, golds),
                "answerable": rank_metrics([lists[i] for i in ans], [golds[i] for i in ans]),
            }
        # Phân bố điểm context check giữa hai nhóm câu hỏi.
        cos = np.array([r["top1_dense_cos"] for r in rows])
        is_ans = np.array([q["answerable"] for q in qs])
        metrics[split]["top1_dense_cos"] = {
            g: {
                "mean": round(float(c.mean()), 4),
                **{f"p{p}": round(float(np.percentile(c, p)), 4) for p in (5, 25, 50, 75, 95)},
            }
            for g, c in (("answerable", cos[is_ans]), ("unanswerable", cos[~is_ans]))
        }

    write_json(OUT / "retrieval_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
