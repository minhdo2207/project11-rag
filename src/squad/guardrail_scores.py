"""Bước 4: chấm các điểm guardrail cho câu trả lời của cấu hình c1 (RAG).

Chỉ tính điểm và lưu lại, chưa áp ngưỡng. C2–C6 và việc dò ngưỡng đọc từ file này.

Điểm được tính:
    ctx_cos     context check: cosine bge-m3 cao nhất giữa câu hỏi và KB (lấy từ bước 2)
    lexical     |T(y) ∩ T(C)| / |T(y)|, T = token nội dung (chữ thường, bỏ stop word)
    hhem_*      HHEM, 4 biến thể: hypothesis là câu trả lời (ans) hoặc câu hỏi + câu trả lời (qa);
                premise là cả 5 đoạn ghép lại (concat) hoặc lấy max trên từng đoạn (max)
    relevance   cosine bge-m3 giữa câu hỏi và câu trả lời

    python -m src.squad.guardrail_scores --split dev

Đầu ra: results/squad/scores_<model>_c1_<split>.jsonl
"""

from __future__ import annotations

import argparse
import time

from src.squad.common import (
    EMBED_MODEL,
    OUT,
    PROC,
    pick_gpu,
    read_jsonl,
    tokenize,
    write_json,
    write_jsonl,
)

STOP = set(
    """a about above after again against all am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do does doesn doing don
down during each few for from further had hadn has hasn have haven having he her here hers herself
him himself his how i if in into is isn it its itself just me more most mustn my myself no nor not
now of off on once only or other our ours ourselves out over own s same shan she should shouldn so
some such t than that the their theirs them themselves then there these they this those through to
too under until up very was wasn we were weren what when where which while who whom why will with
won would wouldn you your yours yourself yourselves""".split()
)


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOP}


def lexical_score(answer: str, context: str) -> float:
    """Tỷ lệ từ nội dung của câu trả lời có xuất hiện trong ngữ cảnh."""
    ty = content_tokens(answer)
    if not ty:
        return 0.0
    return len(ty & content_tokens(context)) / len(ty)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()

    pick_gpu()
    from sentence_transformers import SentenceTransformer

    from src.squad.hhem import HHEM

    qs = {q["qid"]: q for q in read_jsonl(PROC / f"questions_{args.split}.jsonl")}
    kb = {p["para_id"]: p["text"] for p in read_jsonl(PROC / "kb.jsonl")}
    ctx_cos = {
        r["qid"]: r["top1_dense_cos"] for r in read_jsonl(OUT / f"retrieval_{args.split}.jsonl")
    }
    embedder = SentenceTransformer(EMBED_MODEL)
    hhem = HHEM()

    timing = {}
    for path in sorted(OUT.glob(f"gen_*_c1_{args.split}.jsonl")):
        gens = read_jsonl(path)
        name = path.stem.replace("gen_", "")
        n = len(gens)
        questions = [qs[g["qid"]]["question"] for g in gens]
        answers = [g["answer"] for g in gens]
        passages = [[kb[i] for i in g["ctx_ids"]] for g in gens]
        t = {}

        t0 = time.perf_counter()
        lex = [lexical_score(a, "\n".join(p)) for a, p in zip(answers, passages, strict=True)]
        t["lexical"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        qe = embedder.encode(
            questions, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        ae = embedder.encode(
            answers, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        rel = (qe * ae).sum(axis=1).tolist()
        t["relevance"] = time.perf_counter() - t0

        hh = {}
        qa_hyps = [f"{q} {a}" for q, a in zip(questions, answers, strict=True)]
        for hyp_name, hyps in (("ans", answers), ("qa", qa_hyps)):
            t0 = time.perf_counter()
            hh[f"hhem_{hyp_name}_concat"] = hhem.score(
                [("\n\n".join(p), h) for p, h in zip(passages, hyps, strict=True)], batch_size=8
            )
            t[f"hhem_{hyp_name}_concat"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            flat = hhem.score(
                [(p, h) for ps, h in zip(passages, hyps, strict=True) for p in ps], batch_size=32
            )
            k = len(passages[0])
            hh[f"hhem_{hyp_name}_max"] = [max(flat[i * k : (i + 1) * k]) for i in range(n)]
            t[f"hhem_{hyp_name}_max"] = time.perf_counter() - t0

        rows = [
            {
                "qid": g["qid"],
                "ctx_cos": ctx_cos[g["qid"]],
                "lexical": round(lex[i], 4),
                "relevance": round(float(rel[i]), 4),
                **{k: round(v[i], 4) for k, v in hh.items()},
            }
            for i, g in enumerate(gens)
        ]
        write_jsonl(OUT / f"scores_{name}.jsonl", rows)
        # Thời gian trung bình mỗi câu khi chạy theo lô; chỉ để ước lượng, chưa phải latency.
        timing[name] = {"n": n, **{k: round(v / n * 1000, 2) for k, v in t.items()}}
        print(name, "ms/câu:", timing[name], flush=True)

    write_json(OUT / f"scores_timing_{args.split}.json", timing)


if __name__ == "__main__":
    main()
