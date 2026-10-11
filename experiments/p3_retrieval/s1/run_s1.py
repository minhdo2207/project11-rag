#!/usr/bin/env python3
"""S1 ablation runner.

The deliverable of scenario 1 is not the retriever — it is this table:

    mode      recall@1  recall@5  nDCG@10
    dense       ...       ...       ...
    bm25        ...       ...       ...
    hybrid      ...       ...       ...

Three things to read off it:

  1. hybrid must beat BOTH single branches. If it does not, the problem is
     upstream (chunking, or the eval set itself) — do not move on to S2.
  2. if dense ≈ bm25-only, your embedding model is contributing nothing on this
     corpus. Investigate before paying for a bigger one.
  3. the per-query breakdown shows WHICH queries fusion fixes and which it
     breaks. The broken ones are more informative than the aggregate.

    python run_s1.py --docs corpus.jsonl --eval eval.jsonl
    python run_s1.py --demo               # synthetic data, no model download

Formats (JSON lines):
    corpus.jsonl  {"id": "d1", "text": "...", "metadata": {...}}
    eval.jsonl    {"query": "...", "gold_ids": ["d1", "d7"]}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from embedding import Embedder, EmbeddingConfig
from retriever import HybridRetriever, RetrieverConfig, tokenize

MODES = ("dense", "bm25", "hybrid")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    g = set(gold)
    if not g:
        return float("nan")
    dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(retrieved[:k], 1) if i in g)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(g), k) + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_mode(r: HybridRetriever, dataset: list[dict], mode: str,
                  ks=(1, 3, 5, 10)) -> tuple[dict, dict[str, float]]:
    """Returns aggregate metrics and per-query recall@5 (for the diff view)."""
    agg = {f"recall@{k}": [] for k in ks}
    agg["ndcg@10"] = []
    per_query: dict[str, float] = {}

    for row in dataset:
        res = r.retrieve(row["query"], top_k=max(ks), mode=mode)
        ids = res.ids()
        for k in ks:
            agg[f"recall@{k}"].append(recall_at_k(ids, row["gold_ids"], k))
        agg["ndcg@10"].append(ndcg_at_k(ids, row["gold_ids"], 10))
        per_query[row["query"]] = recall_at_k(ids, row["gold_ids"], 5)

    return {k: float(np.nanmean(v)) for k, v in agg.items()}, per_query


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_table(results: dict[str, dict]) -> None:
    cols = ["recall@1", "recall@3", "recall@5", "recall@10", "ndcg@10"]
    print(f"\n{'mode':<10}" + "".join(f"{c:>11}" for c in cols))
    print("-" * (10 + 11 * len(cols)))
    for mode in MODES:
        row = results[mode]
        print(f"{mode:<10}" + "".join(f"{row[c]:>11.3f}" for c in cols))


def print_verdict(results: dict[str, dict]) -> bool:
    """S1 stop condition. Returns True if it is safe to move on to S2."""
    d, b, h = (results[m]["recall@5"] for m in MODES)
    print("\nverdict")
    ok = True

    # saturation guard: if every mode is at the ceiling the comparison is
    # meaningless — the eval set is too easy or the corpus too small
    if min(d, b, h) >= 0.98:
        print("  WARNING all three modes are at recall@5 >= 0.98.")
        print("          The eval set cannot separate them. Add harder queries and")
        print("          hard negatives (same topic, no answer) before reading anything")
        print("          into this table.")
        return False

    if h > d and h > b:
        gain = (h - max(d, b)) / max(max(d, b), 1e-9) * 100
        print(f"  hybrid beats both branches (+{gain:.1f}% over the better one)")
    else:
        better = "dense" if d >= b else "bm25"
        print(f"  WARNING hybrid ({h:.3f}) does not beat {better} ({max(d, b):.3f}).")
        print("          Fusion is not earning its keep. Likely causes, in order:")
        print("          chunking, the eval set, then rrf_k. Do NOT proceed to S2.")
        ok = False

    if abs(d - b) < 0.02:
        print(f"  NOTE dense ({d:.3f}) and bm25 ({b:.3f}) are within 0.02.")
        print("       The embedding model may be contributing little on this corpus;")
        print("       check before paying for a larger one.")
    elif d < b:
        print(f"  NOTE bm25 ({b:.3f}) beats dense ({d:.3f}) — the corpus is")
        print("       identifier-heavy. Consider weighting the lexical branch up in S5.")
    return ok


def print_diff(per_query: dict[str, dict[str, float]], limit: int = 6) -> None:
    """Which queries fusion fixed, and which it broke. The broken ones matter more."""
    fixed, broken = [], []
    for q, h in per_query["hybrid"].items():
        best_single = max(per_query["dense"][q], per_query["bm25"][q])
        if h > best_single:
            fixed.append((q, best_single, h))
        elif h < best_single:
            broken.append((q, best_single, h))

    print(f"\nper-query diff  fixed by fusion: {len(fixed)}   broken by fusion: {len(broken)}")
    for label, rows in (("fixed", fixed), ("broken", broken)):
        for q, before, after in rows[:limit]:
            print(f"  [{label:<6}] {before:.2f} -> {after:.2f}  {q[:58]}")
    if broken:
        print("  Inspect the broken ones: they usually show one branch dragging a")
        print("  strong result down. That is an rrf_k or weighting signal, not noise.")


def sweep_rrf_k(r: HybridRetriever, dataset: list[dict],
                values=(10, 30, 60, 120)) -> None:
    print("\nrrf_k sweep (hybrid)")
    original = r.config.rrf_k
    for k in values:
        r.config.rrf_k = k
        m, _ = evaluate_mode(r, dataset, "hybrid")
        print(f"  rrf_k={k:<4} recall@5={m['recall@5']:.3f}  ndcg@10={m['ndcg@10']:.3f}")
    r.config.rrf_k = original


# --------------------------------------------------------------------------- #
# demo data (no model download needed)
# --------------------------------------------------------------------------- #

class _HashingEmbedder:
    """Deterministic token hashing. Not semantic — exists only so the ablation
    machinery is runnable without weights. Never use it for real numbers."""
    dim = 512

    def _v(self, t: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(t):
            v[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts, **kw):
        return np.vstack([self._v(t) for t in texts])

    def encode_queries(self, texts, **kw):
        return np.vstack([self._v(t) for t in texts])


DEMO_DOCS = [
    {"id": "d1", "text": "Để đặt lại mật khẩu, vào Cài đặt > Bảo mật > Đổi mật khẩu. "
                         "Hệ thống gửi mã OTP về số điện thoại đã đăng ký."},
    {"id": "d2", "text": "Mã lỗi ERR_4012 xuất hiện khi phiên đăng nhập hết hạn. "
                         "Người dùng cần đăng nhập lại để tiếp tục."},
    {"id": "d3", "text": "Gói cước Combo 90 bao gồm 90GB data tốc độ cao mỗi tháng, "
                         "miễn phí gọi nội mạng dưới 20 phút."},
    {"id": "d4", "text": "Thời hạn thanh toán hóa đơn là ngày 15 hàng tháng. "
                         "Quá hạn 7 ngày dịch vụ sẽ bị tạm ngưng."},
    {"id": "d5", "text": "Khi quên mật khẩu, chọn Quên mật khẩu ở màn hình đăng nhập "
                         "và làm theo hướng dẫn xác thực."},
    {"id": "d8", "text": "Để tra cứu lịch sử thanh toán, vào mục Hóa đơn trong ứng dụng, "
                         "chọn tab Đã thanh toán."},
    # --- hard negatives: cùng chủ đề, không chứa đáp án ---
    {"id": "n1", "text": "Mã lỗi ERR_5003 nghĩa là máy chủ quá tải. Vui lòng thử lại sau."},
    {"id": "n2", "text": "Mã lỗi ERR_4015 xảy ra khi thiết bị không được tin cậy."},
    {"id": "n3", "text": "Mã lỗi ERR_4021 báo tài khoản bị khóa tạm thời."},
    {"id": "n4", "text": "Gói cước Combo 120 có 120GB data và miễn phí gọi ngoại mạng 50 phút."},
    {"id": "n5", "text": "Gói cước Combo 60 có 60GB data, phù hợp người dùng cơ bản."},
    {"id": "n6", "text": "Gói cước Combo 90 Plus là phiên bản nâng cấp, chi tiết liên hệ tổng đài."},
    {"id": "n7", "text": "Đổi mã PIN SIM trong mục Cài đặt > SIM, khác với đổi mật khẩu tài khoản."},
    {"id": "n8", "text": "Bảo mật hai lớp yêu cầu xác thực OTP mỗi lần đăng nhập thiết bị mới."},
    {"id": "n9", "text": "Thanh toán tự động được trừ vào ngày 10 nếu đã liên kết ngân hàng."},
    {"id": "n10", "text": "Hóa đơn điện tử được gửi qua email vào ngày 1 hàng tháng."},
    {"id": "n11", "text": "Tạm ngưng dịch vụ theo yêu cầu khách hàng tối đa 3 tháng mỗi năm."},
    {"id": "n12", "text": "Khôi phục dịch vụ sau tạm ngưng cần thanh toán toàn bộ dư nợ."},
    {"id": "n13", "text": "Ứng dụng hỗ trợ đăng nhập bằng vân tay trên thiết bị tương thích."},
    {"id": "n14", "text": "Lịch sử cuộc gọi được lưu trong mục Tra cứu, không nằm ở mục Hóa đơn."},
]

DEMO_EVAL = [
    {"query": "làm sao để đổi mật khẩu", "gold_ids": ["d1", "d5"]},
    {"query": "ERR_4012 là lỗi gì", "gold_ids": ["d2"]},
    {"query": "gói Combo 90 có bao nhiêu data", "gold_ids": ["d3"]},
    {"query": "khi nào bị cắt dịch vụ nếu chậm đóng tiền", "gold_ids": ["d4"]},
    {"query": "tôi quên mật khẩu rồi phải làm sao", "gold_ids": ["d5", "d1"]},
    {"query": "xem lại hóa đơn cũ ở đâu", "gold_ids": ["d8"]},
]


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--eval", dest="eval_path", type=Path)
    ap.add_argument("--demo", action="store_true", help="synthetic data + hashing embedder")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep", action="store_true", help="also sweep rrf_k")
    ap.add_argument("--save", type=Path, help="persist the index here")
    args = ap.parse_args()

    if args.demo:
        docs, dataset = DEMO_DOCS, DEMO_EVAL
        embedder = _HashingEmbedder()
        print("demo mode: hashing embedder, numbers are structural only")
    else:
        if not (args.docs and args.eval_path):
            ap.error("--docs and --eval are required (or use --demo)")
        docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines() if l.strip()]
        dataset = [json.loads(l) for l in args.eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        embedder = Embedder(EmbeddingConfig(model_name=args.model, device=args.device))

    print(f"corpus {len(docs)} docs | eval {len(dataset)} queries")

    cfg = RetrieverConfig()
    r = HybridRetriever(embedder, cfg).index(docs, show_progress=not args.demo)
    print(f"indexed: dim={r.vectors.shape[1]}  bm25 vocab={len(r.bm25.vocab)}")

    results, per_query = {}, {}
    for mode in MODES:
        results[mode], per_query[mode] = evaluate_mode(r, dataset, mode)

    print_table(results)
    ok = print_verdict(results)
    print_diff(per_query)

    if args.sweep:
        sweep_rrf_k(r, dataset)

    if args.save:
        r.save(args.save)
        print(f"\nindex saved to {args.save}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
