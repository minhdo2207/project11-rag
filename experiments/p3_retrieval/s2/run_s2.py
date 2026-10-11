#!/usr/bin/env python3
"""S2 ablation runner — chunking strategies, retrieval held fixed.

THE METHODOLOGICAL POINT OF S2
------------------------------
Chunk ids are not stable across chunking strategies. Change the chunker and
every id changes, so the `gold_ids` labels from S1 are worthless here. Reusing
them would silently measure nothing.

S2 therefore evaluates against **answer spans in the source document**: a
retrieval counts as a hit when a retrieved chunk actually contains the answer
text. That label survives any chunking change, and it measures what you
actually care about.

    eval.jsonl   {"query": "...", "answer_spans": ["ngày 15 hàng tháng"]}

THE METRIC THAT MATTERS MOST
----------------------------
`fragmentation_rate` — the share of gold spans that NO single chunk contains in
full. These are unrecoverable: no reranker, no bigger embedding model, no amount
of k will fix them, because the information was destroyed at index time. It is
computed without running retrieval at all, so it is the cheapest useful
diagnostic in the whole project. Drive it to zero before tuning anything else.

    python run_s2.py --demo
    python run_s2.py --docs corpus.jsonl --eval eval.jsonl

    corpus.jsonl  {"source": "faq.md", "text": "<full document text>"}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from chunking import ChunkConfig, chunk_documents, chunk_stats
from embedding import Embedder, EmbeddingConfig
from retriever import HybridRetriever, RetrieverConfig, tokenize


# --------------------------------------------------------------------------- #
# span matching
# --------------------------------------------------------------------------- #

def norm(s: str) -> str:
    """Normalise for containment testing: NFC (Vietnamese diacritics), lowercase,
    collapse whitespace. Without NFC, visually identical Vietnamese strings fail
    to match because the corpus and the labels use different compositions."""
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def contains_span(chunk_text: str, span: str) -> bool:
    return norm(span) in norm(chunk_text)


def fragmentation_rate(chunks, dataset) -> tuple[float, list[str]]:
    """Share of gold spans that no single chunk contains in full.

    Needs no retrieval — this is a property of the index alone.
    """
    bodies = [norm(c.text) for c in chunks]
    total, lost, examples = 0, 0, []
    for row in dataset:
        for span in row["answer_spans"]:
            total += 1
            if not any(norm(span) in b for b in bodies):
                lost += 1
                if len(examples) < 5:
                    examples.append(span)
    return (lost / total if total else 0.0), examples


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def answer_recall(r: HybridRetriever, dataset, ks=(1, 3, 5, 10)) -> dict:
    """answer_recall@k: at least one of the top-k chunks contains a gold span.
    span_coverage@k: share of ALL gold spans covered by the top-k — the metric
    that catches multi-part answers split across chunks."""
    hit = {k: [] for k in ks}
    cov = {k: [] for k in ks}

    for row in dataset:
        res = r.retrieve(row["query"], top_k=max(ks))
        texts = [h.doc.text for h in res.hits]
        for k in ks:
            top = texts[:k]
            found = [s for s in row["answer_spans"]
                     if any(contains_span(t, s) for t in top)]
            hit[k].append(1.0 if found else 0.0)
            cov[k].append(len(found) / len(row["answer_spans"]))

    out = {f"answer_recall@{k}": float(np.mean(hit[k])) for k in ks}
    out.update({f"span_coverage@{k}": float(np.mean(cov[k])) for k in ks})
    return out


# --------------------------------------------------------------------------- #
# variants — a cumulative ladder, one mechanism added at a time
# --------------------------------------------------------------------------- #

VARIANTS = [
    ("V0 fixed-700",          "fixed",     dict(overlap_chars=0,  protect_atomic=False, enrich_header=False)),
    ("V1 fixed+overlap",      "fixed",     dict(overlap_chars=120, protect_atomic=False, enrich_header=False)),
    ("V2 structure",          "structure", dict(overlap_chars=0,  protect_atomic=False, enrich_header=False)),
    ("V3 +atomic blocks",     "structure", dict(overlap_chars=0,  protect_atomic=True,  enrich_header=False)),
    ("V4 +enrich header",     "structure", dict(overlap_chars=0,  protect_atomic=True,  enrich_header=True)),
]


def run_variant(label, strategy, overrides, docs, dataset, embedder, base_cfg):
    cfg = replace(base_cfg, **overrides)
    chunks = chunk_documents(docs, cfg, strategy=strategy)
    stats = chunk_stats(chunks)
    frag, examples = fragmentation_rate(chunks, dataset)

    r = HybridRetriever(embedder, RetrieverConfig(mode="hybrid")).index(
        [c.to_dict() for c in chunks])
    metrics = answer_recall(r, dataset)

    return {"label": label, "strategy": strategy, "frag": frag,
            "frag_examples": examples, **stats, **metrics}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def print_table(rows: list[dict]) -> None:
    cols = [("n_chunks", "chunks", "{:>7.0f}"),
            ("avg_chars", "avg_len", "{:>8.0f}"),
            ("frag", "frag_rate", "{:>10.3f}"),
            ("answer_recall@1", "ans@1", "{:>7.3f}"),
            ("answer_recall@5", "ans@5", "{:>7.3f}"),
            ("span_coverage@5", "cov@5", "{:>7.3f}")]
    header = f"{'variant':<22}" + "".join(f"{h:>{len(f.format(0))}}" for _, h, f in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:<22}" + "".join(f.format(r[k]) for k, _, f in cols))


def print_verdict(rows: list[dict]) -> bool:
    base, best = rows[0], max(rows, key=lambda r: r["span_coverage@5"])
    print("\nverdict")
    ok = True

    print(f"  baseline  {base['label']}: frag={base['frag']:.3f}  cov@5={base['span_coverage@5']:.3f}")
    print(f"  best      {best['label']}: frag={best['frag']:.3f}  cov@5={best['span_coverage@5']:.3f}")

    if best["frag"] > 0.0:
        print(f"\n  WARNING fragmentation_rate is {best['frag']:.3f} even at the best variant.")
        print("          Those gold spans are UNRECOVERABLE — no reranker or larger")
        print("          embedding model will reach them. Fix chunking first.")
        for ex in best["frag_examples"]:
            print(f"            lost span: {ex[:66]!r}")
        ok = False
    else:
        print("\n  fragmentation_rate = 0: every gold span is contained in some chunk.")
        print("  Retrieval can now reach every answer; remaining errors are ranking errors.")

    if best["span_coverage@5"] <= base["span_coverage@5"]:
        print("\n  WARNING no variant beats the fixed-size baseline on span_coverage@5.")
        print("          Either the corpus has no real structure to exploit, or the")
        print("          eval spans are too short to be split by any chunker.")
        print("          Add multi-sentence spans and procedural answers before concluding.")
        ok = False

    cost = best["n_chunks"] / max(base["n_chunks"], 1)
    print(f"\n  index cost: {cost:.2f}x the baseline chunk count "
          f"({base['n_chunks']:.0f} -> {best['n_chunks']:.0f})")
    return ok


def print_mechanism_deltas(rows: list[dict]) -> None:
    """Attribute the gain to each mechanism separately — the point of a ladder."""
    print("\nmechanism contribution (span_coverage@5)")
    pairs = [("overlap", 0, 1), ("structure boundaries", 1, 2),
             ("atomic blocks", 2, 3), ("enrich header", 3, 4)]
    for name, a, b in pairs:
        if b < len(rows):
            d = rows[b]["span_coverage@5"] - rows[a]["span_coverage@5"]
            df = rows[b]["frag"] - rows[a]["frag"]
            print(f"  {name:<22} cov@5 {d:+.3f}   frag {df:+.3f}")


# --------------------------------------------------------------------------- #
# demo corpus — structures a fixed-size chunker is guaranteed to break
# --------------------------------------------------------------------------- #

class _HashingEmbedder:
    """Token hashing. Not semantic — present so the ablation runs without
    weights. Never read quality numbers off it."""
    dim = 512

    def _v(self, t):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(t):
            v[hash(tok) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts, **kw):
        return np.vstack([self._v(t) for t in texts])

    def encode_queries(self, texts, **kw):
        return np.vstack([self._v(t) for t in texts])


DEMO_DOCS = [{
    "source": "faq_thanh_toan.md",
    "text": """\
# Thanh toán và hóa đơn

## Giới thiệu chung
Tài liệu này mô tả toàn bộ quy định về thanh toán, hóa đơn và xử lý khiếu nại
dành cho khách hàng cá nhân. Các quy định áp dụng từ ngày 01/01/2026 và thay thế
cho các văn bản trước đó. Khách hàng doanh nghiệp tham khảo tài liệu riêng.
Mọi thay đổi về chính sách được thông báo trước 30 ngày qua SMS và ứng dụng.
Trường hợp có khác biệt giữa tài liệu này và hợp đồng đã ký, hợp đồng có hiệu lực
cao hơn. Khách hàng nên kiểm tra phiên bản mới nhất trong mục Trợ giúp.

## Thời hạn thanh toán
Hóa đơn được phát hành vào ngày 1 hàng tháng cho kỳ sử dụng của tháng trước.
Thời hạn thanh toán là ngày 15 hàng tháng. Nếu quá hạn 7 ngày, dịch vụ sẽ bị
tạm ngưng một chiều, nghĩa là khách hàng vẫn nhận được cuộc gọi nhưng không gọi
đi được. Quá hạn 15 ngày, dịch vụ bị tạm ngưng hai chiều và phát sinh phí khôi
phục 20.000 đồng. Quá hạn 60 ngày, hợp đồng bị chấm dứt và số điện thoại được
thu hồi về kho số sau 30 ngày tiếp theo.

## Các hình thức thanh toán
| Hình thức | Phí giao dịch | Thời gian ghi nhận | Hạn mức mỗi lần |
| --- | --- | --- | --- |
| Ứng dụng My Viettel | Miễn phí | Tức thì | 20.000.000 đồng |
| Chuyển khoản ngân hàng | Theo ngân hàng | 1-2 giờ | Theo ngân hàng |
| Điểm giao dịch Viettel | Miễn phí | Tức thì | Không giới hạn |
| Ví điện tử liên kết | Miễn phí | Tức thì | 10.000.000 đồng |
| Thẻ tín dụng quốc tế | 1,1% giá trị | Tức thì | 50.000.000 đồng |
| Ủy nhiệm thu ngân hàng | Miễn phí | Trong ngày | Không giới hạn |
| Cửa hàng tiện lợi đối tác | 3.000 đồng | Tức thì | 5.000.000 đồng |
| Bưu cục VNPost | 5.000 đồng | 1 ngày làm việc | 10.000.000 đồng |

## Quy trình khiếu nại hóa đơn
1. Đăng nhập ứng dụng My Viettel và vào mục Hóa đơn, chọn kỳ hóa đơn cần khiếu nại.
2. Chọn Khiếu nại, điền lý do cụ thể và đính kèm ảnh chứng từ nếu có.
3. Hệ thống cấp mã khiếu nại trong vòng 24 giờ làm việc và gửi qua SMS.
4. Bộ phận chăm sóc khách hàng liên hệ xác minh trong vòng 2 ngày làm việc.
5. Kết quả xử lý được gửi qua SMS trong vòng 5 ngày làm việc kể từ ngày xác minh.
6. Nếu khiếu nại được chấp nhận, số tiền được hoàn vào kỳ hóa đơn tiếp theo.
7. Nếu không đồng ý với kết quả, khách hàng có thể yêu cầu phúc tra một lần duy nhất.
8. Kết quả phúc tra là kết quả cuối cùng và được gửi trong vòng 10 ngày làm việc.

## Thanh toán tự động
Khi đã liên kết ngân hàng, hệ thống tự động trừ tiền vào ngày 10 hàng tháng.
Nếu số dư không đủ, hệ thống thử lại tối đa 3 lần trong 3 ngày liên tiếp.
Sau 3 lần thất bại, thanh toán tự động bị tạm dừng và khách hàng cần thanh toán
thủ công. Để bật lại, vào Cài đặt > Thanh toán > Liên kết ngân hàng.

## Hoàn tiền
Trường hợp thanh toán trùng, số tiền thừa được giữ lại dưới dạng số dư và tự động
trừ vào kỳ hóa đơn tiếp theo. Khách hàng muốn nhận lại bằng tiền mặt cần đến điểm
giao dịch kèm giấy tờ tùy thân và chứng từ thanh toán.
"""
}, {
    "source": "faq_tai_khoan.md",
    "text": """\
# Tài khoản và bảo mật

## Phạm vi áp dụng
Phần này áp dụng cho tài khoản ứng dụng My Viettel trên cả nền tảng iOS và
Android, cũng như cổng web tự phục vụ. Một số tính năng bảo mật nâng cao chỉ có
trên ứng dụng di động do yêu cầu về thiết bị tin cậy. Khách hàng sử dụng nhiều
thiết bị cần xác thực riêng cho từng thiết bị trong lần đăng nhập đầu tiên.

## Đổi mật khẩu
Vào Cài đặt > Bảo mật > Đổi mật khẩu. Hệ thống gửi mã OTP về số điện thoại đã
đăng ký, mã có hiệu lực trong 5 phút. Mật khẩu mới phải có tối thiểu 8 ký tự,
gồm chữ và số, và không được trùng với 3 mật khẩu gần nhất. Sau khi đổi thành
công, toàn bộ phiên đăng nhập trên các thiết bị khác sẽ bị đăng xuất.

## Quên mật khẩu
Chọn Quên mật khẩu ở màn hình đăng nhập, nhập số điện thoại và xác thực OTP.
Nếu số điện thoại không còn sử dụng được, khách hàng cần đến điểm giao dịch kèm
giấy tờ tùy thân. Lưu ý: việc này khác với đổi mã PIN SIM, vốn thực hiện ở mục
Cài đặt > SIM và không liên quan đến mật khẩu tài khoản ứng dụng.

## Mã lỗi đăng nhập
| Mã lỗi | Ý nghĩa | Cách xử lý | Thời gian chờ |
| --- | --- | --- | --- |
| ERR_4010 | Sai mật khẩu | Nhập lại mật khẩu | Không |
| ERR_4012 | Phiên đăng nhập hết hạn | Đăng nhập lại | Không |
| ERR_4015 | Thiết bị không được tin cậy | Xác thực OTP | Không |
| ERR_4018 | Sai mã OTP quá 5 lần | Yêu cầu mã mới | 10 phút |
| ERR_4021 | Tài khoản bị khóa tạm thời | Chờ rồi thử lại | 30 phút |
| ERR_4030 | Tài khoản bị khóa vĩnh viễn | Đến điểm giao dịch | Không |
| ERR_5003 | Máy chủ quá tải | Thử lại sau | 5 phút |
| ERR_5010 | Lỗi kết nối hệ thống cước | Thử lại sau | 15 phút |

## Bảo mật hai lớp
Bật trong Cài đặt > Bảo mật > Xác thực hai lớp. Khi bật, mỗi lần đăng nhập trên
thiết bị mới đều cần mã OTP, kể cả khi mật khẩu đúng. Khách hàng nên bật tính
năng này nếu tài khoản có liên kết ví điện tử hoặc thanh toán tự động.

## Thiết bị tin cậy
Danh sách thiết bị tin cậy nằm ở Cài đặt > Bảo mật > Thiết bị. Mỗi tài khoản
được đăng ký tối đa 5 thiết bị tin cậy. Khi thêm thiết bị thứ 6, thiết bị ít
hoạt động nhất sẽ tự động bị loại bỏ khỏi danh sách.
"""
}]

DEMO_EVAL = [
    # span nhiều câu, nằm vắt qua ranh giới cửa sổ 700 ký tự
    {"query": "khi nào bị cắt dịch vụ nếu chậm đóng tiền",
     "answer_spans": ["quá hạn 7 ngày, dịch vụ sẽ bị tạm ngưng một chiều",
                      "Quá hạn 15 ngày, dịch vụ bị tạm ngưng hai chiều và phát sinh phí khôi"]},
    # span trong bảng dài — fixed-size gần như chắc chắn cắt giữa bảng
    {"query": "các cách thanh toán và phí giao dịch",
     "answer_spans": ["Thẻ tín dụng quốc tế | 1,1% giá trị | Tức thì | 50.000.000 đồng",
                      "Cửa hàng tiện lợi đối tác | 3.000 đồng | Tức thì | 5.000.000 đồng"]},
    # quy trình 8 bước — bị chia đôi thì mất nửa câu trả lời
    {"query": "quy trình khiếu nại hóa đơn gồm những bước nào",
     "answer_spans": ["Hệ thống cấp mã khiếu nại trong vòng 24 giờ làm việc",
                      "Kết quả phúc tra là kết quả cuối cùng và được gửi trong vòng 10 ngày làm việc"]},
    {"query": "thanh toán tự động trừ tiền ngày nào và thử lại mấy lần",
     "answer_spans": ["tự động trừ tiền vào ngày 10 hàng tháng",
                      "thử lại tối đa 3 lần trong 3 ngày liên tiếp"]},
    {"query": "thanh toán trùng thì xử lý thế nào",
     "answer_spans": ["số tiền thừa được giữ lại dưới dạng số dư và tự động"]},
    {"query": "làm sao đổi mật khẩu và điều kiện mật khẩu mới",
     "answer_spans": ["Cài đặt > Bảo mật > Đổi mật khẩu",
                      "tối thiểu 8 ký tự,\ngồm chữ và số, và không được trùng với 3 mật khẩu gần nhất"]},
    {"query": "ERR_4021 nghĩa là gì",
     "answer_spans": ["ERR_4021 | Tài khoản bị khóa tạm thời | Chờ rồi thử lại | 30 phút"]},
    {"query": "sai OTP nhiều lần thì chờ bao lâu",
     "answer_spans": ["ERR_4018 | Sai mã OTP quá 5 lần | Yêu cầu mã mới | 10 phút"]},
    {"query": "được đăng ký tối đa bao nhiêu thiết bị tin cậy",
     "answer_spans": ["tối đa 5 thiết bị tin cậy"]},
]


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path)
    ap.add_argument("--eval", dest="eval_path", type=Path)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-chars", type=int, default=700)
    args = ap.parse_args()

    if args.demo:
        docs, dataset, embedder = DEMO_DOCS, DEMO_EVAL, _HashingEmbedder()
        print("demo mode: hashing embedder — read frag_rate and chunk counts, "
              "not absolute quality")
    else:
        if not (args.docs and args.eval_path):
            ap.error("--docs and --eval are required (or use --demo)")
        docs = [json.loads(l) for l in args.docs.read_text(encoding="utf-8").splitlines() if l.strip()]
        dataset = [json.loads(l) for l in args.eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        embedder = Embedder(EmbeddingConfig(model_name=args.model, device=args.device))

    missing = [r for r in dataset if not r.get("answer_spans")]
    if missing:
        print(f"ERROR {len(missing)} eval rows have no 'answer_spans'. S2 cannot use")
        print("      chunk-id labels — ids change with every chunking strategy.")
        return 2

    print(f"corpus {len(docs)} documents | eval {len(dataset)} queries, "
          f"{sum(len(r['answer_spans']) for r in dataset)} gold spans")

    base = ChunkConfig(target_chars=args.target_chars)
    rows = [run_variant(label, strat, ov, docs, dataset, embedder, base)
            for label, strat, ov in VARIANTS]

    print_table(rows)
    print_mechanism_deltas(rows)
    ok = print_verdict(rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
