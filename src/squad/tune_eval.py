"""Bước 6–7: dò ngưỡng guardrail trên dev, đóng băng, rồi dựng bảng C0–C6.

Cấu hình:
    C0  không RAG                      C1  RAG
    C2  RAG + context check            (ctx_cos)
    C3  RAG + lexical grounding        (lexical)
    C4  RAG + HHEM grounding           (hhem_qa_max: hypothesis = câu hỏi + câu trả lời)
    C4p như C4 nhưng theo định nghĩa gốc của proposal (hhem_ans_concat: chỉ câu trả lời)
    C5  RAG + relevance                (relevance)
    C6  RAG + cả bốn check của C2, C3, C4, C5

Một check "trượt" khi điểm < ngưỡng; khi đó hệ thống ABSTAIN. FLAG được chấm như ANSWER
nên không ảnh hưởng tới các chỉ số ở đây.

Ngưỡng được chọn riêng cho từng model: giảm Hal nhiều nhất với ràng buộc
over-refusal <= BUDGET trên dev.

    python -m src.squad.tune_eval tune                # ghi results/squad/thresholds.json từ dev
    python -m src.squad.tune_eval eval --split dev    # bảng với ngưỡng đã đóng băng
    python -m src.squad.tune_eval eval --split test
    python -m src.squad.tune_eval sweep               # đánh đổi Hal / over-refusal của C4 trên dev
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from src.squad.common import MODELS, OUT, SEED, read_jsonl, write_json

BUDGET = 0.10  # trần over-refusal khi dò ngưỡng
SINGLE = {
    "C2": "ctx_cos",
    "C3": "lexical",
    "C4": "hhem_qa_max",
    "C4p": "hhem_ans_concat",
    "C5": "relevance",
}
FULL = ["ctx_cos", "lexical", "hhem_qa_max", "relevance"]
GUARDED = ["C2", "C3", "C4", "C4p", "C5", "C6"]
# Mức phân vị thử cho mỗi ngưỡng của C6; mức đầu (None) nghĩa là tắt check đó.
LEVELS = [None, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]
N_BOOT = 2000

Triple = tuple[np.ndarray, np.ndarray, np.ndarray]  # (acc, hal, miss) theo từng câu hỏi


def to_arrays(labeled: list[dict], scores: dict[str, dict]) -> dict[str, np.ndarray]:
    """Gom nhãn và điểm guardrail của cấu hình c1 thành các mảng theo câu hỏi."""
    assert not any(r["label"] == "unmatched" for r in labeled), "còn câu chưa được giám khảo chấm"
    d = {
        "answerable": np.array([r["answerable"] for r in labeled]),
        "acc": np.array([r["label"] == "accurate" for r in labeled]),
        "hal": np.array([r["label"] == "hallucination" for r in labeled]),
        "miss": np.array([r["label"] == "missing" for r in labeled]),
    }
    for s in sorted(set(SINGLE.values()) | set(FULL)):
        d[s] = np.array([scores[r["qid"]][s] for r in labeled])
    return d


def load(model: str, split: str) -> dict[str, np.ndarray]:
    lab = read_jsonl(OUT / f"labeled_{model}_c1_{split}.jsonl")
    sc = {r["qid"]: r for r in read_jsonl(OUT / f"scores_{model}_c1_{split}.jsonl")}
    return to_arrays(lab, sc)


def metrics(acc, hal, miss, answerable, boot: bool = False) -> dict:
    n = len(acc)
    a, h, m = acc.mean(), hal.mean(), miss.mean()
    out = {
        "acc": a,
        "hal": h,
        "miss": m,
        "truthfulness": a - h,
        "coverage": 1 - m,
        "selective_risk": h / (a + h) if a + h > 0 else float("nan"),
        "correct_abstention": miss[~answerable].mean(),
        "over_refusal": miss[answerable].mean(),
    }
    if boot:
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        t = acc[idx].mean(axis=1) - hal[idx].mean(axis=1)
        out["truthfulness_ci"] = [float(np.percentile(t, 2.5)), float(np.percentile(t, 97.5))]
    return out


def apply(d: dict, thresholds: dict) -> Triple:
    """Áp các ngưỡng {điểm: giá trị}; trả về (acc, hal, miss) sau guardrail."""
    block = np.zeros(len(d["acc"]), dtype=bool)
    for s, t in thresholds.items():
        if t is not None:
            block |= d[s] < t
    block &= ~d["miss"]  # câu model đã tự từ chối thì giữ nguyên
    return d["acc"] & ~block, d["hal"] & ~block, d["miss"] | block


def tune_single(d: dict, score: str, base: dict | None = None, budget: float = BUDGET):
    """Ngưỡng cao nhất của `score` mà over-refusal vẫn <= budget (Hal giảm đơn điệu theo ngưỡng).

    `base` là các ngưỡng khác đã cố định. Trả về None nếu không còn chỗ để chặn thêm câu nào,
    False nếu riêng các ngưỡng trong `base` đã vượt trần.
    """
    acc, hal, miss = apply(d, base or {})
    ans = d["answerable"]
    room = int(np.floor(budget * ans.sum() + 1e-9)) - int(miss[ans].sum())
    if room < 0:
        return False
    v = np.sort(d[score][ans & ~miss])  # câu answerable còn đang được trả lời
    if room >= len(v):
        return float(np.nextafter(v[-1], np.inf)) if len(v) else None
    t = float(v[room])  # chặn mọi câu có điểm < t, tức tối đa `room` câu answerable
    return t if (v < t).sum() <= room and t > d[score].min() else None


def tune_full(d: dict) -> dict:
    """Lưới phân vị cho ba check yếu, còn ngưỡng HHEM thì giải chính xác cho từng tổ hợp."""
    answered = ~d["miss"]
    others = [s for s in FULL if s != "hhem_qa_max"]
    cands = {
        s: [None] + [float(np.quantile(d[s][answered], q)) for q in LEVELS[1:]] for s in others
    }
    best, best_key = None, None
    for combo in itertools.product(*(cands[s] for s in others)):
        th = dict(zip(others, combo, strict=True))
        t = tune_single(d, "hhem_qa_max", base=th)
        if t is False:
            continue
        th["hhem_qa_max"] = t
        acc, hal, miss = apply(d, th)
        key = (hal.mean(), -acc.mean())  # Hal thấp nhất, hoà thì chọn Acc cao hơn
        if best_key is None or key < best_key:
            best, best_key = th, key
    return {s: best[s] for s in FULL}


def compare(a: Triple, b: Triple) -> dict:
    """So sánh cặp hai cấu hình a -> b trên cùng bộ câu hỏi; a, b là bộ (acc, hal, miss).

    McNemar chính xác (nhị thức hai phía) trên biến "câu này có bị bịa không",
    và bootstrap theo cặp cho chênh lệch Truthfulness.
    """
    from scipy.stats import binomtest

    def score(x: Triple) -> np.ndarray:  # +1 đúng, 0 từ chối, -1 bịa
        return x[0].astype(int) - x[1].astype(int)

    fixed = int((a[1] & ~b[1]).sum())  # a bịa, b không bịa
    broken = int((~a[1] & b[1]).sum())  # a không bịa, b bịa
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    diff = score(b) - score(a)
    rng = np.random.default_rng(SEED)
    boot = diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))].mean(axis=1)
    return {
        "hal_removed": fixed,
        "hal_added": broken,
        "mcnemar_p": float(p),
        "d_truthfulness": float(diff.mean()),
        "d_truthfulness_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def cmd_sweep() -> None:
    """Đường đánh đổi của C4 trên dev: nới trần over-refusal thì Hal giảm bao nhiêu."""
    out: dict = {}
    for model in MODELS:
        d = load(model, "dev")
        out[model] = {}
        print(f"\n{model} — C4 trên dev theo trần over-refusal")
        print(
            f"  {'trần':>6}{'ngưỡng':>9}{'acc':>8}{'hal':>8}{'miss':>8}"
            f"{'truthful':>10}{'corr_abst':>11}"
        )
        for budget in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
            t = tune_single(d, "hhem_qa_max", budget=budget)
            m = metrics(*apply(d, {"hhem_qa_max": t}), d["answerable"])
            out[model][str(budget)] = {"threshold": t, **m}
            print(
                f"  {budget:>6}{(t or 0):>9.3f}{m['acc']:>8.3f}{m['hal']:>8.3f}{m['miss']:>8.3f}"
                f"{m['truthfulness']:>10.3f}{m['correct_abstention']:>11.3f}"
            )
    write_json(OUT / "sweep_c4_dev.json", out)


def cmd_tune() -> None:
    out: dict = {"budget_over_refusal": BUDGET, "tuned_on": "dev", "models": {}}
    for model in MODELS:
        d = load(model, "dev")
        th: dict = {c: {s: tune_single(d, s)} for c, s in SINGLE.items()}
        th["C6"] = tune_full(d)
        out["models"][model] = th
    write_json(OUT / "thresholds.json", out)
    print(json.dumps(out, indent=2))


def cmd_eval(split: str) -> None:
    with open(OUT / "thresholds.json", encoding="utf-8") as f:
        th_all = json.load(f)["models"]
    result = {}
    cols = ["acc", "hal", "miss", "truthfulness", "coverage", "selective_risk",
            "correct_abstention", "over_refusal"]  # fmt: skip
    for model in MODELS:
        d = load(model, split)
        c1_qids = [r["qid"] for r in read_jsonl(OUT / f"labeled_{model}_c1_{split}.jsonl")]
        c0 = {r["qid"]: r["label"] for r in read_jsonl(OUT / f"labeled_{model}_c0_{split}.jsonl")}
        assert "unmatched" not in c0.values(), "còn câu chưa được giám khảo chấm"
        triples: dict[str, Triple] = {
            "C0": tuple(
                np.array([c0[q] == k for q in c1_qids])
                for k in ("accurate", "hallucination", "missing")
            ),
            "C1": (d["acc"], d["hal"], d["miss"]),
        }
        for c in GUARDED:
            triples[c] = apply(d, th_all[model][c])
        rows = {c: metrics(*t, d["answerable"], boot=True) for c, t in triples.items()}
        tests = {
            "C0->C1": compare(triples["C0"], triples["C1"]),
            **{f"C1->{c}": compare(triples["C1"], triples[c]) for c in GUARDED},
        }
        result[model] = {"metrics": rows, "paired_tests": tests}

        print(f"\n{model} — {split} (n={len(d['acc'])})")
        print(
            f"  {'':<5}" + "".join(f"{c[:12]:>13}" for c in cols) + f"{'95% CI truthfulness':>24}"
        )
        for c, m in rows.items():
            lo, hi = m["truthfulness_ci"]
            ci = f"[{lo:.3f}, {hi:.3f}]"
            print(f"  {c:<5}" + "".join(f"{m[k]:>13.3f}" for k in cols) + f"{ci:>24}")
        print(
            f"  {'so sánh cặp':<12}{'bớt bịa':>9}{'thêm bịa':>10}{'McNemar p':>12}"
            f"{'Δ truthful':>12}{'95% CI':>20}"
        )
        for name, t in tests.items():
            lo, hi = t["d_truthfulness_ci"]
            ci = f"[{lo:.3f}, {hi:.3f}]"
            print(
                f"  {name:<12}{t['hal_removed']:>9}{t['hal_added']:>10}{t['mcnemar_p']:>12.2g}"
                f"{t['d_truthfulness']:>12.3f}{ci:>20}"
            )
    write_json(OUT / f"table_{split}.json", result)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tune")
    sub.add_parser("sweep")
    ev = sub.add_parser("eval")
    ev.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()
    if args.cmd == "tune":
        cmd_tune()
    elif args.cmd == "sweep":
        cmd_sweep()
    else:
        cmd_eval(args.split)


if __name__ == "__main__":
    main()
