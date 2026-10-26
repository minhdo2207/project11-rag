"""Bước 8: chỉ số bổ sung của P5 trên kết quả đã gán nhãn — không gọi lại model sinh.

Ba phần, đều tính trên nhãn của C1 (RAG) và điểm guardrail đã có:

- Điểm có penalty (luật +1/−4/0) cho C0, C1, C4 — kèm một bản C4 dò lại ngưỡng HHEM trên dev
  để tối đa điểm +1/−4/0 thay vì truthfulness.
- Đường risk–coverage của HHEM trên test (mỗi ngưỡng là một bản C4).
- Detector LettuceDetect (experiments/p3_retrieval/rag_detector) làm chính sách từ chối SAU
  khi sinh: câu bị gắn cờ thì từ chối. τ dò trên dev với cùng trần over-refusal như tune_eval.

    python -m src.squad.p5_report export      # → results/squad/detector_in/<model>_c1_<split>.jsonl
    # chạy detector cho từng file, mỗi file một thư mục đầu ra:
    #   python run_detect.py --scenario S4 --input <file> --output-dir results/squad/detector_out/<model>_c1_<split>
    python -m src.squad.p5_report report      # → results/squad/p5_metrics.json

Thiếu kết quả detector thì phần detector được bỏ qua.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.evaluation import risk_coverage, three_label_report
from src.squad.common import MODELS, OUT, PROC, read_jsonl, write_json, write_jsonl
from src.squad.tune_eval import BUDGET

PENALTY = 4.0
DET_IN = OUT / "detector_in"
DET_OUT = OUT / "detector_out"


def gate(labels: list[str], block: list[bool]) -> list[str]:
    """Câu bị chặn thành "missing", câu còn lại giữ nhãn."""
    return ["missing" if b else lab for lab, b in zip(labels, block, strict=True)]


def over_refusal(labels: list[str], answerable: list[bool]) -> float:
    ans = [lab for lab, a in zip(labels, answerable, strict=True) if a]
    return sum(lab == "missing" for lab in ans) / max(len(ans), 1)


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(điểm của một phần tử ``pos`` > điểm của một phần tử ``neg``), hoà tính nửa."""
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float(((p > n) + 0.5 * (p == n)).mean())


def tune_flag_tau(labels, answerable, conf, budget: float = BUDGET) -> float | None:
    """τ của detector cho truthfulness cao nhất trên dev mà over-refusal vẫn <= budget."""
    best = None
    for tau in np.round(np.arange(0.05, 1.0, 0.01), 2):
        labs = gate(labels, [c >= tau for c in conf])
        if over_refusal(labs, answerable) <= budget:
            t = three_label_report(labs)["truthfulness"]
            if best is None or t > best[1]:
                best = (float(tau), t)
    return best[0] if best else None


def tune_for_penalty(labels, score, penalty: float = PENALTY) -> float:
    """Ngưỡng chặn (điểm < ngưỡng) cho điểm acc − penalty·hal cao nhất."""
    return max(
        sorted(set(score)),
        key=lambda t: three_label_report(gate(labels, [s < t for s in score]), penalty)["score"],
    )


def summarize(labels: list[str], answerable: list[bool]) -> dict:
    r1, r4 = three_label_report(labels), three_label_report(labels, PENALTY)
    keys = ("accuracy", "hallucination", "missing", "truthfulness")
    return {k: round(r1[k], 1) for k in keys} | {
        "score_p4": round(r4["score"], 1),
        "over_refusal": round(over_refusal(labels, answerable), 3),
    }


def load(model: str, split: str) -> list[dict]:
    rows = read_jsonl(OUT / f"labeled_{model}_c1_{split}.jsonl")
    sc = {r["qid"]: r for r in read_jsonl(OUT / f"scores_{model}_c1_{split}.jsonl")}
    det_path = DET_OUT / f"{model}_c1_{split}" / "S4_results.jsonl"
    det = {r["id"]: r for r in read_jsonl(det_path)} if det_path.exists() else {}
    for r in rows:
        r["hhem"] = sc[r["qid"]]["hhem_qa_max"]
        r["det"] = det[r["qid"]]["max_confidence"] if det else None
    return rows


def cmd_export() -> None:
    kb = {p["para_id"]: p["text"] for p in read_jsonl(PROC / "kb.jsonl")}
    for split in ("dev", "test"):
        qs = {q["qid"]: q for q in read_jsonl(PROC / f"questions_{split}.jsonl")}
        for model in MODELS:
            rows = [
                {
                    "id": r["qid"],
                    "question": qs[r["qid"]]["question"],
                    "contexts": [kb[i] for i in r["ctx_ids"]],  # đúng 5 đoạn đã đưa cho model
                    "answer": r["answer"],
                    "correct": r["label"] == "accurate",
                    **({"decision": "abstain"} if r["label"] == "missing" else {}),
                }
                for r in read_jsonl(OUT / f"labeled_{model}_c1_{split}.jsonl")
            ]
            write_jsonl(DET_IN / f"{model}_c1_{split}.jsonl", rows)
    print(f"đã ghi {DET_IN}")


def cmd_report() -> None:
    with open(OUT / "thresholds.json", encoding="utf-8") as f:
        th = json.load(f)["models"]
    out = {}
    for model in MODELS:
        dev, test = load(model, "dev"), load(model, "test")
        has_det = dev[0]["det"] is not None and test[0]["det"] is not None
        t4 = th[model]["C4"]["hhem_qa_max"]
        t4p = tune_for_penalty([r["label"] for r in dev], [r["hhem"] for r in dev])
        tau = (
            tune_flag_tau(
                [r["label"] for r in dev], [r["answerable"] for r in dev], [r["det"] for r in dev]
            )
            if has_det
            else None
        )
        res: dict = {"thresholds": {"C4": t4, "C4_p4": t4p, "detector_tau": tau}}
        for split, rows in (("dev", dev), ("test", test)):
            labels = [r["label"] for r in rows]
            answerable = [r["answerable"] for r in rows]
            hhem = [r["hhem"] for r in rows]
            c0 = {
                r["qid"]: r["label"] for r in read_jsonl(OUT / f"labeled_{model}_c0_{split}.jsonl")
            }
            table = {
                "C0": summarize([c0[r["qid"]] for r in rows], answerable),
                "C1": summarize(labels, answerable),
                "C4": summarize(gate(labels, [h < t4 for h in hhem]), answerable),
                "C4_p4": summarize(gate(labels, [h < t4p for h in hhem]), answerable),
            }
            ans = [r for r in rows if r["label"] != "missing"]
            hal = [r for r in ans if r["label"] == "hallucination"]
            acc = [r for r in ans if r["label"] == "accurate"]
            separation = {"1-hhem": auroc([-r["hhem"] for r in hal], [-r["hhem"] for r in acc])}
            if has_det and tau is not None:
                det = [r["det"] for r in rows]
                table["C1+detector"] = summarize(gate(labels, [d >= tau for d in det]), answerable)
                table["C4+detector"] = summarize(
                    gate(labels, [h < t4 or d >= tau for h, d in zip(hhem, det, strict=True)]),
                    answerable,
                )
                separation["detector"] = auroc([r["det"] for r in hal], [r["det"] for r in acc])
            res[split] = {"table": table, "auroc_hal_vs_acc": separation}
        curve = risk_coverage(test, "hhem")
        res["risk_coverage_test"] = {
            "ghi_chu": "mỗi điểm = C4 với một ngưỡng HHEM; điểm tốt nhất chọn trên test, chỉ tham khảo",
            "best_p1": max(curve, key=lambda p: p["score_p1"]),
            "best_p4": max(curve, key=lambda p: p["score_p4"]),
            "points": curve,
        }
        out[model] = res

        print(f"\n{model} — test")
        for name, m in res["test"]["table"].items():
            print(
                f"  {name:<12} acc {m['accuracy']:5.1f}  hal {m['hallucination']:5.1f}"
                f"  miss {m['missing']:5.1f}  truth {m['truthfulness']:6.1f}"
                f"  +1/−4/0 {m['score_p4']:7.1f}  over_ref {m['over_refusal']:.3f}"
            )
        print(
            "  AUROC bịa vs đúng:",
            {k: round(v, 3) for k, v in res["test"]["auroc_hal_vs_acc"].items()},
        )
    write_json(OUT / "p5_metrics.json", out)
    print(f"\nđã ghi {OUT / 'p5_metrics.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    sub.add_parser("report")
    args = ap.parse_args()
    cmd_export() if args.cmd == "export" else cmd_report()


if __name__ == "__main__":
    main()
