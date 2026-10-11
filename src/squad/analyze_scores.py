"""Đo khả năng tách câu đúng / câu bịa của từng điểm guardrail (AUC), trước khi dò ngưỡng.

Cần chạy src.squad.label và src.squad.guardrail_scores trước.

    python -m src.squad.analyze_scores --split dev

Đầu ra: results/squad/score_auc_<split>.json
"""

from __future__ import annotations

import argparse

import numpy as np

from src.squad.common import OUT, read_jsonl, write_json

SCORES = [
    "ctx_cos",
    "lexical",
    "relevance",
    "hhem_ans_concat",
    "hhem_ans_max",
    "hhem_qa_concat",
    "hhem_qa_max",
]


def auc(pos, neg) -> float:
    """Xác suất một câu đúng có điểm cao hơn một câu bịa (0,5 = ngẫu nhiên)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(wins + 0.5 * ties)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()

    result = {}
    for path in sorted(OUT.glob(f"scores_*_c1_{args.split}.jsonl")):
        name = path.stem.replace("scores_", "")
        sc = {r["qid"]: r for r in read_jsonl(path)}
        lab = read_jsonl(OUT / f"labeled_{name}.jsonl")
        answered = [r for r in lab if r["label"] != "missing"]
        good = [sc[r["qid"]] for r in answered if r["label"] == "accurate"]
        hal = [sc[r["qid"]] for r in answered if r["label"] == "hallucination"]
        unm = [sc[r["qid"]] for r in answered if r["label"] == "unmatched"]
        res: dict = {"n_accurate": len(good), "n_hallucination": len(hal), "n_unmatched": len(unm)}
        for s in SCORES:
            g, h, u = ([x[s] for x in grp] for grp in (good, hal, unm))
            res[s] = {
                "auc": round(auc(g, h), 3),
                # Cận bi quan: coi mọi câu chưa khớp (chờ giám khảo) là câu sai.
                "auc_unmatched_as_bad": round(auc(g, h + u), 3),
                "median_accurate": round(float(np.median(g)), 3),
                "median_hallucination": round(float(np.median(h)), 3),
            }
        result[name] = res

        print(f"\n{name}: accurate={len(good)} hallucination={len(hal)} unmatched={len(unm)}")
        print(f"  {'điểm':<18}{'AUC':>7}{'AUC (bi quan)':>15}{'med đúng':>10}{'med bịa':>9}")
        for s in SCORES:
            r = res[s]
            print(
                f"  {s:<18}{r['auc']:>7}{r['auc_unmatched_as_bad']:>15}"
                f"{r['median_accurate']:>10}{r['median_hallucination']:>9}"
            )

    write_json(OUT / f"score_auc_{args.split}.json", result)


if __name__ == "__main__":
    main()
