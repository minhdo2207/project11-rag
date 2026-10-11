"""Bước 5: gán nhãn CRAG (accurate / hallucination / missing) cho câu trả lời đã sinh.

Câu answerable, không từ chối, F1 < ngưỡng được gán "unmatched". Nếu truyền --judge thì
phán quyết của giám khảo (src.squad.judge) quyết định accurate / hallucination cho các câu
đó; câu nào giám khảo chưa chấm vẫn là "unmatched" và được báo cáo theo hai cận.

    python -m src.squad.label --split dev                       # lần đầu, trước khi chạy judge
    python -m src.squad.label --split dev --judge llama3.3:70b  # sau khi chạy judge

Đầu ra: results/squad/labeled_<model>_<config>_<split>.jsonl và summary_<split>.json
"""

from __future__ import annotations

import argparse
import collections
import re
import string

from src.squad.common import OUT, PROC, read_jsonl, write_json, write_jsonl

THETA_F1 = 0.5
_PUNCT = set(string.punctuation)


def norm(s: str) -> str:
    """Chuẩn hoá chính thức của SQuAD: chữ thường, bỏ dấu câu, bỏ mạo từ, gộp khoảng trắng."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def is_abstain(answer: str) -> bool:
    """Từ chối = câu rỗng hoặc BẮT ĐẦU bằng "I don't know" / "I do not know"."""
    a = norm(answer)
    return not a or a.startswith("i dont know") or a.startswith("i do not know")


def label(q: dict, answer: str) -> tuple[str, float | None]:
    if is_abstain(answer):
        return "missing", None
    if not q["answerable"]:
        return "hallucination", None
    best = max(f1(answer, g) for g in q["gold_answers"])
    return ("accurate" if best >= THETA_F1 else "unmatched"), round(best, 3)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    c = collections.Counter(r["label"] for r in rows)
    n_ans = sum(r["answerable"] for r in rows)
    n_un = n - n_ans
    acc, hal, miss, unm = (c[k] / n for k in ("accurate", "hallucination", "missing", "unmatched"))
    n_right_refusal = sum(r["label"] == "missing" and not r["answerable"] for r in rows)
    n_wrong_refusal = sum(r["label"] == "missing" and r["answerable"] for r in rows)
    return {
        "n": n,
        "acc": round(acc, 3),
        "hal": round(hal, 3),
        "miss": round(miss, 3),
        "unmatched": round(unm, 3),
        # Cận dưới: unmatched tính là hallucination. Cận trên: unmatched tính là accurate.
        "truthfulness_low": round(acc - hal - unm, 3),
        "truthfulness_high": round(acc + unm - hal, 3),
        "coverage": round(1 - miss, 3),
        "correct_abstention": round(n_right_refusal / max(n_un, 1), 3),
        "over_refusal": round(n_wrong_refusal / max(n_ans, 1), 3),
        "gen_s_mean": round(sum(r["gen_s"] for r in rows) / n, 3),
    }


def judge_tag(judge: str) -> str:
    """Tên model giám khảo → đoạn dùng trong tên file (vd. llama3.3:70b → llama3.3-70b)."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", judge)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--judge", default="", help="tên model giám khảo đã chạy bằng src.squad.judge")
    args = ap.parse_args()
    tag = judge_tag(args.judge)

    qs = {q["qid"]: q for q in read_jsonl(PROC / f"questions_{args.split}.jsonl")}
    summary = {}
    for path in sorted(OUT.glob(f"gen_*_{args.split}.jsonl")):
        name = path.stem.replace("gen_", "")
        jpath = OUT / f"judge_{tag}_{name}.jsonl"
        verdicts = (
            {j["qid"]: j["correct"] for j in read_jsonl(jpath)}
            if args.judge and jpath.exists()
            else {}
        )
        rows = []
        for r in read_jsonl(path):
            q = qs[r["qid"]]
            lab, best = label(q, r["answer"])
            by_judge = False
            if lab == "unmatched" and verdicts.get(r["qid"]) is not None:
                lab, by_judge = ("accurate" if verdicts[r["qid"]] else "hallucination"), True
            rows.append(
                {
                    **r,
                    "answerable": q["answerable"],
                    "label": lab,
                    "best_f1": best,
                    "by_judge": by_judge,
                    "gold_in_ctx": q["gold_para_id"] in r["ctx_ids"],
                }
            )
        write_jsonl(OUT / f"labeled_{name}.jsonl", rows)
        summary[name] = summarize(rows)

    write_json(OUT / f"summary_{args.split}.json", summary)
    cols = ["n", "acc", "hal", "miss", "unmatched", "truthfulness_low", "truthfulness_high",
            "coverage", "correct_abstention", "over_refusal", "gen_s_mean"]  # fmt: skip
    print(f"{'run':<18}" + "".join(f"{c[:11]:>12}" for c in cols))
    for name, s in summary.items():
        print(f"{name:<18}" + "".join(f"{s[c]:>12}" for c in cols))


if __name__ == "__main__":
    main()
