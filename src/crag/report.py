"""Dựng bảng kết quả từ các file đã chấm.

Số đo chính: **bộ ba nhãn của CRAG** — accuracy / hallucination / missing / truthfulness.
   Đây là số đo gốc của benchmark, so thẳng được với Bảng 11 của bài (Bảng 5 chỉ có
   hai model lớn; dòng Llama 3 8B nằm ở Bảng 11, Phụ lục A.3.2).

Số đo phụ, đều tính lại từ nhãn đã có, không gọi model:
   - ``score_p4``: luật +1/−4/0 — phạt bịa nặng gấp bốn
   - ``theo_nhom``: tách theo static/dynamic, loại câu hỏi, lĩnh vực
   - ``risk_coverage_B4``: quét ngưỡng cổng từ chối trên file B4 = cả họ B5
   - ``canary``: các bậc chạy trên bộ canary (``src.crag.make_canary``), nếu đã chấm

    python -m src.crag.report [--results results/crag]

Đầu ra: <results>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.evaluation import (
    answer_threshold,
    breakdown,
    load_scored,
    risk_coverage,
    three_label_report,
)

ROOT = Path(__file__).resolve().parents[2]

LADDER = ("B0", "B1", "B1.1", "B1.2", "B2", "B3", "B4", "B5")
MODEL_TAG = "llama3-8b-instruct-fp16"
# Tất cả đều là bản chạy với trần đầu ra 75 token — đúng bằng cửa sổ chấm của CRAG.
# Bản 50 token cũ vẫn còn trong results/crag/ nhưng KHÔNG dùng nữa: nó cắt thấp hơn
# cửa sổ chấm nên thổi phồng tỷ lệ bịa, và thổi phồng không đều giữa các bậc.
_FILE = re.compile(
    rf"gen_(?P<cfg>B[\d.]+)_{MODEL_TAG}_\w+?_seed1_p75(?P<data>_crag_canary)?\.scored_(?P<mode>crag|three)\.jsonl"
)
GROUP_KEYS = ("static_or_dynamic", "question_type", "domain")
PENALTIES = (1.0, 4.0)


def find_runs(results: Path) -> dict[tuple[str, str, bool], Path]:
    """(bậc, chế độ chấm, có phải canary) → file đã chấm."""
    runs = {}
    for path in results.glob("gen_*.scored_*.jsonl"):
        m = _FILE.fullmatch(path.name)
        if m and m["cfg"] in LADDER:
            runs[(m["cfg"], m["mode"], bool(m["data"]))] = path
    return runs


def table(runs: dict, mode: str, canary: bool, penalty: float = 1.0) -> dict:
    return {
        cfg: three_label_report([r["label"] for r in load_scored(runs[key])], penalty)
        for cfg in LADDER
        if (key := (cfg, mode, canary)) in runs
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "crag")
    args = parser.parse_args()

    runs = find_runs(args.results)
    if not runs:
        raise SystemExit(f"không thấy file đã chấm nào trong {args.results}")

    out = {
        "crag_three_label": table(runs, "crag", False),
        "doi_chung_3_nhan": table(runs, "three", False),
        "score_p4": {
            cfg: rep["score"] for cfg, rep in table(runs, "crag", False, penalty=4.0).items()
        },
        "theo_nhom": {
            cfg: {
                key: breakdown(load_scored(runs[(cfg, "crag", False)]), key) for key in GROUP_KEYS
            }
            for cfg in LADDER
            if (cfg, "crag", False) in runs
        },
    }

    if ("B4", "crag", False) in runs:
        b4 = load_scored(runs[("B4", "crag", False)])
        curve = risk_coverage(b4, "best_rerank", penalties=PENALTIES)
        out["risk_coverage_B4"] = {
            "ghi_chu": "mỗi điểm = một bản B5 với --gate-threshold bằng threshold, mô phỏng trên nhãn B4",
            "nguong_toi_uu_tren_test": {
                f"p{p:g}": max(curve, key=lambda pt, p=p: pt[f"score_p{p:g}"]) for p in PENALTIES
            },
            "nguong_xac_suat_ly_thuyet": {f"p{p:g}": answer_threshold(p) for p in PENALTIES},
            "points": curve,
        }

    canary = {}
    for (cfg, mode, is_canary), path in sorted(runs.items()):
        if not is_canary or mode != "crag":
            continue
        canary[cfg] = breakdown(load_scored(path), "canary_type")
    if canary:
        out["canary"] = canary

    (args.results / "metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name in ("crag_three_label", "doi_chung_3_nhan"):
        print(f"\n{name}")
        for cfg, rep in out[name].items():
            print(
                f"  {cfg:<5} acc {rep['accuracy']:5.1f}  hall {rep['hallucination']:5.1f}"
                f"  miss {rep['missing']:5.1f}  truth {rep['truthfulness']:6.1f}"
            )
    print("\nscore_p4 (+1/−4/0)")
    for cfg, s in out["score_p4"].items():
        print(f"  {cfg:<5} {s:7.1f}")
    if "risk_coverage_B4" in out:
        print("\nngưỡng cổng tốt nhất trên test (chỉ để tham khảo, không dùng để báo cáo):")
        for p, pt in out["risk_coverage_B4"]["nguong_toi_uu_tren_test"].items():
            print(
                f"  {p}: threshold {pt['threshold']:.4f}  coverage {pt['coverage']:.1f}"
                f"  risk {pt['risk']:.1f}  score {pt['score_' + p]:.1f}"
            )
    print(f"\nđã ghi {args.results / 'metrics.json'}")


if __name__ == "__main__":
    main()
