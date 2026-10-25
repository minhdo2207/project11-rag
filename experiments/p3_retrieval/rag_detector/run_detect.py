"""Run the LettuceDetect hallucination detector over RAG scenarios S1-S7.

One detector, one configuration, shared by every scenario:

    python run_detect.py --scenario S3                 # reads data/s3.jsonl
    python run_detect.py --scenario S3 --input out/s3_generations.jsonl
    python run_detect.py --all                         # S1..S7 + comparison table
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from detection import build_detector, detect_records, read_jsonl, write_jsonl
from metrics import summarize

HERE = Path(__file__).resolve().parent
SCENARIOS = {
    "S1": "dense + BM25 + RRF",
    "S2": "structure-aware chunking, atomic blocks, heading enrichment",
    "S3": "cross-encoder rerank (replace / RRF fuse)",
    "S4": "metadata pre-filter (scope, version)",
    "S5": "query understanding (diacritic folding, dictionary, multi-variant)",
    "S6": "adaptive-k, MMR, U-shape ordering",
    "S7": "calibration and 4-level answer decision (may_assert)",
}
COLS = ["accuracy", "hallucination_rate", "rejection_rate", "adjusted_accuracy"]


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def _find_input(input_dir: Path, scenario: str) -> Path | None:
    for name in (f"{scenario.lower()}.jsonl", f"{scenario}.jsonl"):
        if (input_dir / name).is_file():
            return input_dir / name
    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--scenario", choices=sorted(SCENARIOS), help="run a single scenario")
    target.add_argument("--all", action="store_true", help="run S1..S7")
    p.add_argument("--input", type=Path, help="JSONL for --scenario (overrides --input-dir)")
    p.add_argument("--input-dir", type=Path, default=HERE / "data", help="folder with s1.jsonl..s7.jsonl")
    p.add_argument("--output-dir", type=Path, default=HERE / "results")
    p.add_argument("--method", choices=["transformer", "llm"], default="transformer")
    p.add_argument("--model", default="KRLabsOrg/lettucedect-v2-mmbert-base")
    p.add_argument("--lang", default="en", help="detector prompt language")
    p.add_argument("--device", default=None)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--tau", type=float, default=0.5, help="hallucinated if any span >= tau")
    p.add_argument("--taus", type=float, nargs="*", default=[0.5, 0.7, 0.9], help="tau sweep")
    p.add_argument("--resume", action="store_true", help="reuse ids already in the results file")
    return p.parse_args(argv)


def _run_scenario(detector, scenario: str, path: Path, args: argparse.Namespace) -> dict:
    records = read_jsonl(path)
    results_path = args.output_dir / f"{scenario}_results.jsonl"
    done = {}
    if args.resume and results_path.is_file():
        done = {str(r["id"]): r for r in read_jsonl(results_path)}

    print(f"[{scenario}] {SCENARIOS[scenario]}: {len(records)} questions from {path}")
    results = detect_records(detector, records, done)
    write_jsonl(results_path, results)

    summary = summarize(results, args.tau, args.taus)
    summary.update({"scenario": scenario, "model": args.model, "method": args.method})
    summary_path = args.output_dir / f"{scenario}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _write_comparison(summaries: list[dict], args: argparse.Namespace) -> None:
    keys = ["scenario", "tau", "n", "rejected", "hallucinated", "correct", *COLS]
    csv_path = args.output_dir / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            for row in s["tau_sweep"]:
                w.writerow({"scenario": s["scenario"], **{k: row.get(k) for k in keys[1:]}})

    print(f"\nComparison at tau={args.tau} (percent; full tau sweep in {csv_path}):")
    print("scenario | " + " | ".join(COLS))
    for s in summaries:
        op = s["operating_point"]
        print(f"{s['scenario']} | " + " | ".join(_fmt(op[c]) for c in COLS))


def main(argv: list[str] | None = None) -> None:
    """Run the detector for one scenario or for all of them."""
    args = _parse_args(argv)

    if args.all:
        jobs = {s: _find_input(args.input_dir, s) for s in SCENARIOS}
        missing = [s for s, path in jobs.items() if path is None]
        jobs = {s: path for s, path in jobs.items() if path is not None}
        if missing:
            print(f"No input in {args.input_dir} for: {', '.join(missing)} (skipped)")
    else:
        path = args.input or _find_input(args.input_dir, args.scenario)
        jobs = {args.scenario: path} if path else {}
        if not path:
            sys.exit(f"No input for {args.scenario}: pass --input or put it in {args.input_dir}")

    if not jobs:
        sys.exit(f"Nothing to run. Put s1.jsonl..s7.jsonl in {args.input_dir} (see README.md).")
    for path in jobs.values():
        if not path.is_file():
            sys.exit(f"Input not found: {path}")

    detector = build_detector(args.method, args.model, args.device, args.max_length, args.lang)
    summaries = [_run_scenario(detector, s, path, args) for s, path in jobs.items()]

    if args.all:
        _write_comparison(summaries, args)
    else:
        print(json.dumps(summaries[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
