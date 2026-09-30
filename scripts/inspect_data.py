#!/usr/bin/env python3
"""Kiểm tra thực tế dữ liệu đã tải, đối chiếu với con số trong paper 2504.13769.

Chạy: ``python scripts/inspect_data.py [datadog|benign|yara|advisories|ossf|all]``
Kết quả ghi vào ``results/data_check.json``.

⚠️ An toàn: mã độc chỉ được đọc TRONG BỘ NHỚ (``zipfile`` + ``ast.parse``). Không giải
nén ra đĩa, không import, không thực thi. ``ast.parse`` chỉ dựng cây cú pháp, không chạy
mã. Mỗi file giới hạn dung lượng giải nén để chống zip bomb.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import re
import tarfile
import warnings
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "data_check.json"

# mã nguồn package hay có chuỗi escape sai → SyntaxWarning vô hại khi ast.parse
warnings.filterwarnings("ignore", category=SyntaxWarning)

MAX_MEMBER_BYTES = 5_000_000  # bỏ qua file giải nén lớn hơn 5 MB
PASSWORD = b"infected"

# Các lời gọi paper nêu đích danh ở §3.2
SUSPICIOUS_CALLS = {
    "system",
    "Popen",
    "eval",
    "exec",
    "call",
    "run",
    "check_output",
    "b64decode",
    "urlopen",
    "get",
    "post",
}


def _py_stats(sources: dict[str, bytes]) -> dict:
    """Thống kê các file .py của một package: parse được không, có lời gọi đáng ngờ không."""
    n_py = n_ok = 0
    calls: collections.Counter = collections.Counter()
    # setup.py gốc = setup.py nông nhất; độ sâu thư mục trong archive không đồng nhất
    setups = [n for n in sources if n == "setup.py" or n.endswith("/setup.py")]
    setup_len = len(sources[min(setups, key=lambda n: n.count("/"))]) if setups else None
    for name, blob in sources.items():
        if not name.endswith(".py"):
            continue
        n_py += 1
        try:
            tree = ast.parse(blob)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        n_ok += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if attr in SUSPICIOUS_CALLS:
                    calls[attr] += 1
    return {"n_py": n_py, "n_py_parsed": n_ok, "setup_py_len": setup_len, "calls": dict(calls)}


def _read_zip(path: Path, password: bytes | None) -> tuple[dict[str, bytes], str | None]:
    sources: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.file_size > MAX_MEMBER_BYTES:
                    continue
                if info.filename.endswith((".py", "PKG-INFO", ".cfg", ".toml")):
                    sources[info.filename] = zf.read(info, pwd=password)
    except (RuntimeError, zipfile.BadZipFile, NotImplementedError, OSError) as err:
        return sources, type(err).__name__ + ": " + str(err)[:80]
    return sources, None


def _read_tar(path: Path) -> tuple[dict[str, bytes], str | None]:
    sources: dict[str, bytes] = {}
    try:
        with tarfile.open(path) as tf:
            for member in tf:
                if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                    continue
                if member.name.endswith((".py", "PKG-INFO", ".cfg", ".toml")):
                    handle = tf.extractfile(member)
                    if handle:
                        sources[member.name] = handle.read()
    except (tarfile.TarError, OSError, EOFError) as err:
        return sources, type(err).__name__
    return sources, None


def check_datadog() -> dict:
    """Paper: 1.929 package → trích được 1.242; KB setup.py 1.158 → loại 24 (còn 1.134)."""
    base = RAW / "datadog" / "samples" / "pypi"
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    zips = sorted(base.rglob("*.zip"))
    kinds = {"malicious_intent", "compromised_lib"}

    def split_path(path: Path) -> tuple[str, str]:
        # Bản 02/2025: samples/pypi/<tên>/<phiên bản>/x.zip
        # Bản mới:     samples/pypi/<loại>/<tên>/<phiên bản>/x.zip
        parts = path.relative_to(base).parts
        return (parts[0], parts[1]) if parts[0] in kinds else ("(chưa phân loại)", parts[0])

    by_kind = collections.Counter(split_path(p)[0] for p in zips)
    names = {split_path(p)[1] for p in zips}
    per_name = collections.Counter(split_path(p)[1] for p in zips)

    per_pkg, errors = [], collections.Counter()
    for path in zips:
        sources, err = _read_zip(path, PASSWORD)
        if err:
            errors[err.split(":")[0]] += 1
        stats = _py_stats(sources)
        stats["kind"] = split_path(path)[0]
        stats["read_error"] = err
        per_pkg.append(stats)

    has_py = [s for s in per_pkg if s["n_py"] > 0]
    all_parsed = [s for s in has_py if s["n_py_parsed"] == s["n_py"]]
    has_setup = [s for s in per_pkg if s["setup_py_len"] is not None]
    setup_lens = sorted(s["setup_py_len"] for s in has_setup)
    with_calls = [s for s in has_py if s["calls"]]

    return {
        "paper": {
            "packages": 1929,
            "extracted": 1242,
            "setup_py_kb": 1158,
            "setup_py_excluded_long": 24,
        },
        "manifest_names": len(manifest),
        "n_zip": len(zips),
        "n_zip_by_kind": dict(by_kind),
        "n_unique_package_names": len(names),
        "n_names_with_multiple_versions": sum(v > 1 for v in per_name.values()),
        "zip_read_errors": dict(errors),
        "n_with_python_files": len(has_py),
        "n_all_py_files_parse_ok": len(all_parsed),
        "n_with_top_level_setup_py": len(has_setup),
        "setup_py_len_p50_p95_max": (
            [
                setup_lens[len(setup_lens) // 2],
                setup_lens[int(len(setup_lens) * 0.95)],
                setup_lens[-1],
            ]
            if setup_lens
            else None
        ),
        "n_setup_py_over_32k_chars": sum(n > 32_000 for n in setup_lens),
        "n_with_suspicious_calls": len(with_calls),
        "top_calls": collections.Counter(k for s in with_calls for k in s["calls"]).most_common(8),
    }


def check_benign() -> dict:
    """Paper: 5.193 benign (nguồn [38]) → trích được 3.752. Cerebro công bố 2.398."""
    rows = list(csv.DictReader((RAW / "benign" / "manifest.csv").open(encoding="utf-8")))
    status = collections.Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"].startswith("ok")]
    kinds = collections.Counter(r["packagetype"] for r in ok)
    total_mb = sum(int(r["size"]) for r in ok) / 1e6
    years = collections.Counter((r["upload_time"] or "????")[:4] for r in ok)

    per_pkg = []
    for r in ok:
        path = RAW / "benign" / "files" / r["filename"]
        if r["filename"].endswith(".whl") or r["filename"].endswith(".zip"):
            sources, err = _read_zip(path, None)
        else:
            sources, err = _read_tar(path)
        stats = _py_stats(sources)
        stats["read_error"] = err
        per_pkg.append(stats)
    has_py = [s for s in per_pkg if s["n_py"] > 0]

    return {
        "paper": {"benign_listed": 5193, "benign_extracted": 3752},
        "cerebro_listed": len(rows),
        "status": dict(status),
        "n_downloaded": len(ok),
        "packagetype": dict(kinds),
        "total_mb": round(total_mb, 1),
        "upload_year": dict(sorted(years.items())),
        "n_read_errors": sum(1 for s in per_pkg if s["read_error"]),
        "n_with_python_files": len(has_py),
        "n_all_py_files_parse_ok": sum(1 for s in has_py if s["n_py_parsed"] == s["n_py"]),
        "n_with_top_level_setup_py": sum(1 for s in per_pkg if s["setup_py_len"] is not None),
        "n_with_suspicious_calls": sum(1 for s in has_py if s["calls"]),
    }


_RULE = re.compile(rb"^\s*(?:(?:private|global)\s+)*rule\s+\w+", re.MULTILINE)


def check_yara() -> dict:
    """Paper: 6.220 rule từ YARA Forge."""
    out = {"paper": {"rules": 6220}}
    for path in sorted((RAW / "yara").rglob("yara-forge-rules-*.zip")):
        with zipfile.ZipFile(path) as zf:
            n = sum(
                len(_RULE.findall(zf.read(i))) for i in zf.infolist() if i.filename.endswith(".yar")
            )
        out[path.stem] = n
    return out


def check_advisories() -> dict:
    """Paper: 3.474 advisory PyPI, không có malware."""
    base = RAW / "advisory-database" / "advisories" / "github-reviewed"
    total_pypi = before_cutoff = withdrawn_before = 0
    by_year: collections.Counter = collections.Counter()
    for path in base.rglob("GHSA-*.json"):
        adv = json.loads(path.read_text(encoding="utf-8"))
        if not any(
            a.get("package", {}).get("ecosystem") == "PyPI" for a in adv.get("affected", [])
        ):
            continue
        total_pypi += 1
        published = adv.get("published", "")
        by_year[published[:4]] += 1
        if published <= "2025-02-20T23:59:59Z":
            before_cutoff += 1
            if adv.get("withdrawn"):
                withdrawn_before += 1
    return {
        "paper": {"pypi_advisories": 3474},
        "pypi_reviewed_now": total_pypi,
        "pypi_reviewed_published_le_2025_02_20": before_cutoff,
        "of_which_withdrawn": withdrawn_before,
        "by_year": dict(sorted(by_year.items())),
    }


def check_ossf() -> dict:
    """Báo cáo mã độc PyPI của OpenSSF; đo trùng với DataDog và với danh sách benign."""
    base = RAW / "ossf-malicious-packages" / "osv" / "malicious" / "pypi"
    names, by_year = set(), collections.Counter()
    for path in base.rglob("*.json"):
        rep = json.loads(path.read_text(encoding="utf-8"))
        for aff in rep.get("affected", []):
            names.add(aff.get("package", {}).get("name", "").lower())
        by_year[rep.get("published", "")[:4]] += 1

    def norm(s: str) -> str:  # quy tắc chuẩn hoá tên PyPI
        return re.sub(r"[-_.]+", "-", s).lower()

    ossf = {norm(n) for n in names if n}
    dd = {
        norm(n)
        for n in json.loads((RAW / "datadog" / "samples" / "pypi" / "manifest.json").read_text())
    }
    benign = {
        norm(r["name"])
        for r in csv.DictReader((RAW / "benign" / "manifest.csv").open(encoding="utf-8"))
    }
    return {
        "n_reports": sum(by_year.values()),
        "n_unique_names": len(ossf),
        "by_year": dict(sorted(by_year.items())),
        "overlap_with_datadog_manifest": len(ossf & dd),
        "datadog_names_not_in_ossf": len(dd - ossf),
        "benign_names_flagged_malicious_by_ossf": sorted(benign & ossf)[:20],
        "n_benign_names_flagged": len(benign & ossf),
    }


PAPER_CALLS = {"Popen", "system", "eval", "exec"}  # 4 lời gọi paper nêu đích danh ở §3.2


def _setup_py_calls(sources: dict[str, bytes]) -> set[str] | None:
    """Lời gọi trong setup.py gốc thuộc PAPER_CALLS; None nếu không có/không parse được."""
    setups = [n for n in sources if n == "setup.py" or n.endswith("/setup.py")]
    if not setups:
        return None
    try:
        tree = ast.parse(sources[min(setups, key=lambda n: n.count("/"))])
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in PAPER_CALLS:
                found.add(name)
    return found


def check_setup_calls() -> dict:
    """Mức phân biệt của đặc trưng paper: tỷ lệ setup.py chứa Popen/system/eval/exec theo lớp."""
    groups = {
        "malicious": [(p, PASSWORD) for p in (RAW / "datadog" / "samples" / "pypi").rglob("*.zip")],
        "benign": [
            (RAW / "benign" / "files" / r["filename"], None)
            for r in csv.DictReader((RAW / "benign" / "manifest.csv").open(encoding="utf-8"))
            if r["status"].startswith("ok")
        ],
    }
    out = {}
    for label, items in groups.items():
        n_setup = n_any = 0
        per_call: collections.Counter = collections.Counter()
        for path, pwd in items:
            is_zip = path.suffix in {".zip", ".whl"}
            sources, _ = _read_zip(path, pwd) if is_zip else _read_tar(path)
            calls = _setup_py_calls(sources)
            if calls is None:
                continue
            n_setup += 1
            n_any += bool(calls)
            per_call.update(calls)
        out[label] = {
            "n_setup_py_parsed": n_setup,
            "n_with_any_paper_call": n_any,
            "rate_any": round(n_any / max(n_setup, 1), 3),
            "rate_per_call": {k: round(v / max(n_setup, 1), 3) for k, v in per_call.most_common()},
        }
    return out


CHECKS = {
    "datadog": check_datadog,
    "benign": check_benign,
    "yara": check_yara,
    "advisories": check_advisories,
    "ossf": check_ossf,
    "setup_calls": check_setup_calls,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=[*CHECKS, "all"])
    args = parser.parse_args()
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in CHECKS if args.what == "all" else [args.what]:
        print(f"===== {name} =====", flush=True)
        result[name] = CHECKS[name]()
        print(json.dumps(result[name], indent=2, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
