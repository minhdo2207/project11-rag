#!/usr/bin/env python3
"""Tải toàn bộ dữ liệu cho phần tái lập paper 2504.13769 (và PackageHallucination).

Mỗi nguồn được ghim về đúng thời điểm paper truy cập (02/2025) khi có thể, để kết quả
truy vết được. Nguồn gốc từng tập được ghi vào ``data/raw/SOURCES.json``.

Chạy:
    python scripts/download_data.py all
    python scripts/download_data.py datadog | benign | yara | advisories | ossf | pkghal

⚠️ An toàn: ``datadog`` tải về các file zip MÃ ĐỘC THẬT, mã hoá bằng mật khẩu
``infected``. Script này KHÔNG giải nén chúng. Không bao giờ cài đặt hay chạy các
package này; chỉ đọc nội dung trong bộ nhớ (xem ``scripts/inspect_data.py``).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import fcntl
import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCES = RAW / "SOURCES.json"

# Ghim theo ngày paper truy cập nguồn (tài liệu tham khảo [17] và [33] của paper)
DATADOG_COMMIT = "a8a53702efa2"  # commit ngày 19/02/2025
YARA_RELEASE = "20250216"  # release gần nhất trước 20/02/2025

USER_AGENT = "project11-rag-replication/0.1"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Chạy lệnh ngoài, dừng ngay nếu lỗi."""
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _record(name: str, info: dict) -> None:
    """Ghi nguồn gốc của một tập dữ liệu vào SOURCES.json."""
    RAW.mkdir(parents=True, exist_ok=True)
    # khoá file: các bước có thể chạy song song ở nhiều tiến trình
    with (RAW / ".sources.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(SOURCES.read_text(encoding="utf-8")) if SOURCES.exists() else {}
        info["downloaded_at"] = dt.datetime.now().isoformat(timespec="seconds")
        data[name] = info
        SOURCES.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _sparse_clone(url: str, dest: Path, paths: list[str], ref: str | None = None) -> None:
    """Clone chỉ những thư mục cần, không tải blob của phần còn lại."""
    if not dest.exists():
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)])
    _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=dest)
    _run(["git", "sparse-checkout", "set", "--no-cone", *paths], cwd=dest)
    _run(["git", "checkout", ref or "HEAD"], cwd=dest)


def download_datadog() -> None:
    """Package PyPI độc hại của DataDog, tại commit ghim."""
    dest = RAW / "datadog"
    _sparse_clone(
        "https://github.com/DataDog/malicious-software-packages-dataset",
        dest,
        ["/samples/pypi/"],
        ref=DATADOG_COMMIT,
    )
    n_zip = sum(1 for _ in (dest / "samples" / "pypi").rglob("*.zip"))
    _record(
        "datadog",
        {
            "url": "https://github.com/DataDog/malicious-software-packages-dataset",
            "commit": DATADOG_COMMIT,
            "path": "samples/pypi",
            "n_zip": n_zip,
            "zip_password": "infected",
        },
    )
    print(f"DataDog: {n_zip} file zip")


# ---------------------------------------------------------------------------
# Benign: danh sách Cerebro → tải lại mã từ PyPI
# ---------------------------------------------------------------------------

_WHEEL_TAG = re.compile(r"^(.*)-(py\d[\w.]*|cp\d+)-[\w.]+-[\w.]+$")


def parse_name_version(entry: str) -> tuple[str, str]:
    """Tách ``tên-phiên bản``; 127 dòng của Cerebro là tên file wheel, phải bỏ tag."""
    match = _WHEEL_TAG.match(entry)
    if match:
        entry = match.group(1)
    name, _, version = entry.rpartition("-")
    return name, version


def _pick_file(urls: list[dict]) -> dict | None:
    """Ưu tiên sdist (có setup.py như paper phân tích), không có thì lấy wheel thuần Python."""
    sdists = [u for u in urls if u["packagetype"] == "sdist"]
    if sdists:
        return sdists[0]
    wheels = [u for u in urls if u["packagetype"] == "bdist_wheel"]
    pure = [u for u in wheels if u["filename"].endswith("none-any.whl")]
    return (pure or wheels or [None])[0]


def _fetch_benign(entry: str, out_dir: Path) -> dict:
    name, version = parse_name_version(entry)
    row = {"entry": entry, "name": name, "version": version}
    try:
        req = urllib.request.Request(
            f"https://pypi.org/pypi/{name}/{version}/json",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            meta = json.load(resp)
    except urllib.error.HTTPError as err:
        return row | {"status": f"http_{err.code}"}
    except Exception as err:  # mạng chập chờn: ghi lại để chạy bù
        return row | {"status": f"error_{type(err).__name__}"}

    chosen = _pick_file(meta.get("urls", []))
    if chosen is None:
        return row | {"status": "no_files"}

    target = out_dir / chosen["filename"]
    row |= {
        "filename": chosen["filename"],
        "packagetype": chosen["packagetype"],
        "size": chosen["size"],
        "upload_time": chosen.get("upload_time_iso_8601", ""),
        "yanked": chosen.get("yanked", False),
    }
    if target.exists() and target.stat().st_size == chosen["size"]:
        return row | {"status": "ok_cached"}

    for attempt in range(3):
        try:
            req = urllib.request.Request(chosen["url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                blob = resp.read()
            break
        except Exception:
            if attempt == 2:
                return row | {"status": "download_failed"}
            time.sleep(2 * (attempt + 1))

    # kiểm tra toàn vẹn bằng sha256 PyPI công bố
    if hashlib.sha256(blob).hexdigest() != chosen["digests"]["sha256"]:
        return row | {"status": "sha256_mismatch"}
    target.write_bytes(blob)
    return row | {"status": "ok"}


def download_benign(workers: int = 12) -> None:
    """Tải mã package lành tính theo danh sách Cerebro (tài liệu [38] của paper)."""
    listing = ROOT / "data" / "benign" / "cerebro_pypi_benign.csv"
    entries = [r["package"] for r in csv.DictReader(listing.open(encoding="utf-8"))]
    out_dir = RAW / "benign" / "files"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with cf.ThreadPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(lambda e: _fetch_benign(e, out_dir), entries), 1):
            rows.append(row)
            if i % 200 == 0:
                print(f"benign: {i}/{len(entries)}", flush=True)

    manifest = RAW / "benign" / "manifest.csv"
    fields = [
        "entry",
        "name",
        "version",
        "status",
        "filename",
        "packagetype",
        "size",
        "upload_time",
        "yanked",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(r["status"].startswith("ok") for r in rows)
    _record(
        "benign",
        {
            "list": "Cerebro Zenodo 10.5281/zenodo.8277447, poisoning-dataset/pypi_benign.csv",
            "n_listed": len(entries),
            "n_downloaded": ok,
            "manifest": str(manifest.relative_to(ROOT)),
        },
    )
    print(f"benign: tải được {ok}/{len(entries)}")


def download_yara() -> None:
    """YARA Forge release ghim ngày 16/02/2025."""
    dest = RAW / "yara" / YARA_RELEASE
    dest.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/YARAHQ/yara-forge/releases/download/{YARA_RELEASE}"
    names = [
        "yara-forge-rules-core.zip",
        "yara-forge-rules-extended.zip",
        "yara-forge-rules-full.zip",
        "yara-forge-log.txt",
    ]
    for name in names:
        target = dest / name
        if not target.exists():
            req = urllib.request.Request(f"{base}/{name}", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp:
                target.write_bytes(resp.read())
        print(f"yara: {name} {target.stat().st_size:,} B")
    _record(
        "yara",
        {"url": "https://github.com/YARAHQ/yara-forge", "release": YARA_RELEASE, "files": names},
    )


def download_advisories() -> None:
    """GitHub Advisory Database — nhánh github-reviewed (CVE), lọc ngày ở bước kiểm tra."""
    dest = RAW / "advisory-database"
    if not dest.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/github/advisory-database",
                str(dest),
            ]
        )
    _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=dest)
    _run(["git", "sparse-checkout", "set", "--no-cone", "/advisories/github-reviewed/"], cwd=dest)
    _run(["git", "checkout", "HEAD"], cwd=dest)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=True
    ).stdout.strip()
    _record(
        "advisories",
        {
            "url": "https://github.com/github/advisory-database",
            "commit": head,
            "path": "advisories/github-reviewed",
            "note": "lọc ecosystem=PyPI, published <= 2025-02-20 khi dùng",
        },
    )


def download_ossf() -> None:
    """Báo cáo OSV mã độc PyPI của OpenSSF.

    Thay cho advisory ``type=malware`` của GitHub: repo advisory-database không chứa
    nhóm malware, còn API không token chỉ cho 60 lượt/giờ.
    """
    dest = RAW / "ossf-malicious-packages"
    if not dest.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/ossf/malicious-packages",
                str(dest),
            ]
        )
    _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=dest)
    _run(["git", "sparse-checkout", "set", "--no-cone", "/osv/malicious/pypi/"], cwd=dest)
    _run(["git", "checkout", "HEAD"], cwd=dest)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=True
    ).stdout.strip()
    _record(
        "ossf",
        {
            "url": "https://github.com/ossf/malicious-packages",
            "commit": head,
            "path": "osv/malicious/pypi",
            "note": "thay cho GitHub advisory type=malware (P7)",
        },
    )


def download_crag() -> None:
    """Benchmark CRAG task 1 & 2 (Meta, NeurIPS 2024) — bản v4 mà docs/dataset.md trỏ tới."""
    dest = RAW / "crag"
    dest.mkdir(parents=True, exist_ok=True)
    name = "crag_task_1_and_2_dev_v4.jsonl.bz2"
    target = dest / name
    url = f"https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/{name}"
    expected = 739_384_484  # đã kiểm tra bằng HTTP HEAD ngày 15/09/2026

    if not (target.exists() and target.stat().st_size == expected):
        # -C - để tải tiếp nếu lần trước đứt giữa chừng
        _run(["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(target), url])
    size = target.stat().st_size
    if size != expected:
        raise RuntimeError(f"dung lượng lệch: {size:,} ≠ {expected:,} byte")

    _record(
        "crag",
        {
            "url": url,
            "file": name,
            "bytes": size,
            "note": "task 1 & 2; split=1 là public test (bài báo dùng 1.335 câu)",
        },
    )
    print(f"CRAG: {name} {size:,} B")


def download_crag_task3() -> None:
    """CRAG task 3 — 50 trang web mỗi câu thay vì 5. Chia làm 4 phần trên GitHub.

    Bốn phần nối lại thành một file .tar.bz2 duy nhất; không giải nén ra đĩa mà đọc
    luồng thẳng vào bước nạp (xem src/crag/ingest.py --task3).
    """
    dest = RAW / "crag"
    dest.mkdir(parents=True, exist_ok=True)
    base = "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data"
    parts = {  # dung lượng lấy bằng HTTP HEAD ngày 25/09/2026
        "crag_task_3_dev_v5.tar.bz2.part0": 2_147_483_648,
        "crag_task_3_dev_v5.tar.bz2.part1": 2_147_483_648,
        "crag_task_3_dev_v5.tar.bz2.part2": 2_147_483_648,
        "crag_task_3_dev_v5.tar.bz2.part3": 1_283_117_287,
    }
    total = 0
    for name, expected in parts.items():
        target = dest / name
        if not (target.exists() and target.stat().st_size == expected):
            _run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "5",
                    "-C",
                    "-",
                    "-o",
                    str(target),
                    f"{base}/{name}",
                ]
            )
        size = target.stat().st_size
        if size != expected:
            raise RuntimeError(f"{name}: dung lượng lệch {size:,} != {expected:,}")
        total += size
        print(f"  {name} {size:,} B")

    _record(
        "crag_task3",
        {
            "base_url": base,
            "parts": list(parts),
            "bytes": total,
            "note": "50 trang/câu; nối 4 phần rồi đọc luồng tar.bz2",
        },
    )
    print(f"CRAG task 3: {total:,} B trong {len(parts)} phần")


def download_pkghal() -> None:
    """Repo PackageHallucination (USENIX Security 2025) — dataset cho Task 2."""
    dest = RAW / "PackageHallucination"
    if not dest.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/Spracks/PackageHallucination",
                str(dest),
            ]
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=True
    ).stdout.strip()
    _record(
        "package_hallucination",
        {"url": "https://github.com/Spracks/PackageHallucination", "commit": head},
    )


STEPS = {
    "crag": download_crag,
    "crag_task3": download_crag_task3,
    "datadog": download_datadog,
    "benign": download_benign,
    "yara": download_yara,
    "advisories": download_advisories,
    "ossf": download_ossf,
    "pkghal": download_pkghal,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=[*STEPS, "all"])
    args = parser.parse_args()
    for name in STEPS if args.step == "all" else [args.step]:
        print(f"\n===== {name} =====", flush=True)
        STEPS[name]()


if __name__ == "__main__":
    main()
