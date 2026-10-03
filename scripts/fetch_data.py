"""
Script tải toàn bộ data về thư mục data/
Chạy: python scripts/fetch_data.py

Yêu cầu: pip install requests
"""

import json
import time
from pathlib import Path

import requests

# ============================================================
# CẤU HÌNH — chỉ cần sửa GITHUB_TOKEN
# ============================================================
GITHUB_TOKEN = "....."
DATA_DIR = Path("data")

# URL gốc của repo paper
RAW_BASE = "https://raw.githubusercontent.com/malexternalsc/mal-LLM/main/RQ_experiments/data"


# ============================================================
# TIỆN ÍCH
# ============================================================
def download_file(url: str, dest: Path):
    """Tải file từ URL về dest, bỏ qua nếu đã có."""
    if dest.exists():
        print(f"  [✓] Đã có: {dest.name}")
        return
    print(f"  [↓] Đang tải: {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  [✓] Xong: {dest.name}")


# ============================================================
# PHẦN 1: MALICIOUS PACKAGES
# Nguồn: repo gốc của paper (malexternalsc/mal-LLM)
# Dùng cho: KB MaliciousCode
# ============================================================
def fetch_malicious():
    print("\n[MALICIOUS PACKAGES]")
    files = [
        "train_malicious_packages_final.json",
        "test_malicious_packages_final.json",
    ]
    for filename in files:
        url = f"{RAW_BASE}/{filename}"
        dest = DATA_DIR / "malicious" / filename
        download_file(url, dest)


# ============================================================
# PHẦN 2: BENIGN PACKAGES
# Nguồn: repo gốc của paper (malexternalsc/mal-LLM)
# Dùng cho: KB BenignCode
# ============================================================
def fetch_benign():
    print("\n[BENIGN PACKAGES]")
    files = [
        "train_benign_packages_final.json",
        "test_benign_packages_final.json",
    ]
    for filename in files:
        url = f"{RAW_BASE}/{filename}"
        dest = DATA_DIR / "benign" / filename
        download_file(url, dest)


# ============================================================
# PHẦN 3: YARA RULES
# Nguồn: file có sẵn trong repo của nhóm (data/yara/)
# Dùng cho: KB YARA
# ============================================================
def check_yara():
    print("\n[YARA RULES]")
    yara_path = DATA_DIR / "yara" / "sample-yara-rules-core.yar"
    if yara_path.exists():
        print(f"  [✓] Đã có: {yara_path.name}")
    else:
        print("  [✗] Thiếu file YARA — đặt file .yar vào data/yara/")


# ============================================================
# PHẦN 4: GHSA ADVISORIES
# Nguồn: GitHub Advisory Database API
# Dùng cho: KB GHSA (~185,000 advisories)
# ============================================================
def fetch_ghsa():
    print("\n[GHSA ADVISORIES]")
    out_path = DATA_DIR / "ghsa_all.json"

    if out_path.exists():
        print(f"  [✓] Đã có: {out_path.name}")
        return

    if GITHUB_TOKEN == "ghp_...":
        print("  [!] Chưa điền GITHUB_TOKEN — bỏ qua GHSA")
        return

    print("  [↓] Bắt đầu tải GHSA (1850 pages x 100 advisories)...")
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    all_advisories = []
    page = 1

    while True:
        url = f"https://api.github.com/advisories?ecosystem=pip&per_page=100&page={page}"
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code != 200:
            print(f"  [!] Lỗi trang {page}: {resp.status_code}")
            break

        data = resp.json()
        if not data:
            break

        all_advisories.extend(data)
        print(f"  ... trang {page} — tổng: {len(all_advisories)}", end="\r")
        page += 1
        time.sleep(0.3)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_advisories, f, ensure_ascii=False, indent=2)

    print(f"\n  [✓] Đã lưu {len(all_advisories)} advisories → {out_path}")


# ============================================================
# CHẠY TẤT CẢ
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Fetch data cho Knowledge Base — Project 11")
    print("=" * 50)

    DATA_DIR.mkdir(exist_ok=True)

    fetch_malicious()  # KB MaliciousCode
    fetch_benign()  # KB BenignCode
    check_yara()  # KB YARA
    fetch_ghsa()  # KB GHSA

    print("\n[✓] Hoàn tất! Chạy ingest để nạp vào Knowledge Base:")
    print(
        '    python -c "from src.repository import KnowledgeRepository; '
        "from src.ingest import ingest_all; "
        'repo = KnowledgeRepository(); print(ingest_all(repo))"'
    )
