"""P2 — Data & Knowledge Base: nạp dữ liệu vào KnowledgeRepository."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from src.models import Document
from src.repository import KnowledgeRepository


def ingest_yara(repo: KnowledgeRepository, yara_dir: str | Path) -> int:
    yara_dir = Path(yara_dir)
    count = 0
    for path in sorted(yara_dir.glob("**/*.ya*")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc = Document(
            doc_id=f"yara::{path.stem}",
            text=text,
            metadata={"source": "yara", "type": "YARA", "filename": path.name},
        )
        repo.add(doc)
        count += 1
    return count


def ingest_packages_json(repo: KnowledgeRepository, json_path: str | Path, label: str) -> int:
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)
    count = 0
    for r in records:
        setup_py = r.get("setup.py", "Not Available")
        if not setup_py or setup_py == "Not Available":
            continue
        package_name = r.get("package_name", "unknown")
        content_hash = hashlib.md5(setup_py.encode()).hexdigest()[:8]
        doc = Document(
            doc_id=f"{label}::{package_name}::{content_hash}",
            text=setup_py,
            metadata={
                "source": "pypi_sample",
                "type": "MaliciousCode" if label == "malicious" else "BenignCode",
                "package": package_name,
                "file_list": r.get("file_list", []),
            },
            label=label,
        )
        repo.add(doc)
        count += 1
    return count


def ingest_ghsa(repo: KnowledgeRepository, ghsa_dir: str | Path) -> int:
    ghsa_dir = Path(ghsa_dir)
    count = 0
    for path in sorted(ghsa_dir.glob("**/*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        text = "\n\n".join(filter(None, [data.get("summary", ""), data.get("details", "")]))
        doc = Document(
            doc_id=f"ghsa::{data.get('id', path.stem)}",
            text=text,
            metadata={"source": "ghsa", "type": "GHSA", "ghsa_id": data.get("id", "")},
        )
        repo.add(doc)
        count += 1
    return count


def ingest_all(repo: KnowledgeRepository, data_dir: str | Path = "data") -> dict:
    data_dir = Path(data_dir)
    n_yara = ingest_yara(repo, data_dir / "yara")
    n_mal = ingest_packages_json(repo, data_dir / "malicious" / "train_malicious_packages_final.json", "malicious")
    n_mal += ingest_packages_json(repo, data_dir / "malicious" / "test_malicious_packages_final.json", "malicious")
    n_ben = ingest_packages_json(repo, data_dir / "benign" / "train_benign_packages_final.json", "benign")
    n_ben += ingest_packages_json(repo, data_dir / "benign" / "test_benign_packages_final.json", "benign")
    n_ghsa = ingest_ghsa(repo, data_dir / "ghsa")
    stats = {"yara": n_yara, "malicious_code": n_mal, "benign_code": n_ben, "ghsa": n_ghsa, "total": n_yara + n_mal + n_ben + n_ghsa}
    print(f"[ingest] Đã nạp: {stats}")
    return stats
