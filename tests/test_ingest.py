"""Unit test cho ingest (P2): dùng dữ liệu giả nhỏ, backend in-memory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import ingest_ghsa, ingest_packages_json, ingest_yara  # noqa: E402
from src.repository import InMemoryKnowledgeRepository  # noqa: E402


def test_packages_skip_not_available(tmp_path):
    records = [
        {"package_name": "evil", "setup.py": "import os; os.system('x')"},
        {"package_name": "empty", "setup.py": "Not Available"},
    ]
    path = tmp_path / "pkgs.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    repo = InMemoryKnowledgeRepository()
    assert ingest_packages_json(repo, path, "malicious") == 1
    assert repo.count() == 1
    assert repo.all()[0].label == "malicious"


def test_yara_ingest(tmp_path):
    (tmp_path / "rule.yar").write_text("rule x { condition: true }", encoding="utf-8")
    repo = InMemoryKnowledgeRepository()
    assert ingest_yara(repo, tmp_path) == 1
    assert repo.get("yara::rule") is not None


def test_ghsa_ingest_and_missing_file(tmp_path):
    path = tmp_path / "ghsa_all.json"
    path.write_text(json.dumps([{"id": "GHSA-1", "summary": "bad pkg", "details": ""}]))

    repo = InMemoryKnowledgeRepository()
    assert ingest_ghsa(repo, path) == 1
    assert ingest_ghsa(repo, tmp_path / "nope.json") == 0
