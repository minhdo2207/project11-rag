"""Test cho Qdrant backend. Tu dong skip neu chua cai qdrant-client
(vi day la dependency tuy chon)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("qdrant_client")

from src.models import Document  # noqa: E402
from src.repository.qdrant_repo import QdrantKnowledgeRepository  # noqa: E402


@pytest.fixture
def repo():
    r = QdrantKnowledgeRepository(path=None, dim=3)  # in-memory mode
    yield r
    r.clear()


def _docs():
    return [
        Document("d1", "eval(base64_payload)", embedding=[1.0, 0.0, 0.0], label="malicious"),
        Document("d2", "print(hello)", embedding=[0.0, 1.0, 0.0], label="benign"),
    ]


def test_add_get_count(repo):
    for d in _docs():
        repo.add(d)
    assert repo.count() == 2
    assert repo.get("d1").label == "malicious"
    assert repo.get("khong-co") is None


def test_search(repo):
    for d in _docs():
        repo.add(d)
    top = repo.search([1.0, 0.0, 0.0], top_k=1)
    assert top[0].doc_id == "d1"
