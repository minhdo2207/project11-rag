"""Unit test cho lớp persistence.

Điểm cốt lõi: chứng minh hai backend (InMemory và File) hành xử GIỐNG NHAU qua
cùng interface ``KnowledgeRepository`` — bằng chứng cho yêu cầu non-functional
"đổi backend không đổi hành vi".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import Document  # noqa: E402
from src.repository import (  # noqa: E402
    FileKnowledgeRepository,
    InMemoryKnowledgeRepository,
    KnowledgeRepository,
)


def _sample_docs() -> list[Document]:
    return [
        Document("d1", "eval(base64)", embedding=[1.0, 0.0, 0.0], label="malicious"),
        Document("d2", "print(hello)", embedding=[0.0, 1.0, 0.0], label="benign"),
        Document("d3", "os.system(rm)", embedding=[0.9, 0.1, 0.0], label="malicious"),
    ]


@pytest.fixture(params=["memory", "file"])
def repo(request, tmp_path) -> KnowledgeRepository:
    if request.param == "memory":
        return InMemoryKnowledgeRepository()
    return FileKnowledgeRepository(path=str(tmp_path / "kb"))


def test_add_and_get(repo: KnowledgeRepository) -> None:
    for doc in _sample_docs():
        repo.add(doc)
    assert repo.count() == 3
    assert repo.get("d1").text == "eval(base64)"
    assert repo.get("missing") is None


def test_search_ranks_by_similarity(repo: KnowledgeRepository) -> None:
    for doc in _sample_docs():
        repo.add(doc)
    results = repo.search([1.0, 0.0, 0.0], top_k=2)
    assert [d.doc_id for d in results] == ["d1", "d3"]


def test_clear(repo: KnowledgeRepository) -> None:
    repo.add(_sample_docs()[0])
    repo.clear()
    assert repo.count() == 0


def test_file_backend_persists(tmp_path) -> None:
    path = str(tmp_path / "kb")
    first = FileKnowledgeRepository(path=path)
    for doc in _sample_docs():
        first.add(doc)
    # Mở lại từ đĩa -> dữ liệu vẫn còn.
    second = FileKnowledgeRepository(path=path)
    assert second.count() == 3
    assert second.get("d3").label == "malicious"
