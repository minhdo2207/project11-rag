"""Test VectorRetriever, embed_missing và add_many trên cả hai backend lưu trữ."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.embedding import HashingEmbedder  # noqa: E402
from src.models import Document  # noqa: E402
from src.repository import FileKnowledgeRepository, InMemoryKnowledgeRepository  # noqa: E402
from src.retriever import VectorRetriever, build_retriever, embed_missing  # noqa: E402


@pytest.fixture(params=["memory", "file"])
def repo(request, tmp_path):
    if request.param == "memory":
        return InMemoryKnowledgeRepository()
    return FileKnowledgeRepository(path=str(tmp_path / "kb"))


def _docs() -> list[Document]:
    return [
        Document("mal", "exec(base64.b64decode(payload)) # run hidden code", label="malicious"),
        Document("ben", "from setuptools import setup; setup(name='demo')", label="benign"),
        Document("rule", "rule b64exec: base64 decode then exec"),
    ]


def test_add_many_them_du_va_ghi_de_theo_id(repo):
    repo.add_many(_docs())
    repo.add_many([Document("mal", "replaced")])
    assert repo.count() == 3
    assert repo.get("mal").text == "replaced"


def test_add_many_file_backend_ben_vung(tmp_path):
    path = str(tmp_path / "kb")
    FileKnowledgeRepository(path=path).add_many(_docs())
    assert FileKnowledgeRepository(path=path).count() == 3


def test_embed_missing_chi_nhung_tai_lieu_chua_co_vector(repo):
    embedder = HashingEmbedder(dim=32)
    done = Document("done", "da nhung", embedding=[1.0] + [0.0] * 31)
    repo.add_many([*_docs(), done])

    assert embed_missing(repo, embedder, batch_size=2) == 3

    assert all(len(d.embedding) == 32 for d in repo.all())
    assert repo.get("done").embedding == [1.0] + [0.0] * 31  # không bị nhúng lại
    assert embed_missing(repo, embedder) == 0


def test_embed_missing_ben_vung_xuong_file(tmp_path):
    path = str(tmp_path / "kb")
    repo = FileKnowledgeRepository(path=path)
    repo.add_many(_docs())
    embed_missing(repo, HashingEmbedder(dim=16))
    assert all(len(d.embedding) == 16 for d in FileKnowledgeRepository(path=path).all())


def test_retriever_tra_ve_tai_lieu_gan_nhat(repo):
    embedder = HashingEmbedder(dim=256)
    repo.add_many(_docs())
    embed_missing(repo, embedder)

    result = VectorRetriever(repo, embedder).retrieve("exec(base64.b64decode(x))", top_k=1)

    assert [d.doc_id for d in result] == ["mal"]


def test_retriever_nguong_loai_bo_tai_lieu_yeu(repo):
    embedder = HashingEmbedder(dim=256)
    repo.add_many(_docs())
    embed_missing(repo, embedder)

    strict = VectorRetriever(repo, embedder, score_threshold=0.99)
    assert strict.retrieve("completely unrelated words here", top_k=3) == []


def test_build_retriever_vector_ok_con_hybrid_chua_noi():
    repo = InMemoryKnowledgeRepository()
    ok = build_retriever(Config({"pipeline": {"retrieval": "vector"}}), repo, HashingEmbedder())
    assert isinstance(ok, VectorRetriever)
    with pytest.raises(NotImplementedError, match="experiments/p3_retrieval"):
        build_retriever(Config({"pipeline": {"retrieval": "hybrid"}}), repo, HashingEmbedder())
