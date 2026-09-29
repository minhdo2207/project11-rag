import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.repository import FileKnowledgeRepository, InMemoryKnowledgeRepository  # noqa: E402
from src.repository.factory import build_repository  # noqa: E402


def test_build_memory():
    cfg = Config({"storage": {"backend": "memory"}})
    assert isinstance(build_repository(cfg), InMemoryKnowledgeRepository)


def test_build_file(tmp_path):
    cfg = Config({"storage": {"backend": "file", "path": str(tmp_path / "kb")}})
    assert isinstance(build_repository(cfg), FileKnowledgeRepository)


def test_unknown_backend_raises():
    cfg = Config({"storage": {"backend": "mysql"}})
    with pytest.raises(ValueError):
        build_repository(cfg)
