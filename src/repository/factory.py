"""Dung KnowledgeRepository tu config.

Day la cho the hien yeu cau "doi backend chi sua 1 dong": business logic goi
build_repository(cfg) va nhan ve mot KnowledgeRepository, khong quan tam ben duoi
la RAM, file JSON hay Qdrant.
"""

from __future__ import annotations

from ..config import Config
from .base import KnowledgeRepository
from .file_based import FileKnowledgeRepository
from .in_memory import InMemoryKnowledgeRepository


def build_repository(cfg: Config) -> KnowledgeRepository:
    backend = cfg.backend
    path = cfg.get("storage.path", "data/kb")

    if backend == "memory":
        return InMemoryKnowledgeRepository()
    if backend == "file":
        return FileKnowledgeRepository(path=path)
    if backend == "qdrant":
        # import lazy vi qdrant-client la dependency tuy chon,
        # khong bat ca nhom phai cai neu chi chay backend file/memory.
        from .qdrant_repo import QdrantKnowledgeRepository

        return QdrantKnowledgeRepository(path=path)

    raise ValueError(f"Backend khong ho tro: {backend!r} (memory | file | qdrant)")
