"""Backend lưu trữ qua file (JSON).

Dữ liệu bền vững giữa các lần chạy. Mỗi thao tác ghi sẽ đồng bộ xuống một file
JSON duy nhất — đủ dùng cho quy mô của project và dễ kiểm tra bằng mắt. Có thể
thay bằng Chroma/SQLite sau này mà không đụng tới business logic, nhờ chung
interface ``KnowledgeRepository``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..models import Document
from ._similarity import rank_by_similarity
from .base import KnowledgeRepository


class FileKnowledgeRepository(KnowledgeRepository):
    """Lưu tài liệu vào ``<path>/knowledge.json``."""

    def __init__(self, path: str = "data/kb") -> None:
        self._dir = Path(path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "knowledge.json"
        self._store: dict[str, Document] = self._load()

    def _load(self) -> dict[str, Document]:
        if not self._file.exists():
            return {}
        raw = json.loads(self._file.read_text(encoding="utf-8"))
        return {doc_id: Document(**data) for doc_id, data in raw.items()}

    def _flush(self) -> None:
        raw = {doc_id: asdict(doc) for doc_id, doc in self._store.items()}
        self._file.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, document: Document) -> None:
        self._store[document.doc_id] = document
        self._flush()

    def get(self, doc_id: str) -> Document | None:
        return self._store.get(doc_id)

    def all(self) -> list[Document]:
        return list(self._store.values())

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        return rank_by_similarity(self.all(), query_embedding, top_k)

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
        self._flush()
