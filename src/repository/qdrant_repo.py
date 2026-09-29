"""Backend Qdrant (memory hoac local file, cung mot API).

Qdrant hay o cho: QdrantClient(":memory:") va QdrantClient(path=...) dung y het
nhau, nen cung mot class phuc vu duoc ca hai mode. Khi can scale thi doi sang
server (url=...) ma khong sua business logic.

Day la dependency tuy chon: `pip install qdrant-client`.
"""

from __future__ import annotations

import uuid

from ..models import Document
from .base import KnowledgeRepository

COLLECTION = "knowledge"


class QdrantKnowledgeRepository(KnowledgeRepository):
    def __init__(self, path: str | None = None, dim: int = 1024):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:  # pragma: no cover - chi chay khi thieu lib
            raise ImportError("Can qdrant-client cho backend nay: pip install qdrant-client") from e

        # path=None -> chay trong RAM; co path -> luu ra o dia
        self._client = (
            QdrantClient(location=":memory:") if path is None else QdrantClient(path=path)
        )
        self._dim = dim
        self._Distance = Distance
        self._VectorParams = VectorParams
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION not in existing:
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=self._VectorParams(size=self._dim, distance=self._Distance.COSINE),
            )

    @staticmethod
    def _point_id(doc_id: str) -> str:
        # Qdrant can id la uuid/int; map doc_id string sang uuid on dinh.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))

    def add(self, document: Document) -> None:
        from qdrant_client.models import PointStruct

        payload = {
            "doc_id": document.doc_id,
            "text": document.text,
            "metadata": document.metadata,
            "label": document.label,
        }
        self._client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=self._point_id(document.doc_id), vector=document.embedding, payload=payload
                )
            ],
        )

    def _to_doc(self, payload: dict) -> Document:
        return Document(
            doc_id=payload["doc_id"],
            text=payload["text"],
            metadata=payload.get("metadata") or {},
            label=payload.get("label"),
        )

    def get(self, doc_id: str) -> Document | None:
        res = self._client.retrieve(collection_name=COLLECTION, ids=[self._point_id(doc_id)])
        return self._to_doc(res[0].payload) if res else None

    def all(self) -> list[Document]:
        docs, offset = [], None
        while True:
            points, offset = self._client.scroll(
                collection_name=COLLECTION, limit=256, offset=offset, with_payload=True
            )
            docs.extend(self._to_doc(p.payload) for p in points)
            if offset is None:
                break
        return docs

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        hits = self._client.search(
            collection_name=COLLECTION, query_vector=query_embedding, limit=top_k
        )
        return [self._to_doc(h.payload) for h in hits]

    def count(self) -> int:
        return self._client.count(collection_name=COLLECTION).count

    def clear(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._ensure_collection()
