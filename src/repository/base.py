"""Abstract interface for knowledge base storage.

This is the core of our storage abstraction strategy: business logic
(retriever, detector, evaluator) depends ONLY on this abstract interface,
never on concrete backends.

Benefit: Swapping backends (memory ↔ file ↔ Qdrant) requires only changing
the factory, not touching retriever/detector logic. This is **dependency inversion**
in action and fulfills the non-functional requirement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Document


class KnowledgeRepository(ABC):
    """Abstract interface for storing and retrieving knowledge documents.

    Implementations:
      - InMemoryKnowledgeRepository: Fast, ephemeral (in-process dict)
      - FileKnowledgeRepository: Persistent (JSON on disk)
      - QdrantKnowledgeRepository: Vector database (in-memory or persistent)

    All methods must produce **identical behavior** across backends.
    Shared implementation (e.g., _similarity.py) ensures this invariant.
    """

    @abstractmethod
    def add(self, document: Document) -> None:
        """Add or overwrite a document by doc_id.

        Args:
            document: Document to store.
        """

    @abstractmethod
    def get(self, doc_id: str) -> Document | None:
        """Retrieve a document by id.

        Args:
            doc_id: Unique document identifier.

        Returns:
            Document if found, None otherwise.
        """

    @abstractmethod
    def all(self) -> list[Document]:
        """Retrieve all stored documents.

        Returns:
            List of all documents (order unspecified).
        """

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Document]:
        """Search documents by cosine similarity to query embedding.

        Args:
            query_embedding: Dense vector (must match embedding dimension).
            top_k: Number of results to return.

        Returns:
            Ranked list of top_k most similar documents.
        """

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored documents."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored documents."""
