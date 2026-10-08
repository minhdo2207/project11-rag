"""Shared data structures for the RAG pipeline.

Defines ``Document`` — the unit of knowledge stored in the repository and
returned by retrieval. All modules (repository, retriever, detector, evaluator)
exchange documents to maintain interface consistency and type safety.

A document represents a discrete piece of evidence: a Python package snippet,
a YARA rule, a GHSA advisory. Embeddings are computed lazily and cached.
Labels are optional and used for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A unit of knowledge in the RAG system.

    Represents a discrete piece of evidence (code snippet, security advisory, rule).
    Can be retrieved via vector search and used to ground LLM responses.

    Attributes:
        doc_id: Unique document identifier (e.g., 'malicious::setuptools::a1b2c3d4').
        text: Raw content (Python code, YARA rule, advisory summary).
        embedding: Dense vector representation (empty until computed by Embedder).
        metadata: Auxiliary info (source, type, package name, file list, etc.).
        label: Ground truth label if available ('malicious', 'benign', or None).

    Example:
        doc = Document(
            doc_id="ghsa::GHSA-1234-5678-9abc",
            text="Package X has SQL injection vulnerability in v1.0.0...",
            metadata={"source": "ghsa", "type": "GHSA", "severity": "high"},
            label="malicious"
        )
        assert doc.is_labeled
        print(doc.preview())  # "Package X has SQL injection vulnerability..."
    """

    doc_id: str
    text: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    label: str | None = None

    @property
    def is_labeled(self) -> bool:
        """Returns True if this document has a ground-truth label."""
        return self.label is not None

    def preview(self, n: int = 60) -> str:
        """Return a truncated single-line preview of text for logging.

        Args:
            n: Max characters to show (default 60).

        Returns:
            Text collapsed to one line, truncated with '…' if needed.
        """
        one_line = " ".join(self.text.split())
        return one_line if len(one_line) <= n else one_line[: n - 1] + "…"
