"""Lớp persistence cho knowledge base (memory / file hoán đổi được)."""

from .base import KnowledgeRepository
from .file_based import FileKnowledgeRepository
from .in_memory import InMemoryKnowledgeRepository

__all__ = [
    "KnowledgeRepository",
    "InMemoryKnowledgeRepository",
    "FileKnowledgeRepository",
]
