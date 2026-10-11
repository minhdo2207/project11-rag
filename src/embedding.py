"""P3 — Nhúng văn bản/mã nguồn thành vector.

Hai bản hiện thực sau interface ``Embedder``:

  HashingEmbedder   băm token vào vector cố định. Không cần model, chạy offline, kết quả
                    ổn định giữa các lần chạy; dùng cho test và demo. Không có ngữ nghĩa.
  OllamaEmbedder    model nhúng thật chạy qua Ollama (vd: bge-m3, nomic-embed-text).
"""

from __future__ import annotations

import math
import re
import zlib
from abc import ABC, abstractmethod

from ._ollama import DEFAULT_HOST, OllamaError, post_json
from .config import Config

# Định danh, số, hoặc một ký tự đơn lẻ (dấu, toán tử): đủ cho mã nguồn.
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")


class Embedder(ABC):
    """Chuyển một đoạn text thành vector số."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Nhúng một đoạn text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Nhúng nhiều đoạn (mặc định gọi lần lượt; bản hiện thực có thể tối ưu)."""
        return [self.embed(t) for t in texts]


class HashingEmbedder(Embedder):
    """Bag-of-tokens được băm vào ``dim`` chiều rồi chuẩn hoá L2."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            # crc32 thay vì hash(): hash() của Python đổi seed mỗi tiến trình nên vector
            # (và mọi kết quả truy xuất) sẽ khác nhau giữa các lần chạy.
            vec[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class OllamaEmbedder(Embedder):
    """Gọi ``/api/embed`` của Ollama."""

    def __init__(
        self, model: str = "bge-m3", host: str = DEFAULT_HOST, timeout: float = 120.0
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = post_json(
            f"{self.host}/api/embed", {"model": self.model, "input": texts}, self.timeout
        )
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError(f"Ollama trả số vector không khớp với {len(texts)} đầu vào")
        return vectors


def build_embedder(cfg: Config) -> Embedder:
    """Dựng embedder từ mục ``embedding`` trong config."""
    provider = cfg.get("embedding.provider", "ollama")
    if provider == "hashing":
        return HashingEmbedder(dim=cfg.get("embedding.dim", 256))
    if provider == "ollama":
        return OllamaEmbedder(
            model=cfg.get("embedding.model", "bge-m3"),
            host=cfg.get("embedding.host", DEFAULT_HOST),
            timeout=cfg.get("embedding.timeout", 120.0),
        )
    raise ValueError(f"Embedding provider không hỗ trợ: {provider!r} (hashing | ollama)")
