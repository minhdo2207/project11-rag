"""P4 — Giao tiếp với LLM sinh (Qwen/Mistral/Llama qua Ollama).

``LLMClient`` là interface; ``OllamaClient`` là bản hiện thực gọi Ollama cục bộ.
Business logic (detector) chỉ biết interface nên đổi model hay đổi nhà cung cấp
chỉ cần sửa ``config.yaml``.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from ._ollama import DEFAULT_HOST, OllamaError, post_json
from .config import Config

logger = logging.getLogger(__name__)

# Một số model (qwen3...) in phần suy luận trong <think>...</think> trước câu trả lời.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMClient(ABC):
    """Gửi prompt tới mô hình sinh và nhận về text."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Sinh câu trả lời cho ``prompt``."""


class OllamaClient(LLMClient):
    """Gọi ``/api/chat`` của Ollama, không streaming."""

    def __init__(
        self,
        model: str = "qwen2.5",
        host: str = DEFAULT_HOST,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 120.0,
        think: bool | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # None = để mặc định của model. Model không hỗ trợ suy luận sẽ báo lỗi nếu gửi cờ này.
        self.think = think

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if self.think is not None:
            payload["think"] = self.think

        data = post_json(f"{self.host}/api/chat", payload, self.timeout)
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise OllamaError(f"Phản hồi Ollama thiếu message.content: {str(data)[:200]}")
        if data.get("done_reason") == "length":
            logger.warning(
                "Câu trả lời bị cắt do chạm max_tokens=%d; tăng llm.max_tokens trong config.",
                self.max_tokens,
            )
        return _THINK_BLOCK.sub("", content).strip()


def build_llm_client(cfg: Config) -> LLMClient:
    """Dựng client từ mục ``llm`` trong config."""
    provider = cfg.get("llm.provider", "ollama")
    if provider != "ollama":
        raise ValueError(f"LLM provider không hỗ trợ: {provider!r} (ollama)")
    return OllamaClient(
        model=cfg.get("llm.model", "qwen2.5"),
        host=cfg.get("llm.host", DEFAULT_HOST),
        temperature=cfg.get("llm.temperature", 0.1),
        max_tokens=cfg.get("llm.max_tokens", 512),
        timeout=cfg.get("llm.timeout", 120.0),
        think=cfg.get("llm.think", None),
    )
