"""P4 — Giao tiếp với LLM sinh (Qwen/Mistral/Llama qua Ollama).

STUB: định nghĩa interface ``LLMClient``. P4 sẽ hiện thực bằng client Ollama
hoặc vLLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Gửi prompt tới mô hình sinh và nhận về text."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Sinh câu trả lời cho ``prompt``."""
