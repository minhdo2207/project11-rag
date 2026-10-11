"""HTTP helper dùng chung cho các client nói chuyện với Ollama (LLM và embedding).

Dùng urllib trong thư viện chuẩn để không phải cài thêm package chỉ để gọi một REST API.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    """Không gọi được Ollama, hoặc Ollama trả về lỗi / dữ liệu không đọc được."""


def post_json(url: str, payload: dict, timeout: float) -> dict:
    """POST ``payload`` dạng JSON, trả về JSON đã parse; mọi lỗi đều đổi thành OllamaError."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Phần body thường nói rõ nguyên nhân (vd: model chưa được pull).
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise OllamaError(f"Ollama trả HTTP {exc.code} cho {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OllamaError(f"Không kết nối được Ollama tại {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Ollama trả dữ liệu không phải JSON từ {url}") from exc
