"""Fixture dùng chung: một server HTTP giả đóng vai Ollama chạy trên localhost.

Dùng server thật (chỉ là giả nội dung) thay vì mock urllib để test đi qua đúng đường
HTTP thật: serialize JSON, mã lỗi, timeout.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class FakeOllama:
    """Lưu request nhận được và trả về response đã đăng ký theo path."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.responses: dict[str, tuple[int, dict]] = {}
        self._server = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def reply(self, path: str, body: dict, status: int = 200) -> None:
        self.responses[path] = (status, body)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - tên do http.server quy định
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append((self.path, json.loads(self.rfile.read(length))))
                status, body = outer.responses.get(self.path, (404, {"error": "not registered"}))
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # im lặng, đừng làm nhiễu output của pytest
                pass

        return Handler


@pytest.fixture
def fake_ollama():
    server = FakeOllama()
    yield server
    server.close()
