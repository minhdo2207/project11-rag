"""Test OllamaClient trên một server Ollama giả (xem conftest.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src._ollama import OllamaError  # noqa: E402
from src.config import Config  # noqa: E402
from src.llm_client import OllamaClient, build_llm_client  # noqa: E402


def _chat_reply(content: str, done_reason: str = "stop") -> dict:
    return {"message": {"role": "assistant", "content": content}, "done_reason": done_reason}


def test_generate_gui_dung_payload(fake_ollama):
    fake_ollama.reply("/api/chat", _chat_reply("xin chao"))
    client = OllamaClient(model="qwen2.5", host=fake_ollama.url, temperature=0.2, max_tokens=64)

    assert client.generate("hello", system="be brief") == "xin chao"

    path, payload = fake_ollama.requests[0]
    assert path == "/api/chat"
    assert payload["model"] == "qwen2.5"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.2, "num_predict": 64}
    assert payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert "think" not in payload  # mặc định không gửi cờ think


def test_khong_co_system_thi_khong_gui_message_system(fake_ollama):
    fake_ollama.reply("/api/chat", _chat_reply("ok"))
    OllamaClient(host=fake_ollama.url).generate("hi")
    assert [m["role"] for m in fake_ollama.requests[0][1]["messages"]] == ["user"]


def test_cho_phep_tat_think(fake_ollama):
    fake_ollama.reply("/api/chat", _chat_reply("ok"))
    OllamaClient(host=fake_ollama.url, think=False).generate("hi")
    assert fake_ollama.requests[0][1]["think"] is False


def test_bo_khoi_think_trong_cau_tra_loi(fake_ollama):
    fake_ollama.reply("/api/chat", _chat_reply("<think>suy nghi\nnhieu dong</think>\n\nbenign"))
    assert OllamaClient(host=fake_ollama.url).generate("hi") == "benign"


def test_canh_bao_khi_bi_cat_cut(fake_ollama, caplog):
    fake_ollama.reply("/api/chat", _chat_reply("{partial", done_reason="length"))
    with caplog.at_level("WARNING"):
        OllamaClient(host=fake_ollama.url, max_tokens=8).generate("hi")
    assert "max_tokens=8" in caplog.text


def test_model_khong_ton_tai_bao_loi_ro_rang(fake_ollama):
    fake_ollama.reply("/api/chat", {"error": "model 'nope' not found"}, status=404)
    with pytest.raises(OllamaError, match="404.*not found"):
        OllamaClient(model="nope", host=fake_ollama.url).generate("hi")


def test_phan_hoi_thieu_content_bao_loi(fake_ollama):
    fake_ollama.reply("/api/chat", {"done": True})
    with pytest.raises(OllamaError, match="message.content"):
        OllamaClient(host=fake_ollama.url).generate("hi")


def test_khong_ket_noi_duoc_bao_loi():
    # Cổng 9 (discard) trên localhost không có ai nghe.
    with pytest.raises(OllamaError, match="Không kết nối được"):
        OllamaClient(host="http://127.0.0.1:9", timeout=2).generate("hi")


def test_build_tu_config(fake_ollama):
    cfg = Config(
        {
            "llm": {
                "provider": "ollama",
                "model": "qwen2.5:7b",
                "host": fake_ollama.url,
                "temperature": 0.0,
                "max_tokens": 99,
                "think": False,
            }
        }
    )
    client = build_llm_client(cfg)
    assert isinstance(client, OllamaClient)
    assert (client.model, client.max_tokens, client.temperature, client.think) == (
        "qwen2.5:7b",
        99,
        0.0,
        False,
    )


def test_provider_la_bao_loi():
    with pytest.raises(ValueError, match="provider"):
        build_llm_client(Config({"llm": {"provider": "openai"}}))
