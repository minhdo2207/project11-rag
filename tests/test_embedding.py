"""Test embedder: HashingEmbedder (offline) và OllamaEmbedder (server giả, xem conftest.py)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src._ollama import OllamaError  # noqa: E402
from src.config import Config  # noqa: E402
from src.embedding import HashingEmbedder, OllamaEmbedder, build_embedder  # noqa: E402
from src.repository._similarity import cosine_similarity  # noqa: E402


def test_hashing_dung_so_chieu_va_chuan_hoa():
    vec = HashingEmbedder(dim=64).embed("import os; os.system('ls')")
    assert len(vec) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0)


def test_hashing_text_rong_ra_vector_khong_thay_vi_chia_cho_khong():
    assert HashingEmbedder(dim=8).embed("") == [0.0] * 8


def test_hashing_text_giong_nhau_gan_hon_text_khac_nhau():
    e = HashingEmbedder(dim=256)
    a = e.embed("exec(base64.b64decode(payload))")
    close = e.embed("exec(base64.b64decode(other_payload))")
    far = e.embed("def add(x, y): return x + y")
    assert cosine_similarity(a, close) > cosine_similarity(a, far)


def test_hashing_on_dinh_giua_cac_tien_trinh():
    # Regression: dùng hash() có sẵn của Python sẽ cho vector khác nhau theo PYTHONHASHSEED.
    code = "from src.embedding import HashingEmbedder; print(HashingEmbedder(32).embed('exec(x)'))"
    outputs = {
        subprocess.run(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); " + code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(outputs) == 1


def test_embed_batch_mac_dinh_goi_tung_cai():
    e = HashingEmbedder(dim=16)
    assert e.embed_batch(["a b", "c d"]) == [e.embed("a b"), e.embed("c d")]


def test_ollama_embedder_gui_ca_lo_trong_mot_request(fake_ollama):
    fake_ollama.reply("/api/embed", {"embeddings": [[1.0, 0.0], [0.0, 1.0]]})
    embedder = OllamaEmbedder(model="bge-m3", host=fake_ollama.url)

    assert embedder.embed_batch(["x", "y"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert len(fake_ollama.requests) == 1
    assert fake_ollama.requests[0] == ("/api/embed", {"model": "bge-m3", "input": ["x", "y"]})


def test_ollama_embedder_embed_don_le(fake_ollama):
    fake_ollama.reply("/api/embed", {"embeddings": [[0.5, 0.5]]})
    assert OllamaEmbedder(host=fake_ollama.url).embed("x") == [0.5, 0.5]


def test_ollama_embedder_so_vector_khong_khop_thi_bao_loi(fake_ollama):
    fake_ollama.reply("/api/embed", {"embeddings": [[1.0]]})
    with pytest.raises(OllamaError, match="không khớp"):
        OllamaEmbedder(host=fake_ollama.url).embed_batch(["x", "y"])


def test_ollama_embedder_lo_rong_khong_goi_server(fake_ollama):
    assert OllamaEmbedder(host=fake_ollama.url).embed_batch([]) == []
    assert fake_ollama.requests == []


def test_build_embedder_theo_config():
    assert isinstance(
        build_embedder(Config({"embedding": {"provider": "hashing", "dim": 32}})), HashingEmbedder
    )
    ollama = build_embedder(
        Config({"embedding": {"provider": "ollama", "model": "nomic-embed-text"}})
    )
    assert isinstance(ollama, OllamaEmbedder) and ollama.model == "nomic-embed-text"


def test_build_embedder_provider_la_bao_loi():
    with pytest.raises(ValueError, match="provider"):
        build_embedder(Config({"embedding": {"provider": "openai"}}))
