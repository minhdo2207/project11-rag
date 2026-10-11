"""Test detector với LLM và retriever giả: không cần Ollama, không cần knowledge base thật."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.detector import MAX_CODE_CHARS, MaliciousCodeDetector, parse_verdict  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.models import Document  # noqa: E402
from src.retriever import Retriever  # noqa: E402


class FakeLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        return self.reply


class FakeRetriever(Retriever):
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.calls.append((query, top_k))
        return self.docs[:top_k]


MALICIOUS_JSON = json.dumps(
    {
        "label": "malicious",
        "reason": "Executes a decoded payload.",
        "suspicious_lines": ["exec(base64.b64decode(x))"],
    }
)
KB_DOCS = [
    Document("yara::b64exec", 'rule b64exec { strings: $a = "b64decode" }'),
    Document("ghsa::GHSA-1", "Package steals environment variables on install."),
]


# --- parse_verdict ---------------------------------------------------------


def test_parse_json_day_du():
    assert parse_verdict(MALICIOUS_JSON) == (
        "malicious",
        "Executes a decoded payload.",
        ["exec(base64.b64decode(x))"],
    )


def test_parse_json_trong_rao_markdown_va_loi_dan():
    text = f"Sure! Here is my answer:\n```json\n{MALICIOUS_JSON}\n```\nHope it helps."
    assert parse_verdict(text)[0] == "malicious"


def test_parse_label_viet_hoa_va_co_khoang_trang():
    assert parse_verdict('{"label": "  Benign ", "reason": "ok"}')[0] == "benign"


def test_parse_json_bi_cat_cut_van_vot_duoc_label():
    truncated = '{"label": "malicious", "reason": "Downloads and runs a scr'
    assert parse_verdict(truncated) == ("malicious", "", [])


def test_parse_label_la_thi_la_unknown():
    assert parse_verdict('{"label": "suspicious", "reason": "?"}')[0] == "unknown"


def test_parse_rac_thi_la_unknown_khong_doan():
    # Câu này có chữ "malicious" nhưng không phải JSON: không được suy ra nhãn từ đó.
    assert parse_verdict("This does not look malicious to me.") == ("unknown", "", [])


def test_parse_suspicious_lines_sai_kieu_thi_bo_qua():
    assert parse_verdict('{"label": "benign", "suspicious_lines": "oops"}')[2] == []


# --- detect ----------------------------------------------------------------


def test_detect_tra_ve_verdict_day_du():
    llm = FakeLLM(MALICIOUS_JSON)
    detector = MaliciousCodeDetector(FakeRetriever(KB_DOCS), llm)

    verdict = detector.detect("exec(base64.b64decode(x))")

    assert verdict.label == "malicious"
    assert verdict.reason == "Executes a decoded payload."
    assert verdict.suspicious_lines == ["exec(base64.b64decode(x))"]
    assert verdict.raw == MALICIOUS_JSON
    assert [e.split(": ")[0] for e in verdict.evidence] == ["yara::b64exec", "ghsa::GHSA-1"]


def test_rag_dua_bang_chung_vao_prompt_va_dung_top_k():
    retriever = FakeRetriever(KB_DOCS)
    llm = FakeLLM(MALICIOUS_JSON)
    MaliciousCodeDetector(retriever, llm, top_k=1).detect("some code")

    assert retriever.calls == [("some code", 1)]
    prompt = llm.prompts[0]
    assert "[1] rule b64exec" in prompt
    assert "GHSA-1" not in prompt and "steals environment" not in prompt  # top_k=1
    assert "some code" in prompt
    assert llm.systems[0] is not None


def test_no_rag_khong_goi_retriever_va_khong_co_bang_chung():
    retriever = FakeRetriever(KB_DOCS)
    llm = FakeLLM(MALICIOUS_JSON)
    verdict = MaliciousCodeDetector(retriever, llm, use_rag=False).detect("some code")

    assert retriever.calls == []
    assert verdict.evidence == []
    assert "Reference knowledge" not in llm.prompts[0]


def test_no_rag_chay_duoc_ma_khong_can_retriever():
    verdict = MaliciousCodeDetector(None, FakeLLM(MALICIOUS_JSON), use_rag=False).detect("x")
    assert verdict.label == "malicious"


def test_rag_ma_thieu_retriever_thi_bao_loi_ngay():
    with pytest.raises(ValueError, match="retriever"):
        MaliciousCodeDetector(None, FakeLLM(""), use_rag=True)


def test_rag_nhung_kb_khong_co_gi_thi_khong_chen_muc_bang_chung():
    llm = FakeLLM(MALICIOUS_JSON)
    MaliciousCodeDetector(FakeRetriever([]), llm).detect("some code")
    assert "Reference knowledge" not in llm.prompts[0]


def test_few_shot_them_vi_du_co_nhan_vao_prompt():
    llm = FakeLLM(MALICIOUS_JSON)
    MaliciousCodeDetector(FakeRetriever(KB_DOCS), llm, few_shot=True).detect("some code")
    assert "Worked examples" in llm.prompts[0]
    assert '"label": "benign"' in llm.prompts[0]


def test_code_qua_dai_bi_cat_giua_nhung_giu_dau_va_cuoi():
    llm = FakeLLM(MALICIOUS_JSON)
    code = "HEAD_MARK" + "x" * (MAX_CODE_CHARS * 2) + "TAIL_MARK"
    MaliciousCodeDetector(None, llm, use_rag=False).detect(code)

    prompt = llm.prompts[0]
    assert "HEAD_MARK" in prompt and "TAIL_MARK" in prompt
    assert "characters omitted" in prompt
    assert len(prompt) < MAX_CODE_CHARS * 2


def test_llm_tra_rac_thi_verdict_unknown_nhung_van_giu_raw():
    verdict = MaliciousCodeDetector(None, FakeLLM("¯\\_(ツ)_/¯"), use_rag=False).detect("x")
    assert verdict.label == "unknown"
    assert verdict.raw == "¯\\_(ツ)_/¯"


# --- from_config -----------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expect_calls", "expect_few_shot"),
    [("no_rag", 0, False), ("rag", 1, False), ("rag_fewshot", 1, True)],
)
def test_from_config_chon_dung_che_do(mode, expect_calls, expect_few_shot):
    cfg = Config({"pipeline": {"mode": mode}, "retrieval": {"top_k": 3}})
    retriever = FakeRetriever(KB_DOCS)
    llm = FakeLLM(MALICIOUS_JSON)

    MaliciousCodeDetector.from_config(cfg, retriever, llm).detect("code")

    assert len(retriever.calls) == expect_calls
    if expect_calls:
        assert retriever.calls[0][1] == 3
    assert ("Worked examples" in llm.prompts[0]) is expect_few_shot


def test_from_config_che_do_la_bao_loi():
    with pytest.raises(ValueError, match="pipeline.mode"):
        MaliciousCodeDetector.from_config(
            Config({"pipeline": {"mode": "magic"}}), None, FakeLLM("")
        )
