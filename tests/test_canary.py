"""Kiểm thử cho src/crag/make_canary.py và nhánh chấm canary của src/crag/score.py."""

from __future__ import annotations

import random

from src.crag.make_canary import fake_name, no_evidence
from src.crag.score import is_refusal


def _item(qid, answer, alt=()):
    return {"interaction_id": qid, "query": "q", "answer": answer, "alt_ans": list(alt)}


def test_no_evidence_bo_doan_chua_dap_an():
    items = [_item("a", "Peter Collinson")]
    retrieval = {"a": [{"text": "directed by Peter Collinson in 1969"}, {"text": "a heist film"}]}
    ((question, retr),) = no_evidence(items, retrieval, max_words=5)
    assert question["canary_type"] == "no_evidence"
    assert question["interaction_id"] == "canary-ne-a"
    assert [c["text"] for c in retr["chunks"]] == ["a heist film"]


def test_no_evidence_xet_ca_dap_an_phu():
    items = [_item("a", "NYC", alt=["New York City"])]
    retrieval = {"a": [{"text": "born in New York City"}, {"text": "an actor"}]}
    ((_, retr),) = no_evidence(items, retrieval, max_words=5)
    assert [c["text"] for c in retr["chunks"]] == ["an actor"]


def test_no_evidence_bo_ca_doan_nhac_tat():
    items = [_item("a", "Nashville Predators")]
    retrieval = {"a": [{"text": "the Predators won 3-1"}, {"text": "a hockey game"}]}
    ((_, retr),) = no_evidence(items, retrieval, max_words=5)
    assert [c["text"] for c in retr["chunks"]] == ["a hockey game"]


def test_no_evidence_bo_qua_cau_khong_loc_duoc():
    items = [
        _item("dai", "a very long answer sentence with many words in it"),
        _item("yesno", "yes"),
        _item("khong_doan_nao_co", "Paris"),
        _item("het_sach", "Paris"),
    ]
    retrieval = {
        "dai": [{"text": "x"}],
        "yesno": [{"text": "yes it is"}, {"text": "x"}],
        "khong_doan_nao_co": [{"text": "London"}],
        "het_sach": [{"text": "Paris"}],
    }
    assert no_evidence(items, retrieval, max_words=5) == []


def test_ten_bia_khong_trung_tu_trong_corpus():
    rng = random.Random(0)
    corpus = {"velor"}
    for _ in range(50):
        assert not any(p.lower() in corpus for p in fake_name(rng, corpus).split())


def test_tu_choi_va_invalid_question_deu_tinh_la_tu_choi():
    assert is_refusal("I don't know")
    assert is_refusal("invalid question")
    assert not is_refusal("1999")
