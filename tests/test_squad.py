"""Kiểm thử cho src/squad — các hàm thuần của luồng SQuAD 2.0 + guardrail."""

from __future__ import annotations

import pytest

from src.squad.build_data import build, validate
from src.squad.generate import IDK, build_messages
from src.squad.guardrail_scores import content_tokens, lexical_score
from src.squad.judge import build_user_message, parse_verdict
from src.squad.label import f1, is_abstain, judge_tag, label, norm, summarize
from src.squad.retrieve import rank_metrics, rrf_fuse

# ---------- build_data ----------


def _article(title: str, n_para: int = 2) -> dict:
    paragraphs = []
    for i in range(n_para):
        ctx = f"{title} paragraph {i} mentions token{i}."
        paragraphs.append(
            {
                "context": ctx,
                "qas": [
                    {
                        "id": f"{title}-{i}-yes",
                        "question": f"What does paragraph {i} mention?",
                        "is_impossible": False,
                        "answers": [{"text": f"token{i}"}, {"text": f"token{i}"}],
                    },
                    {
                        "id": f"{title}-{i}-no",
                        "question": f"Who wrote paragraph {i}?",
                        "is_impossible": True,
                        "answers": [],
                    },
                ],
            }
        )
    return {"title": title, "paragraphs": paragraphs}


def test_build_chia_theo_bai_va_khong_ro_ri():
    articles = [_article(f"Bai_{i}") for i in range(6)]
    kb, questions, chosen = build(articles, n_articles=4, n_dev=1, seed=0)
    validate(kb, questions)
    assert len(chosen) == 4
    dev_titles = {q["title"] for q in questions["dev"]}
    test_titles = {q["title"] for q in questions["test"]}
    assert len(dev_titles) == 1 and len(test_titles) == 3
    assert not dev_titles & test_titles
    assert len(kb) == 8  # 4 bài x 2 đoạn
    assert kb[0]["text"].startswith("Bai ")  # tên bài chèn ở đầu chunk, bỏ dấu gạch dưới


def test_build_gop_dap_an_trung_va_danh_dau_unanswerable():
    kb, questions, _ = build([_article("Mot_bai")], n_articles=1, n_dev=1, seed=0)
    by_id = {q["qid"]: q for q in questions["dev"]}
    assert by_id["Mot_bai-0-yes"]["gold_answers"] == ["token0"]
    assert by_id["Mot_bai-0-no"]["answerable"] is False
    assert by_id["Mot_bai-0-no"]["gold_answers"] == []


def test_build_khu_doan_trung():
    article = _article("Trung", n_para=1)
    article["paragraphs"].append(article["paragraphs"][0])
    kb, questions, _ = build([article], n_articles=1, n_dev=1, seed=0)
    assert len(kb) == 1
    assert {q["gold_para_id"] for q in questions["dev"]} == {kb[0]["para_id"]}


def test_build_cung_seed_cho_cung_ket_qua():
    articles = [_article(f"Bai_{i}") for i in range(6)]
    assert build(articles, 4, 1, seed=7)[2] == build(list(reversed(articles)), 4, 1, seed=7)[2]


# ---------- retrieve ----------


def test_rank_metrics():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["m", "n", "o"]]
    m = rank_metrics(ranked, ["a", "z", "khong_co"])
    assert m["recall@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["recall@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mrr@10"] == pytest.approx((1 + 1 / 3) / 3, abs=1e-4)


def test_rrf_fuse_uu_tien_tai_lieu_dung_cao_o_ca_hai_bang():
    fused = rrf_fuse([{1: 1, 2: 2}, {2: 1, 3: 2}], k=60)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert max(fused, key=fused.get) == 2
    assert fused[1] > fused[3]  # cùng xuất hiện một lần, hạng 1 hơn hạng 2


# ---------- generate ----------


def test_prompt_c0_khong_co_ngu_canh():
    msgs = build_messages("c0", "Ai viet?", [])
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "Question: Ai viet?"}
    assert all("Context:" not in m["content"] for m in msgs[1:])
    assert IDK in [m["content"] for m in msgs if m["role"] == "assistant"]


def test_prompt_c1_danh_so_doan_va_co_vi_du_tu_choi():
    msgs = build_messages("c1", "Sau bao nhieu?", ["doan mot", "doan hai"])
    assert (
        msgs[-1]["content"] == "Context:\n[1] doan mot\n\n[2] doan hai\n\nQuestion: Sau bao nhieu?"
    )
    assert [m["role"] for m in msgs[1:-1]] == ["user", "assistant"] * 4
    assert [m["content"] for m in msgs if m["role"] == "assistant"].count(IDK) == 2


# ---------- guardrail_scores ----------


def test_lexical_score():
    assert lexical_score("Baikal seal", "The lake is home to the Baikal seal.") == 1.0
    assert lexical_score("Baikal whale", "The lake is home to the Baikal seal.") == 0.5
    assert lexical_score("the of", "bat ky") == 0.0  # chỉ có stop word → không có từ nội dung
    assert content_tokens("The Deepest lake") == {"deepest", "lake"}


# ---------- label ----------


def test_norm_va_f1_theo_chuan_squad():
    assert norm("The  Baikal-seal!") == "baikalseal"
    assert f1("the Baikal seal", "Baikal seal") == 1.0
    assert f1("Baikal seal species", "Baikal seal") == pytest.approx(0.8)
    assert f1("Canberra", "Sydney") == 0.0
    assert f1("", "Sydney") == 0.0


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("I don't know", True),
        ("I do not know.", True),
        ("I don't know, but maybe 1643", True),
        ("", True),
        ("  ", True),
        ("1643. I don't know for sure", False),  # chỉ tính khi cụm từ đứng ĐẦU câu
        ("Unknown", False),
    ],
)
def test_is_abstain(answer, expected):
    assert is_abstain(answer) is expected


def test_label_bon_truong_hop():
    yes = {"answerable": True, "gold_answers": ["1,642 metres", "1642 m"]}
    no = {"answerable": False, "gold_answers": []}
    assert label(yes, "I don't know") == ("missing", None)
    assert label(no, "I don't know") == ("missing", None)
    assert label(no, "1643") == ("hallucination", None)  # không cần giám khảo
    assert label(yes, "1,642 metres") == ("accurate", 1.0)
    assert label(yes, "about 23 percent")[0] == "unmatched"  # chờ giám khảo


def test_summarize():
    rows = [
        {"label": "accurate", "answerable": True, "gen_s": 0.2},
        {"label": "missing", "answerable": True, "gen_s": 0.2},
        {"label": "missing", "answerable": False, "gen_s": 0.2},
        {"label": "hallucination", "answerable": False, "gen_s": 0.2},
    ]
    s = summarize(rows)
    assert (s["acc"], s["hal"], s["miss"]) == (0.25, 0.25, 0.5)
    assert s["truthfulness_low"] == s["truthfulness_high"] == 0.0
    assert s["correct_abstention"] == 0.5  # 1 trong 2 câu unanswerable được từ chối
    assert s["over_refusal"] == 0.5  # 1 trong 2 câu answerable bị từ chối nhầm


def test_judge_tag():
    assert judge_tag("llama3.3:70b") == "llama3.3-70b"


# ---------- judge ----------


def test_judge_message_va_parse():
    msg = build_user_message("Sau bao nhieu?", ["1,642 metres", "1642 m"], "1642")
    assert msg == (
        "Question: Sau bao nhieu?\n"
        "Reference answer 1: 1,642 metres\nReference answer 2: 1642 m\nPrediction: 1642"
    )
    assert parse_verdict('{"correct": true}') is True
    assert parse_verdict('{"correct": false}') is False
    assert parse_verdict("khong phai json") is None


# ---------- analyze_scores, tune_eval (cần numpy) ----------


def _toy():
    np = pytest.importorskip("numpy")
    from src.squad.tune_eval import to_arrays

    # 10 câu answerable đúng (điểm cao), 10 câu unanswerable bị bịa (điểm thấp), 2 câu từ chối
    labeled, scores = [], {}
    for i in range(10):
        labeled.append({"qid": f"a{i}", "answerable": True, "label": "accurate"})
        scores[f"a{i}"] = 0.50 + 0.05 * i
    for i in range(10):
        labeled.append({"qid": f"u{i}", "answerable": False, "label": "hallucination"})
        scores[f"u{i}"] = 0.05 * i
    for qid, answerable in (("m0", True), ("m1", False)):
        labeled.append({"qid": qid, "answerable": answerable, "label": "missing"})
        scores[qid] = 0.0
    names = ["ctx_cos", "lexical", "hhem_qa_max", "hhem_ans_concat", "relevance"]
    return np, to_arrays(labeled, {q: dict.fromkeys(names, v) for q, v in scores.items()})


def test_auc():
    pytest.importorskip("numpy")
    from src.squad.analyze_scores import auc

    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auc([0.5], [0.5]) == 0.5


def test_apply_chi_chan_cau_dang_duoc_tra_loi():
    np, d = _toy()
    from src.squad.tune_eval import apply, metrics

    acc, hal, miss = apply(d, {"hhem_qa_max": 0.5, "lexical": None})
    assert hal.sum() == 0 and acc.sum() == 10 and miss.sum() == 12
    assert (apply(d, {})[2] == d["miss"]).all()  # không ngưỡng thì giữ nguyên
    m = metrics(acc, hal, miss, d["answerable"])
    assert m["truthfulness"] == pytest.approx(10 / 22)
    assert m["correct_abstention"] == 1.0
    assert m["over_refusal"] == pytest.approx(1 / 11)
    assert m["selective_risk"] == 0.0


def test_tune_single_ton_trong_tran_tu_choi_nham():
    np, d = _toy()
    from src.squad.tune_eval import apply, tune_single

    # 11 câu answerable, 1 câu đã bị từ chối sẵn → trần 0,20 cho phép chặn thêm đúng 1 câu
    t = tune_single(d, "hhem_qa_max", budget=0.20)
    _, _, miss = apply(d, {"hhem_qa_max": t})
    assert miss[d["answerable"]].sum() == 2
    assert miss[d["answerable"]].mean() <= 0.20
    # trần thấp hơn mức từ chối nhầm sẵn có của `base` → báo False
    assert tune_single(d, "hhem_qa_max", base={"lexical": 0.9}, budget=0.10) is False


def test_tune_full_va_compare():
    np, d = _toy()
    pytest.importorskip("scipy")
    from src.squad.tune_eval import FULL, apply, compare, tune_full

    th = tune_full(d)
    assert set(th) == set(FULL)
    after = apply(d, th)
    assert after[2][d["answerable"]].mean() <= 0.10 + 1e-9
    c = compare((d["acc"], d["hal"], d["miss"]), after)
    assert c["hal_added"] == 0  # guardrail chỉ chặn, không tạo thêm câu bịa
    assert c["hal_removed"] == int(d["hal"].sum() - after[1].sum())
    assert c["d_truthfulness"] >= 0
