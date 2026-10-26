"""Kiểm thử cho src/evaluation.py — phần gộp nhãn thành chỉ số của P5."""

from __future__ import annotations

import json

import pytest

from src.evaluation import (
    answer_threshold,
    breakdown,
    report_from_file,
    risk_coverage,
    three_label_report,
)


def test_tinh_dung_ba_ty_le_va_truthfulness():
    labels = ["accurate"] * 5 + ["hallucinated"] * 3 + ["missing"] * 2
    r = three_label_report(labels)
    assert r["n"] == 10
    assert r["accuracy"] == 50.0
    assert r["hallucination"] == 30.0
    assert r["missing"] == 20.0
    assert r["truthfulness"] == 20.0  # 50 − 30


def test_cau_tu_choi_khong_bi_tru_diem():
    """Toàn từ chối thì truthfulness = 0, không âm."""
    r = three_label_report(["missing"] * 4)
    assert r["truthfulness"] == 0.0
    assert r["missing"] == 100.0


def test_toan_bia_thi_truthfulness_am_toi_da():
    r = three_label_report(["hallucinated"] * 4)
    assert r["truthfulness"] == -100.0


def test_unparsed_tinh_diem_nhu_missing_nhung_dem_rieng():
    r = three_label_report(["accurate", "unparsed", "missing", "unparsed"])
    assert r["missing"] == 75.0  # 1 missing + 2 unparsed
    assert r["truthfulness"] == 25.0  # unparsed không bị trừ
    assert r["counts"]["unparsed"] == 2
    assert r["counts"]["missing"] == 1


def test_danh_sach_rong_thi_bao_loi():
    with pytest.raises(ValueError, match="rỗng"):
        three_label_report([])


def test_nhan_la_thi_bao_loi_thay_vi_nuot_im_lang():
    with pytest.raises(ValueError, match="không hợp lệ"):
        three_label_report(["accurate", "malicious"])


def test_doc_duoc_file_da_cham(tmp_path):
    path = tmp_path / "gen_B0.scored_crag.jsonl"
    rows = [
        {"interaction_id": "a", "label": "accurate"},
        {"interaction_id": "b", "label": "hallucinated"},
        {"interaction_id": "c", "label": "missing"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    r = report_from_file(path)
    assert r["n"] == 3
    assert r["truthfulness"] == pytest.approx(0.0)


def test_doc_duoc_nhan_cua_luong_squad(tmp_path):
    """src.squad.label ghi "hallucination"; phải tính như "hallucinated"."""
    path = tmp_path / "labeled_llama3_c1_dev.jsonl"
    rows = [
        {"qid": "a", "label": "accurate"},
        {"qid": "b", "label": "accurate"},
        {"qid": "c", "label": "hallucination"},
        {"qid": "d", "label": "missing"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    r = report_from_file(path)
    assert r["hallucination"] == 25.0
    assert r["truthfulness"] == pytest.approx(25.0)  # 50 − 25
    assert r["counts"]["hallucinated"] == 1


def test_cau_chua_qua_giam_khao_thi_bao_loi():
    with pytest.raises(ValueError, match="không hợp lệ"):
        three_label_report(["accurate", "unmatched"])


def test_dong_thieu_truong_label_thi_bao_loi(tmp_path):
    path = tmp_path / "hong.jsonl"
    path.write_text(json.dumps({"interaction_id": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu trường"):
        report_from_file(path)


def test_penalty_1_cho_ra_dung_truthfulness_cu():
    labels = ["accurate"] * 5 + ["hallucinated"] * 3 + ["missing"] * 2
    r = three_label_report(labels)
    assert r["score"] == r["truthfulness"] == 20.0


def test_penalty_4_phat_bia_nang_gap_bon():
    labels = ["accurate"] * 5 + ["hallucinated"] * 3 + ["missing"] * 2
    r = three_label_report(labels, penalty=4)
    assert r["score"] == pytest.approx(50 - 4 * 30)
    assert r["truthfulness"] == 20.0  # không đổi theo penalty


def test_nguong_tra_loi_theo_penalty():
    assert answer_threshold(1) == 0.5
    assert answer_threshold(4) == pytest.approx(0.8)


def test_tach_nhom_theo_truong():
    rows = [
        {"label": "accurate", "static_or_dynamic": "static"},
        {"label": "hallucinated", "static_or_dynamic": "static"},
        {"label": "hallucinated", "static_or_dynamic": "real-time"},
    ]
    r = breakdown(rows, "static_or_dynamic")
    assert r["static"]["n"] == 2 and r["static"]["truthfulness"] == 0.0
    assert r["real-time"]["truthfulness"] == -100.0


def test_risk_coverage_chan_cau_duoi_nguong():
    rows = [
        {"label": "accurate", "best_rerank": 0.9},
        {"label": "hallucinated", "best_rerank": 0.1},
        {"label": "hallucinated", "best_rerank": 0.8},
        {"label": "missing", "best_rerank": 0.95},  # model tự từ chối
    ]
    pts = risk_coverage(rows, thresholds=[0.0, 0.5])
    no_gate, gate = pts
    assert no_gate["coverage"] == 75.0  # 3/4 câu có trả lời
    assert no_gate["risk"] == pytest.approx(200 / 3)
    assert gate["coverage"] == 50.0  # câu 0,1 bị chặn
    assert gate["risk"] == 50.0
    assert gate["score_p1"] == 0.0
    assert gate["score_p4"] == pytest.approx(25 - 4 * 25)


def test_risk_coverage_diem_dau_la_khong_chan():
    rows = [
        {"label": "hallucinated", "best_rerank": -0.5},
        {"label": "accurate", "best_rerank": 0.3},
    ]
    pts = risk_coverage(rows)
    assert pts[0]["coverage"] == 100.0
    assert pts[-1]["coverage"] == 50.0  # ngưỡng cuối = điểm cao nhất, chỉ câu đó qua
