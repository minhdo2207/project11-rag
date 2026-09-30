"""Kiểm thử cho src/evaluation.py — phần gộp nhãn thành chỉ số của P5."""

from __future__ import annotations

import json

import pytest

from src.evaluation import report_from_file, three_label_report


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


def test_dong_thieu_truong_label_thi_bao_loi(tmp_path):
    path = tmp_path / "hong.jsonl"
    path.write_text(json.dumps({"interaction_id": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu trường"):
        report_from_file(path)
