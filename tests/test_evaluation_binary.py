"""Test accuracy / balanced_accuracy cho bài toán độc/lành của detector."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import accuracy, balanced_accuracy  # noqa: E402


def test_accuracy_co_ban():
    assert accuracy(["malicious", "benign", "benign"], ["malicious", "benign", "malicious"]) == (
        pytest.approx(2 / 3)
    )


def test_nhan_unknown_tinh_la_sai():
    assert accuracy(["malicious", "benign"], ["unknown", "benign"]) == 0.5


def test_balanced_accuracy_vach_tran_detector_luon_tra_benign():
    y_true = ["benign"] * 9 + ["malicious"]
    y_pred = ["benign"] * 10
    assert accuracy(y_true, y_pred) == 0.9
    assert balanced_accuracy(y_true, y_pred) == 0.5


def test_balanced_accuracy_hoan_hao_va_toan_sai():
    y_true = ["malicious", "malicious", "benign", "benign"]
    assert balanced_accuracy(y_true, y_true) == 1.0
    assert balanced_accuracy(y_true, ["benign", "benign", "malicious", "malicious"]) == 0.0


def test_loi_khi_rong_hoac_lech_do_dai():
    with pytest.raises(ValueError, match="rỗng"):
        accuracy([], [])
    with pytest.raises(ValueError, match="độ dài"):
        balanced_accuracy(["benign"], ["benign", "malicious"])
