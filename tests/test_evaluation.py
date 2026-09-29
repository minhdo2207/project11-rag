import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import accuracy, balanced_accuracy  # noqa: E402


def test_accuracy():
    assert accuracy(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert accuracy(["a", "b"], ["a", "a"]) == 0.5
    assert accuracy([], []) == 0.0


def test_balanced_accuracy_handles_imbalance():
    # 4 benign / 1 malicious. Doan sai dung cai malicious duy nhat.
    y_true = ["benign", "benign", "benign", "benign", "malicious"]
    y_pred = ["benign", "benign", "benign", "benign", "benign"]
    # accuracy cao gia tao (0.8) nhung balanced phai thap vi recall lop malicious = 0
    assert accuracy(y_true, y_pred) == 0.8
    assert balanced_accuracy(y_true, y_pred) == 0.5
