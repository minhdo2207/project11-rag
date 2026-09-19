"""P5 — Chỉ số đánh giá.

Cung cấp sẵn accuracy và balanced accuracy (quan trọng khi dữ liệu mất cân bằng).
P5 sẽ mở rộng thêm precision/recall/F1, confusion matrix và biểu đồ.
"""

from __future__ import annotations


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Tỉ lệ dự đoán đúng tổng thể."""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Trung bình recall trên từng lớp — công bằng với dữ liệu lệch."""
    labels = set(y_true)
    if not labels:
        return 0.0
    recalls = []
    for label in labels:
        pos = [i for i, t in enumerate(y_true) if t == label]
        if not pos:
            continue
        hit = sum(1 for i in pos if y_pred[i] == label)
        recalls.append(hit / len(pos))
    return sum(recalls) / len(recalls) if recalls else 0.0
