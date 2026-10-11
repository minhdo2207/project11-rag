"""P5 — Gộp nhãn từng câu thành chỉ số đánh giá.

Phần cuối file (``accuracy``, ``balanced_accuracy``) đo bài toán độc/lành của detector;
phần đầu (``three_label_report``) đo ba nhãn của CRAG.

Ranh giới với phần còn lại của đường ống:

    src.crag.score   so câu trả lời với đáp án chuẩn  →  gán MỘT nhãn cho mỗi câu
                              │
                              ▼  list[str]
    src.evaluation   gộp nhãn thành chỉ số  →  accurate / hallucinated / missing / truthfulness

Việc so với đáp án chuẩn đã xong ở bước chấm, nên module này **không cần đáp án chuẩn**
và không phụ thuộc vào bất kỳ module nào khác — chỉ nhận một danh sách nhãn.

Ba nhãn và cách tính điểm lấy nguyên của CRAG (arXiv 2406.04744, §4.1): câu đúng +1,
câu bịa −1, câu từ chối 0. Phạt câu bịa nặng hơn câu từ chối là có chủ ý — thà nói
"tôi không biết" còn hơn nói sai.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ACCURATE = "accurate"
HALLUCINATED = "hallucinated"
MISSING = "missing"
UNPARSED = "unparsed"  # giám khảo trả về thứ không đọc được — tính điểm như missing

LABELS = (ACCURATE, HALLUCINATED, MISSING, UNPARSED)


def three_label_report(labels: list[str]) -> dict:
    """Ba nhãn gốc của CRAG, đơn vị phần trăm.

    Args:
        labels: nhãn của từng câu, mỗi phần tử thuộc ``LABELS``.

    Returns:
        dict gồm ``n``, ba tỷ lệ, ``truthfulness``, và ``counts`` đếm thô.
        ``unparsed`` được tính điểm như ``missing`` (đều 0 điểm) nhưng vẫn đếm
        riêng trong ``counts`` để biết bộ chấm hỏng bao nhiêu câu.

    Raises:
        ValueError: danh sách rỗng, hoặc có nhãn ngoài ``LABELS``.
    """
    if not labels:
        raise ValueError("danh sách nhãn rỗng")
    unknown = set(labels) - set(LABELS)
    if unknown:
        raise ValueError(f"nhãn không hợp lệ: {sorted(unknown)}; hợp lệ: {list(LABELS)}")

    n = len(labels)
    count = Counter(labels)
    acc = count[ACCURATE] / n
    hal = count[HALLUCINATED] / n
    mis = (count[MISSING] + count[UNPARSED]) / n
    return {
        "n": n,
        "accuracy": 100 * acc,
        "hallucination": 100 * hal,
        "missing": 100 * mis,
        "truthfulness": 100 * (acc - hal),
        "counts": {label: count[label] for label in LABELS},
    }


def report_from_file(path: str | Path) -> dict:
    """Đọc file ``.scored_*.jsonl`` của ``src.crag.score`` rồi rút trường ``label``."""
    path = Path(path)
    labels = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "label" not in row:
                raise ValueError(f"{path}: dòng {i} thiếu trường 'label'")
            labels.append(row["label"])
    return three_label_report(labels)


def _check_pairs(y_true: list[str], y_pred: list[str]) -> None:
    if not y_true:
        raise ValueError("danh sách nhãn rỗng")
    if len(y_true) != len(y_pred):
        raise ValueError(f"độ dài khác nhau: y_true={len(y_true)}, y_pred={len(y_pred)}")


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Tỷ lệ dự đoán đúng (0..1) cho bài toán độc/lành. Nhãn ``unknown`` tính là sai."""
    _check_pairs(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Trung bình recall của từng lớp có trong ``y_true`` (0..1).

    Quan trọng khi dữ liệu lệch (gói lành nhiều hơn gói độc): một detector luôn đáp
    "benign" có accuracy cao nhưng balanced accuracy chỉ 0.5.
    """
    _check_pairs(y_true, y_pred)
    recalls = []
    for cls in sorted(set(y_true)):
        hits = [p == cls for t, p in zip(y_true, y_pred, strict=True) if t == cls]
        recalls.append(sum(hits) / len(hits))
    return sum(recalls) / len(recalls)
