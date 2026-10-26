"""P5 — Gộp nhãn từng câu thành chỉ số đánh giá.

Phần cuối file (``accuracy``, ``balanced_accuracy``) đo bài toán độc/lành của detector;
phần đầu (``three_label_report``) đo ba nhãn của CRAG.

Ranh giới với phần còn lại của đường ống:

    src.squad.label  so câu trả lời với đáp án chuẩn  →  gán MỘT nhãn cho mỗi câu
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

# Luồng SQuAD (src.squad.label) ghi nhãn câu bịa là "hallucination"; coi như cùng một nhãn.
ALIASES = {"hallucination": HALLUCINATED}


def three_label_report(labels: list[str], penalty: float = 1.0) -> dict:
    """Ba nhãn gốc của CRAG, đơn vị phần trăm.

    Args:
        labels: nhãn của từng câu, mỗi phần tử thuộc ``LABELS`` (hoặc tên khác trong
            ``ALIASES``).
        penalty: số điểm bị trừ cho một câu bịa. CRAG dùng 1; luật +1/−4/0 của
            why-rags-hallucinate dùng 4. Chỉ ảnh hưởng ``score``, không ảnh hưởng
            ``truthfulness`` — để số cũ vẫn so được.

    Returns:
        dict gồm ``n``, ba tỷ lệ, ``truthfulness`` (= acc − hall), ``score``
        (= acc − penalty·hall), ``penalty`` và ``counts`` đếm thô.
        ``unparsed`` được tính điểm như ``missing`` (đều 0 điểm) nhưng vẫn đếm
        riêng trong ``counts`` để biết bộ chấm hỏng bao nhiêu câu.

    Raises:
        ValueError: danh sách rỗng, hoặc có nhãn ngoài ``LABELS``.
    """
    if not labels:
        raise ValueError("danh sách nhãn rỗng")
    labels = [ALIASES.get(label, label) for label in labels]
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
        "score": 100 * (acc - penalty * hal),
        "penalty": penalty,
        "counts": {label: count[label] for label in LABELS},
    }


def answer_threshold(penalty: float) -> float:
    """Ngưỡng xác suất đúng tối thiểu để trả lời có lợi hơn từ chối.

    Trả lời: kỳ vọng p·1 − (1 − p)·penalty. Từ chối: 0. Hoà khi p = penalty / (1 + penalty).
    CRAG (penalty 1) → 0,5; luật +1/−4/0 → 0,8.
    """
    return penalty / (1 + penalty)


def breakdown(rows: list[dict], key: str, penalty: float = 1.0) -> dict:
    """Tách chỉ số theo một trường của câu hỏi, vd. ``static_or_dynamic`` hay ``domain``."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row["label"])
    return {g: three_label_report(labels, penalty) for g, labels in sorted(groups.items())}


def risk_coverage(
    rows: list[dict],
    score_key: str = "best_rerank",
    thresholds: list[float] | None = None,
    penalties: tuple[float, ...] = (1.0, 4.0),
) -> list[dict]:
    """Mô phỏng một cổng từ chối trên file ĐÃ CHẤM, không cần gọi lại model.

    Câu có ``score_key`` dưới ngưỡng bị coi như từ chối (nhãn ``missing``); câu còn lại
    giữ nguyên nhãn. Chạy trên file B4 thì mỗi ngưỡng chính là một bản B5 — B5 chỉ khác
    B4 ở cổng này (cùng prompt, cùng đoạn, cùng seed).

    Returns:
        mỗi ngưỡng một dict: ``threshold``, ``coverage`` (% câu thực sự trả lời),
        ``risk`` (% bịa trong số câu đã trả lời), ``accuracy``, ``hallucination`` và
        ``score_p<penalty>`` cho từng penalty.
    """
    if not rows:
        raise ValueError("danh sách câu rỗng")
    if thresholds is None:
        values = sorted({float(r[score_key]) for r in rows if r.get(score_key) is not None})
        thresholds = [values[0] - 1.0] + values  # điểm đầu = không chặn câu nào
    points = []
    for tau in thresholds:
        labels = [
            MISSING if r.get(score_key) is not None and float(r[score_key]) < tau else r["label"]
            for r in rows
        ]
        rep = three_label_report(labels)
        answered = rep["counts"][ACCURATE] + rep["counts"][HALLUCINATED]
        point = {
            "threshold": tau,
            "coverage": 100 * answered / rep["n"],
            "risk": 100 * rep["counts"][HALLUCINATED] / answered if answered else 0.0,
            "accuracy": rep["accuracy"],
            "hallucination": rep["hallucination"],
        }
        for p in penalties:
            point[f"score_p{p:g}"] = rep["accuracy"] - p * rep["hallucination"]
        points.append(point)
    return points


def load_scored(path: str | Path) -> list[dict]:
    """Đọc nguyên các dòng của file ``.scored_*.jsonl``, bỏ dòng trống."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def report_from_file(path: str | Path) -> dict:
    """Đọc file nhãn từng câu (vd. ``labeled_*.jsonl`` của ``src.squad.label``), rút ``label``.

    Câu còn nhãn ``unmatched`` (chưa qua giám khảo) sẽ bị báo là nhãn không hợp lệ: phải chạy
    giám khảo rồi gán nhãn lại trước khi tính chỉ số.
    """
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
