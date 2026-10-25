"""Test khâu rag_detector (P3) bằng một detector giả: không cần torch hay model LettuceDetect.

Kiểm tra luồng đọc dữ liệu, nhận diện câu từ chối, công thức metric và CLI. Phần gọi model
LettuceDetect thật chưa được kiểm ở đây.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DETECTOR_DIR = Path(__file__).resolve().parents[1] / "experiments" / "p3_retrieval" / "rag_detector"
sys.path.insert(0, str(DETECTOR_DIR))

import detection  # noqa: E402
import metrics  # noqa: E402
import run_detect  # noqa: E402


class FakeDetector:
    """Gắn cờ câu trả lời nào chứa "69"; ghi lại các lần gọi."""

    def __init__(self, spans: list[dict] | None = None) -> None:
        self.calls: list[dict] = []
        self._spans = spans

    def predict(self, **kwargs) -> list[dict]:
        self.calls.append(kwargs)
        if self._spans is not None:
            return [dict(s) for s in self._spans]
        if "69" in kwargs["answer"]:
            return [{"text": "69 million", "confidence": 0.9}]
        return []


# --- is_refusal ------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"answer": "Paris", "refused": True}, True),
        ({"answer": "", "refused": False}, False),  # cờ refused ưu tiên hơn mọi suy đoán
        ({"answer": "x", "decision": "abstain"}, True),
        ({"answer": "x", "decision": "ABSTAIN"}, True),
        ({"answer": "x", "decision": "answer"}, False),
        ({"answer": ""}, True),
        ({"answer": "   "}, True),
        ({}, True),
        ({"answer": "I don't have enough information to say."}, True),
        ({"answer": "Tôi không thể trả lời câu này"}, True),
        ({"answer": "The capital is Paris."}, False),
    ],
)
def test_is_refusal(record, expected):
    assert detection.is_refusal(record) is expected


# --- detect_records --------------------------------------------------------


def test_cau_tu_choi_khong_goi_detector():
    det = FakeDetector()
    out = detection.detect_records(
        det, [{"id": "q", "contexts": ["c"], "answer": "", "decision": "abstain"}]
    )
    assert det.calls == []
    assert out[0]["refused"] is True
    assert out[0]["max_confidence"] == 0.0 and out[0]["spans"] == []


def test_goi_detector_dung_tham_so_va_lay_max_confidence():
    det = FakeDetector(spans=[{"text": "a", "confidence": 0.4}, {"text": "b", "confidence": 0.8}])
    rec = {"id": "q1", "question": "Q?", "contexts": ["c1", "c2"], "answer": "A", "correct": False}

    out = detection.detect_records(det, [rec])[0]

    assert det.calls == [
        {"context": ["c1", "c2"], "question": "Q?", "answer": "A", "output_format": "spans"}
    ]
    assert out["max_confidence"] == 0.8
    assert out["n_contexts"] == 2 and out["correct"] is False and out["refused"] is False


def test_span_thieu_confidence_duoc_coi_la_1():
    # Detector LLM không luôn trả confidence; thiếu thì coi là chắc chắn.
    det = FakeDetector(spans=[{"text": "x"}])
    out = detection.detect_records(det, [{"id": "q", "contexts": ["c"], "answer": "A"}])[0]
    assert out["max_confidence"] == 1.0
    assert out["spans"][0]["confidence"] == 1.0


def test_khong_co_span_thi_max_confidence_bang_0():
    out = detection.detect_records(
        FakeDetector(spans=[]), [{"id": "q", "contexts": ["c"], "answer": "A"}]
    )
    assert out[0]["max_confidence"] == 0.0


def test_resume_dung_lai_ket_qua_da_co():
    det = FakeDetector()
    cached = {
        "id": "q1",
        "refused": False,
        "correct": True,
        "n_contexts": 1,
        "spans": [],
        "max_confidence": 0.0,
    }
    records = [
        {"id": "q1", "contexts": ["c"], "answer": "A"},
        {"id": "q2", "contexts": ["c"], "answer": "B"},
    ]

    out = detection.detect_records(det, records, done={"q1": cached})

    assert out[0] is cached
    assert len(det.calls) == 1 and det.calls[0]["answer"] == "B"


def test_id_thieu_thi_dung_chi_so_dong():
    out = detection.detect_records(FakeDetector(), [{"contexts": ["c"], "answer": "A"}])
    assert out[0]["id"] == "0"


def test_read_write_jsonl_khu_hoi(tmp_path):
    rows = [{"id": "a", "answer": "tiếng Việt"}, {"id": "b", "answer": ""}]
    path = tmp_path / "sub" / "out.jsonl"
    detection.write_jsonl(path, rows)  # tự tạo thư mục cha
    assert detection.read_jsonl(path) == rows
    assert "tiếng Việt" in path.read_text(encoding="utf-8")  # không bị escape thành \uXXXX


# --- metrics ---------------------------------------------------------------


def _results() -> list[dict]:
    """5 câu: 1 từ chối, 2 bị gắn cờ (0.9 và 0.6), 2 không gắn cờ (1 đúng, 1 sai)."""
    return [
        {"refused": True, "correct": None, "max_confidence": 0.0},
        {"refused": False, "correct": False, "max_confidence": 0.9},
        {"refused": False, "correct": False, "max_confidence": 0.6},
        {"refused": False, "correct": True, "max_confidence": 0.1},
        {"refused": False, "correct": False, "max_confidence": 0.2},
    ]


def test_metrics_tai_tau_05():
    m = metrics.metrics_at(_results(), 0.5)
    assert (m["n"], m["rejected"], m["hallucinated"], m["correct"]) == (5, 1, 2, 1)
    assert m["incorrect_unflagged"] == 1
    assert m["accuracy"] == pytest.approx(20.0)  # 1/5
    assert m["hallucination_rate"] == pytest.approx(40.0)  # 2/5
    assert m["rejection_rate"] == pytest.approx(20.0)  # 1/5
    assert m["adjusted_accuracy"] == pytest.approx(100 * 1 / 3)  # 1/(1+2)


def test_tau_cao_hon_thi_it_cau_bi_gan_co_hon():
    low, high = metrics.metrics_at(_results(), 0.5), metrics.metrics_at(_results(), 0.7)
    assert low["hallucinated"] == 2 and high["hallucinated"] == 1
    assert high["hallucinated"] < low["hallucinated"]


def test_tau_bang_confidence_thi_tinh_la_hallucinated():
    assert metrics.metrics_at(_results(), 0.9)["hallucinated"] == 1  # điều kiện là >= tau


def test_thieu_gold_thi_metric_phu_thuoc_gold_la_none():
    results = [
        {"refused": False, "correct": None, "max_confidence": 0.1},
        {"refused": False, "correct": None, "max_confidence": 0.9},
    ]
    m = metrics.metrics_at(results, 0.5)
    assert m["accuracy"] is None and m["adjusted_accuracy"] is None and m["correct"] is None
    assert m["hallucination_rate"] == pytest.approx(50.0)  # không cần gold


def test_chia_cho_khong_tra_none_thay_vi_loi():
    all_refused = [{"refused": True, "correct": None, "max_confidence": 0.0}]
    m = metrics.metrics_at(all_refused, 0.5)
    assert m["adjusted_accuracy"] is None  # correct + hallucinated = 0
    assert m["rejection_rate"] == 100.0
    assert metrics.metrics_at([], 0.5)["hallucination_rate"] is None


def test_summarize_sap_xep_va_khu_trung_tau():
    s = metrics.summarize(_results(), 0.5, [0.9, 0.5, 0.9, 0.7])
    assert [row["tau"] for row in s["tau_sweep"]] == [0.5, 0.7, 0.9]
    assert s["operating_point"]["tau"] == 0.5


# --- CLI (run_detect.main) -------------------------------------------------


@pytest.fixture
def fake_detector(monkeypatch):
    det = FakeDetector()
    monkeypatch.setattr(run_detect, "build_detector", lambda *a, **k: det)
    return det


def test_cli_chay_voi_du_lieu_mau(fake_detector, tmp_path, capsys):
    example = DETECTOR_DIR / "examples" / "generations_example.jsonl"
    run_detect.main(["--scenario", "S1", "--input", str(example), "--output-dir", str(tmp_path)])

    summary = json.loads((tmp_path / "S1_summary.json").read_text(encoding="utf-8"))
    op = summary["operating_point"]
    # q1 bị gắn cờ (69 million), q2 đúng, q3 bị từ chối.
    assert (op["n"], op["hallucinated"], op["correct"], op["rejected"]) == (3, 1, 1, 1)
    assert op["accuracy"] == pytest.approx(100 / 3)
    assert op["adjusted_accuracy"] == pytest.approx(50.0)
    assert summary["scenario"] == "S1" and len(fake_detector.calls) == 2  # q3 không gọi detector

    rows = detection.read_jsonl(tmp_path / "S1_results.jsonl")
    assert [r["id"] for r in rows] == ["q1", "q2", "q3"]
    printed = capsys.readouterr().out
    assert "[S1]" in printed  # dòng tiến trình
    assert (
        json.loads(printed[printed.index("{") :])["scenario"] == "S1"
    )  # rồi tới khối JSON tóm tắt


def test_cli_all_bo_qua_kich_ban_thieu_input_va_ghi_comparison(fake_detector, tmp_path, capsys):
    example = DETECTOR_DIR / "examples" / "generations_example.jsonl"
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for name in ("s1", "s3"):
        (in_dir / f"{name}.jsonl").write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    run_detect.main(["--all", "--input-dir", str(in_dir), "--output-dir", str(tmp_path / "out")])

    out = capsys.readouterr().out
    assert "S2, S4, S5, S6, S7" in out and "skipped" in out
    csv_text = (tmp_path / "out" / "comparison.csv").read_text(encoding="utf-8")
    scenarios = {line.split(",")[0] for line in csv_text.splitlines()[1:]}
    assert scenarios == {"S1", "S3"}
    assert len(csv_text.splitlines()) == 1 + 2 * 3  # header + 2 kịch bản x 3 giá trị tau mặc định


def test_cli_resume_khong_goi_lai_detector(fake_detector, tmp_path):
    example = DETECTOR_DIR / "examples" / "generations_example.jsonl"
    args = ["--scenario", "S1", "--input", str(example), "--output-dir", str(tmp_path)]
    run_detect.main(args)
    calls_after_first = len(fake_detector.calls)

    run_detect.main([*args, "--resume"])

    assert len(fake_detector.calls) == calls_after_first


def test_cli_thieu_input_thi_thoat_voi_thong_bao(fake_detector, tmp_path):
    with pytest.raises(SystemExit, match="No input for S2"):
        run_detect.main(["--scenario", "S2", "--input-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="Nothing to run"):
        run_detect.main(["--all", "--input-dir", str(tmp_path)])
