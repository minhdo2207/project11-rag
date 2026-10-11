"""Smoke test cho các gói P3 (S1–S7): mỗi runner --demo phải chạy hết và trả đúng mã thoát.

Chỉ kiểm tra "chạy được", không kiểm tra chất lượng. Runner S1 trả mã 1 là chủ đích:
trên bộ demo cả ba chế độ đều đạt recall@5 = 1.0 nên bộ eval không phân biệt được chúng.

Embedder demo của cả 7 runner dùng hash() có sẵn của Python, mà hash() đổi seed mỗi tiến
trình, nên kết quả (và mã thoát của S5) dao động giữa các lần chạy. Test ghim
PYTHONHASHSEED=0 để kết quả ổn định.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

P3_DIR = Path(__file__).resolve().parents[1] / "experiments" / "p3_retrieval"

# s1 dừng với mã 1 theo thiết kế (xem docstring); các gói còn lại phải thoát 0.
EXPECTED_EXIT = {"s1": 1, "s2": 0, "s3": 0, "s4": 0, "s5": 0, "s6": 0, "s7": 0}


@pytest.mark.parametrize("name", sorted(EXPECTED_EXIT))
def test_demo_runner(name: str) -> None:
    cmd = [sys.executable, f"run_{name}.py", "--demo"]
    if name == "s4":
        cmd += ["--as-of", "2026-10-07"]  # runner dùng ngày hôm nay nếu không truyền
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    result = subprocess.run(
        cmd, cwd=P3_DIR / name, capture_output=True, text=True, timeout=300, check=False, env=env
    )
    assert result.returncode == EXPECTED_EXIT[name], result.stdout[-500:] + result.stderr[-500:]
