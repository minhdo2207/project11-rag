"""P4 — Business logic phán quyết một gói/đoạn code là độc hay lành.

Detector chỉ phụ thuộc vào hai *interface* (``Retriever``, ``LLMClient``) — không hề
biết knowledge base nằm ở RAM, file hay Qdrant. Nhờ đó đổi backend persistence hay đổi
model sinh không cần sửa lớp này.

Ba chế độ (``pipeline.mode`` trong config), dùng để so sánh trong ablation:

  no_rag         chỉ đưa code cho LLM (baseline zero-shot)
  rag            thêm bằng chứng truy xuất từ knowledge base vào prompt
  rag_fewshot    như ``rag`` và thêm vài ví dụ có nhãn cố định
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import Config
from .llm_client import LLMClient
from .models import Document
from .retriever import Retriever

LABEL_MALICIOUS = "malicious"
LABEL_BENIGN = "benign"
LABEL_UNKNOWN = "unknown"  # LLM trả về thứ không đọc được; không đoán bừa

MAX_CODE_CHARS = 8000  # quá dài thì giữ đầu + cuối, tránh tràn context
MAX_DOC_CHARS = 800  # mỗi mẩu bằng chứng

SYSTEM_PROMPT = (
    "You are a security analyst who reviews Python package source code (setup.py, "
    "install hooks, modules) for malicious behaviour. Answer with a JSON object only."
)

TASK_PROMPT = """\
Decide whether the code below is "malicious" or "benign".

Malicious signals: encoded or obfuscated payloads that get executed (base64/zlib + exec/eval), \
downloading and running remote code, reading credentials or environment secrets and sending \
them out, reverse shells, persistence (cron, startup entries), and anything that runs \
automatically at install time (custom install commands in setup.py).
Ordinary packaging code (metadata, dependencies, console_scripts) is benign.
Base your answer only on what is in the code; do not claim behaviour that is not there.

Respond with JSON only:
{"label": "malicious" or "benign", "reason": "<one or two sentences>", \
"suspicious_lines": ["<exact line copied from the code>"]}
Use an empty list for suspicious_lines when the code is benign."""

# Hai ví dụ nhỏ cố định cho chế độ rag_fewshot (một độc, một lành).
FEW_SHOT_EXAMPLES = [
    (
        "import base64\nexec(base64.b64decode('aW1wb3J0IG9z...'))\n",
        LABEL_MALICIOUS,
        "Decodes a base64 blob at import time and executes it.",
        ["exec(base64.b64decode('aW1wb3J0IG9z...'))"],
    ),
    (
        "from setuptools import setup\nsetup(name='demo', version='1.0', install_requires=['requests'])\n",
        LABEL_BENIGN,
        "Plain packaging metadata and a dependency list; nothing executes at install time.",
        [],
    ),
]


@dataclass
class Verdict:
    """Kết quả phán quyết."""

    label: str  # "malicious" | "benign" | "unknown"
    reason: str
    evidence: list[str]  # các mẩu tri thức đã đưa vào prompt (rỗng khi no_rag)
    suspicious_lines: list[str] = field(default_factory=list)  # dòng code LLM chỉ ra
    raw: str = ""  # câu trả lời gốc của LLM, để debug / chấm lại


def parse_verdict(text: str) -> tuple[str, str, list[str]]:
    """Đọc câu trả lời của LLM thành ``(label, reason, suspicious_lines)``.

    Ưu tiên JSON đầy đủ. Nếu JSON hỏng (thường do bị cắt cụt) thì vẫn cố vớt trường
    ``label`` bằng regex. Không đọc được gì thì trả ``unknown`` thay vì đoán.
    """
    data = _load_json_object(text)
    if data is not None:
        label = str(data.get("label", "")).strip().lower()
        reason = str(data.get("reason", "")).strip()
        lines = data.get("suspicious_lines", [])
        lines = [str(x) for x in lines] if isinstance(lines, list) else []
        return (label if label in (LABEL_MALICIOUS, LABEL_BENIGN) else LABEL_UNKNOWN), reason, lines

    match = re.search(r'"label"\s*:\s*"(malicious|benign)"', text, re.IGNORECASE)
    if match:
        return match.group(1).lower(), "", []
    return LABEL_UNKNOWN, "", []


def _load_json_object(text: str) -> dict | None:
    """Lấy object JSON đầu tiên trong ``text`` (chịu được rào ```json và lời dẫn thừa)."""
    candidates = [text.strip()]
    span = re.search(r"\{.*\}", text, re.DOTALL)
    if span:
        candidates.append(span.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _shorten(text: str, limit: int) -> str:
    """Cắt ``text`` nếu quá dài: giữ phần đầu và phần cuối, báo rõ chỗ bị bỏ."""
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    tail = limit - head
    return f"{text[:head]}\n# ... {len(text) - limit} characters omitted ...\n{text[-tail:]}"


class MaliciousCodeDetector:
    """Phán quyết bằng RAG: truy xuất bằng chứng rồi hỏi LLM.

    ``use_rag=False`` cho phép so sánh baseline no-RAG; khi đó ``retriever`` có thể là
    ``None``.
    """

    def __init__(
        self,
        retriever: Retriever | None,
        llm: LLMClient,
        use_rag: bool = True,
        top_k: int = 5,
        few_shot: bool = False,
    ):
        if use_rag and retriever is None:
            raise ValueError("use_rag=True cần một retriever")
        self._retriever = retriever
        self._llm = llm
        self._use_rag = use_rag
        self._top_k = top_k
        self._few_shot = few_shot

    @classmethod
    def from_config(
        cls, cfg: Config, retriever: Retriever | None, llm: LLMClient
    ) -> MaliciousCodeDetector:
        """Chọn chế độ theo ``pipeline.mode`` và ``retrieval.top_k`` trong config."""
        mode = cfg.mode
        top_k = cfg.get("retrieval.top_k", 5)
        if mode == "no_rag":
            return cls(retriever, llm, use_rag=False, top_k=top_k)
        if mode == "rag":
            return cls(retriever, llm, use_rag=True, top_k=top_k)
        if mode == "rag_fewshot":
            return cls(retriever, llm, use_rag=True, top_k=top_k, few_shot=True)
        raise ValueError(f"pipeline.mode không hợp lệ: {mode!r} (no_rag | rag | rag_fewshot)")

    def _retrieve(self, code: str) -> list[Document]:
        if not self._use_rag or self._retriever is None:
            return []
        return self._retriever.retrieve(code, top_k=self._top_k)

    def _build_prompt(self, code: str, docs: list[Document]) -> str:
        parts = [TASK_PROMPT]

        if self._few_shot:
            examples = []
            for i, (example_code, label, reason, lines) in enumerate(FEW_SHOT_EXAMPLES, 1):
                answer = json.dumps(
                    {"label": label, "reason": reason, "suspicious_lines": lines},
                    ensure_ascii=False,
                )
                examples.append(f"Example {i}\nCode:\n{example_code}\nAnswer: {answer}")
            parts.append("Worked examples:\n\n" + "\n\n".join(examples))

        if docs:
            references = "\n".join(
                f"[{i}] {_shorten(' '.join(d.text.split()), MAX_DOC_CHARS)}"
                for i, d in enumerate(docs, 1)
            )
            parts.append(
                "Reference knowledge retrieved from the knowledge base. It may be irrelevant: "
                "use it only where it matches the code, and ignore it otherwise.\n" + references
            )

        parts.append("Code to review:\n" + _shorten(code, MAX_CODE_CHARS))
        return "\n\n".join(parts)

    def detect(self, code: str) -> Verdict:
        """Phán quyết ``code`` là độc hay lành, kèm lý do và bằng chứng đã dùng."""
        docs = self._retrieve(code)
        raw = self._llm.generate(self._build_prompt(code, docs), system=SYSTEM_PROMPT)
        label, reason, lines = parse_verdict(raw)
        return Verdict(
            label=label,
            reason=reason,
            evidence=[f"{d.doc_id}: {d.preview(80)}" for d in docs],
            suspicious_lines=lines,
            raw=raw,
        )
