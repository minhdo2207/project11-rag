"""S5 — Query understanding.

S1–S4 all improve the corpus side. S5 is the first scenario that works on the
*input* side, where a user-facing Q&A system takes most of its damage: the
corpus is clean, the index is right, and retrieval still misses because the
question arrived as "lam sao doi mat khau ko".

Four mechanisms, each independently toggleable:

  1. normalisation      NFC, lowercase, whitespace, stray punctuation
  2. diacritic folding   the Vietnamese-specific one, and the highest-value item
                         here: a large share of real queries arrive unaccented,
                         and "mat khau" shares no token at all with "mật khẩu"
  3. glossary expansion  domain abbreviations and synonyms (tk, ck, hđ, cước...)
  4. rewriting / decomposition
                         an LLM turns a colloquial or multi-part question into
                         explicit sub-queries

The rule for 4, which is what keeps it safe: a rewritten query is ALWAYS fused
with the original, never substituted for it. A rewrite that drifts off-target
then costs a little ranking, instead of losing the answer outright.

Dependency: stdlib only. The LLM rewriter is injected; a rule-based stub is
included so everything runs offline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol, Sequence

__all__ = [
    "QueryConfig", "QueryPlan", "QueryUnderstanding",
    "fold_diacritics", "normalize_text", "classify_query",
    "RewriterProtocol", "RuleRewriter", "LLMRewriter", "DEFAULT_GLOSSARY",
]


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(s: str, *, strip_punct: bool = True) -> str:
    """NFC first: Vietnamese text arrives in both composed and decomposed forms,
    and two visually identical strings will not match across them."""
    s = unicodedata.normalize("NFC", s).lower()
    if strip_punct:
        s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def fold_diacritics(s: str) -> str:
    """'mật khẩu' -> 'mat khau'.

    The single highest-value transform for Vietnamese retrieval. Unaccented
    input is normal, not an edge case — phone keyboards, haste, habit — and
    without folding the lexical branch contributes nothing at all for those
    queries, because the two strings share no token.

    đ/Đ needs handling separately: it is a distinct letter, not a combining mark,
    so NFD leaves it untouched.
    """
    s = s.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


# --------------------------------------------------------------------------- #
# glossary
# --------------------------------------------------------------------------- #

DEFAULT_GLOSSARY: dict[str, list[str]] = {
    # viết tắt người dùng hay dùng -> dạng đầy đủ
    "tk": ["tài khoản"],
    "mk": ["mật khẩu"],
    "ck": ["chuyển khoản"],
    "hd": ["hóa đơn"],
    "hđ": ["hóa đơn"],
    "tt": ["thanh toán"],
    "dv": ["dịch vụ"],
    "kh": ["khách hàng"],
    "sdt": ["số điện thoại"],
    "otp": ["mã xác thực", "mã otp"],
    "app": ["ứng dụng"],
    # cách nói thông tục -> thuật ngữ trong tài liệu
    "cắt mạng": ["tạm ngưng dịch vụ"],
    "khóa máy": ["tạm ngưng dịch vụ"],
    "đóng tiền": ["thanh toán"],
    "nạp tiền": ["thanh toán"],
    "tiền mạng": ["cước"],
    "gói mạng": ["gói cước"],
    "đứt mạng": ["mất kết nối"],
    "bị treo": ["lỗi"],
}


def expand_glossary(query: str, glossary: dict[str, list[str]]) -> list[str]:
    """Return expansion terms to append — the original wording is kept, so this
    only ever adds signal."""
    q = normalize_text(query)
    folded = fold_diacritics(q)
    extra: list[str] = []
    for term, repl in glossary.items():
        t = normalize_text(term)
        if re.search(rf"\b{re.escape(t)}\b", q) or \
           re.search(rf"\b{re.escape(fold_diacritics(t))}\b", folded):
            extra.extend(repl)
    return list(dict.fromkeys(extra))


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #

_CODE = re.compile(r"\b[A-Z]{2,}[_-]?\d{3,}\b|\b[A-Z][a-z]+(?:Error|Exception)\b")
_NUMERIC = re.compile(r"\b\d{3,}\b")
_PROPER = re.compile(r"\b(?:combo|gói)\s*\d+|\bv\d+(?:\.\d+)?\b", re.I)
_MULTI = re.compile(r"\bvà\b|\bhoặc\b|\bcùng\b|[,;]|\bthì sao\b|\bcòn\b")
_WH = re.compile(r"\b(sao|nào|gì|đâu|bao nhiêu|khi nào|thế nào|làm sao|tại sao|mấy)\b")


def classify_query(q: str) -> str:
    """exact | semantic | mixed — selects the fusion weights.

    Rule-based on purpose: it must be auditable, and the signal (does the query
    carry a code, a product name, or a number?) is strong enough that a trained
    classifier would add opacity without adding accuracy.
    """
    qn = normalize_text(q, strip_punct=False)
    words = [w for w in re.split(r"\W+", qn) if w]
    if not words:
        return "mixed"

    hits = len(_CODE.findall(q)) + len(_NUMERIC.findall(q)) + len(_PROPER.findall(q))
    wh = bool(_WH.search(qn))

    if hits and len(words) <= 6:
        return "exact"
    if hits:
        return "mixed"
    return "semantic" if wh else "mixed"


def is_multipart(q: str) -> bool:
    return bool(_MULTI.search(normalize_text(q, strip_punct=False)))


# --------------------------------------------------------------------------- #
# rewriting
# --------------------------------------------------------------------------- #

class RewriterProtocol(Protocol):
    def rewrite(self, query: str) -> list[str]: ...


class RuleRewriter:
    """Offline rewriter: splits coordinated clauses and emits a glossary-expanded
    variant. No model needed, and it covers the two cases that matter most —
    multi-part questions and colloquial vocabulary."""

    def __init__(self, glossary: dict[str, list[str]] | None = None):
        self.glossary = glossary or DEFAULT_GLOSSARY

    def rewrite(self, query: str) -> list[str]:
        out: list[str] = []
        parts = [p.strip() for p in re.split(r"\bvà\b|\bcòn\b|[,;?]", query) if p.strip()]
        if len(parts) > 1:
            out.extend(parts)
        extra = expand_glossary(query, self.glossary)
        if extra:
            out.append(query + " " + " ".join(extra))
        return list(dict.fromkeys(out))


class LLMRewriter:
    """Wraps any callable `llm(prompt) -> str`.

    The prompt asks for sub-queries only, one per line. Keep it narrow: a
    rewriter that is allowed to "improve" the question will quietly change what
    was asked, and that failure is invisible in the retrieval metrics.
    """

    PROMPT = (
        "Viết lại câu hỏi sau thành tối đa 3 truy vấn tìm kiếm ngắn, dùng thuật "
        "ngữ chuẩn trong tài liệu chính sách/cước. Nếu câu hỏi gồm nhiều ý, tách "
        "thành các truy vấn riêng. Chỉ trả về các truy vấn, mỗi dòng một truy vấn, "
        "không giải thích.\n\nCâu hỏi: {q}"
    )

    def __init__(self, llm, max_variants: int = 3):
        self.llm = llm
        self.max_variants = max_variants

    def rewrite(self, query: str) -> list[str]:
        try:
            raw = self.llm(self.PROMPT.format(q=query))
        except Exception:
            return []          # a rewriter failure must never break retrieval
        lines = [re.sub(r"^\s*[-*\d.)]+\s*", "", l).strip()
                 for l in str(raw).splitlines() if l.strip()]
        return [l for l in lines if l][: self.max_variants]


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #

@dataclass
class QueryConfig:
    normalize: bool = True
    fold_diacritics: bool = True
    use_glossary: bool = True
    rewrite: bool = False              # off until S5 proves it helps
    decompose: bool = True             # only fires on multi-part questions
    max_variants: int = 4              # including the original
    glossary: dict = field(default_factory=lambda: dict(DEFAULT_GLOSSARY))

    # fusion weights by query class
    weight_dense: dict = field(default_factory=lambda: {
        "exact": 0.4, "semantic": 1.0, "mixed": 0.7})
    weight_lexical: dict = field(default_factory=lambda: {
        "exact": 1.0, "semantic": 0.4, "mixed": 0.7})


@dataclass
class QueryPlan:
    original: str
    variants: list[str]                # original first, always
    query_type: str
    multipart: bool
    weight_dense: float
    weight_lexical: float
    notes: list[str] = field(default_factory=list)


class QueryUnderstanding:
    def __init__(self, config: QueryConfig | None = None,
                 rewriter: RewriterProtocol | None = None):
        self.config = config or QueryConfig()
        self.rewriter = rewriter or RuleRewriter(self.config.glossary)

    def plan(self, query: str) -> QueryPlan:
        c = self.config
        notes: list[str] = []

        base = normalize_text(query, strip_punct=False) if c.normalize else query
        variants = [base]

        if c.fold_diacritics:
            folded = fold_diacritics(base)
            if folded != base:
                variants.append(folded)
                notes.append("folded")

        if c.use_glossary:
            extra = expand_glossary(base, c.glossary)
            if extra:
                variants.append(base + " " + " ".join(extra))
                notes.append(f"glossary:{len(extra)}")

        multipart = is_multipart(base)
        if (c.rewrite or (c.decompose and multipart)) and self.rewriter is not None:
            for v in self.rewriter.rewrite(base):
                if v not in variants:
                    variants.append(v)
            if multipart:
                notes.append("decomposed")

        # the original is always variant 0 and is never dropped: a rewrite that
        # drifts should cost ranking, not the answer
        variants = list(dict.fromkeys(variants))[: c.max_variants]

        qtype = classify_query(query)
        return QueryPlan(
            original=query, variants=variants, query_type=qtype, multipart=multipart,
            weight_dense=c.weight_dense.get(qtype, 1.0),
            weight_lexical=c.weight_lexical.get(qtype, 1.0),
            notes=notes,
        )
