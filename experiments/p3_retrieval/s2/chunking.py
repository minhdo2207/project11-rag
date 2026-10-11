"""S2 — Context-preserving chunking.

S2 targets one failure mode: **context fragmentation**. A chunk cut through the
middle of a definition, a table, or a stack trace produces an answer that reads
fluently and is wrong — the model joins half of one fact to half of another.
Retrieval quality cannot fix this, because no amount of ranking recovers
information that no single chunk contains.

Three mechanisms, each independently toggleable so the ablation can separate
them:

  1. structure-aware boundaries   split on headings / paragraphs / AST nodes
                                  instead of character offsets
  2. atomic block protection      never split a code fence, markdown table,
                                  numbered procedure, or stack trace
  3. enrichment header            prepend source + heading path to `embed_text`
                                  so a chunk carries where it came from

Every chunker emits provenance (`source`, `char_start`, `char_end`), which is
what makes span-based evaluation possible — see run_s2.py for why chunk-id
labels do not survive a chunking change.

Dependency: stdlib only.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "ChunkConfig", "Chunk",
    "FixedSizeChunker", "StructureChunker", "PythonASTChunker",
    "chunk_documents", "CHUNKERS",
]


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    id: str
    text: str                       # shown to the user / LLM
    embed_text: str                 # what gets embedded + fed to BM25
    metadata: dict = field(default_factory=dict)   # source, char_start, char_end, heading

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text,
                "embed_text": self.embed_text, "metadata": self.metadata}


@dataclass
class ChunkConfig:
    target_chars: int = 700         # soft target; atomic blocks may exceed it
    max_chars: int = 1400           # hard ceiling before a forced split
    min_chars: int = 60             # below this, merge into the neighbour
    overlap_chars: int = 0          # fixed-size only; structure chunkers use boundaries

    protect_atomic: bool = True     # never split tables / code fences / tracebacks
    enrich_header: bool = True      # prepend source + heading path to embed_text
    heading_in_text: bool = False   # also show the heading to the user


def _cid(*parts) -> str:
    return hashlib.sha1("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _mk(cfg: ChunkConfig, source: str, body: str, start: int, end: int,
        heading: str = "", extra: dict | None = None) -> Chunk:
    """Single construction point so every chunker produces identical provenance
    and applies the enrichment header the same way."""
    meta = {"source": source, "char_start": start, "char_end": end}
    if heading:
        meta["heading"] = heading
    if extra:
        meta.update(extra)

    text = f"{heading}\n{body}" if (heading and cfg.heading_in_text) else body

    if cfg.enrich_header:
        bits = [f"Nguồn: {source}"]
        if heading:
            bits.append(f"Mục: {heading}")
        embed_text = "\n".join(bits) + "\n---\n" + body
    else:
        embed_text = body

    return Chunk(id=_cid(source, start, end), text=text,
                 embed_text=embed_text, metadata=meta)


# --------------------------------------------------------------------------- #
# atomic blocks — the units that must never be split
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")
_TRACE_CONT = re.compile(r"^(?:\s+at\s|at\s|Caused by:|\s+File \"|\.\.\.\s+\d+\s+more)")
_TRACE_START = re.compile(r"Traceback \(most recent call last\)|^\w+(?:Error|Exception):")


def atomic_spans(lines: Sequence[str]) -> list[tuple[int, int]]:
    """Return [start, end) line ranges that must stay in one chunk.

    Covers fenced code, contiguous markdown tables, contiguous list blocks
    (a numbered procedure split in half is worse than a long chunk), and
    stack traces including every `Caused by:` line — the root cause lives in
    the LAST one, so truncating the tail is the worst possible cut.
    """
    spans: list[tuple[int, int]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        if _FENCE.match(line):                      # fenced block
            j = i + 1
            while j < n and not _FENCE.match(lines[j]):
                j += 1
            spans.append((i, min(j + 1, n)))
            i = j + 1
            continue

        if _TABLE_ROW.match(line):                  # contiguous table
            j = i
            while j < n and (_TABLE_ROW.match(lines[j]) or lines[j].strip().startswith("|")):
                j += 1
            if j - i >= 2:
                spans.append((i, j))
            i = j
            continue

        if _LIST_ITEM.match(line):                  # contiguous list / procedure
            j = i
            while j < n and (_LIST_ITEM.match(lines[j]) or
                             (lines[j].startswith((" ", "\t")) and lines[j].strip())):
                j += 1
            if j - i >= 2:
                spans.append((i, j))
            i = j
            continue

        if _TRACE_START.search(line):               # stack trace
            j = i + 1
            while j < n and (_TRACE_CONT.match(lines[j]) or
                             lines[j].lstrip().startswith("Caused by:")):
                j += 1
            spans.append((i, j))
            i = j
            continue

        i += 1
    return spans


def _line_offsets(text: str) -> list[int]:
    offs, pos = [], 0
    for line in text.split("\n"):
        offs.append(pos)
        pos += len(line) + 1
    offs.append(pos)
    return offs


# --------------------------------------------------------------------------- #
# 1. baseline — fixed size
# --------------------------------------------------------------------------- #

class FixedSizeChunker:
    """The baseline to beat. Splits on character offsets, which is exactly the
    behaviour S2 exists to replace: it cuts through sentences, tables and
    tracebacks with no regard for structure."""

    name = "fixed"

    def __init__(self, cfg: ChunkConfig):
        self.cfg = cfg

    def chunk(self, source: str, text: str) -> list[Chunk]:
        c, out, pos = self.cfg, [], 0
        step = max(c.target_chars - c.overlap_chars, 1)
        while pos < len(text):
            end = min(pos + c.target_chars, len(text))
            body = text[pos:end].strip()
            if len(body) >= c.min_chars:
                out.append(_mk(c, source, body, pos, end))
            pos += step
        return out


# --------------------------------------------------------------------------- #
# 2. structure-aware — headings, paragraphs, atomic blocks
# --------------------------------------------------------------------------- #

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class StructureChunker:
    """Splits on document structure, then packs sections up to `target_chars`.

    Boundary priority: heading > blank line > atomic-block edge. A section
    longer than `max_chars` is packed greedily, but a packing step never lands
    inside an atomic span.
    """

    name = "structure"

    def __init__(self, cfg: ChunkConfig):
        self.cfg = cfg

    def chunk(self, source: str, text: str) -> list[Chunk]:
        c = self.cfg
        lines = text.split("\n")
        offs = _line_offsets(text)
        atomics = atomic_spans(lines) if c.protect_atomic else []

        def in_atomic(i: int) -> tuple[int, int] | None:
            for s, e in atomics:
                if s <= i < e:
                    return (s, e)
            return None

        out: list[Chunk] = []
        heading_stack: list[tuple[int, str]] = []
        buf_start = 0
        buf: list[str] = []

        def heading_path() -> str:
            return " > ".join(h for _, h in heading_stack)

        def flush(end_line: int) -> None:
            nonlocal buf, buf_start
            body = "\n".join(buf).strip()
            if body and len(body) >= c.min_chars:
                out.append(_mk(c, source, body, offs[buf_start], offs[end_line],
                               heading=heading_path()))
            elif body and out:
                # too small to stand alone: merge into the previous chunk rather
                # than emitting a fragment nothing can retrieve
                prev = out[-1]
                merged = prev.text + "\n" + body
                out[-1] = _mk(c, source, merged, prev.metadata["char_start"],
                              offs[end_line], heading=prev.metadata.get("heading", ""))
            buf = []
            buf_start = end_line

        i, n = 0, len(lines)
        while i < n:
            span = in_atomic(i)
            if span:
                s, e = span
                block = lines[s:e]
                block_len = sum(len(x) + 1 for x in block)
                # flush first if adding the whole block would blow the ceiling
                if buf and sum(len(x) + 1 for x in buf) + block_len > c.max_chars:
                    flush(s)
                buf.extend(block)
                i = e
                continue

            m = _HEADING.match(lines[i])
            if m:
                flush(i)
                level = len(m.group(1))
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, m.group(2)))
                buf_start = i + 1
                i += 1
                continue

            buf.append(lines[i])
            cur = sum(len(x) + 1 for x in buf)
            # pack up to target, but only break at a blank line
            if cur >= c.target_chars and (i + 1 >= n or not lines[i + 1].strip()):
                flush(i + 1)
            elif cur >= c.max_chars:
                flush(i + 1)
            i += 1

        flush(n)
        return out


# --------------------------------------------------------------------------- #
# 3. code — AST boundaries
# --------------------------------------------------------------------------- #

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class PythonASTChunker:
    """For Python sources: one chunk per top-level function/class, so a chunk is
    always a self-contained syntactic unit. Falls back to StructureChunker when
    the file does not parse — a silent drop is how a corpus develops coverage
    holes nobody notices."""

    name = "ast"

    def __init__(self, cfg: ChunkConfig):
        self.cfg = cfg
        self._fallback = StructureChunker(cfg)

    def chunk(self, source: str, text: str) -> list[Chunk]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return self._fallback.chunk(source, text)

        c = self.cfg
        offs = _line_offsets(text)
        lines = text.split("\n")
        out: list[Chunk] = []
        covered: set[int] = set()

        for node in tree.body:
            if not isinstance(node, _DEFS):
                continue
            s, e = node.lineno - 1, node.end_lineno
            covered |= set(range(s, e))
            body = "\n".join(lines[s:e])
            doc = ast.get_docstring(node) or ""
            out.append(_mk(c, source, body, offs[s], offs[e],
                           heading=getattr(node, "name", ""),
                           extra={"kind": "def", "docstring": doc[:200]}))

        # module-level code: imports, constants, __main__ — often where config lives
        loose = [i for i in range(len(lines)) if i not in covered and lines[i].strip()]
        if loose:
            body = "\n".join(lines[i] for i in loose)
            if len(body) >= c.min_chars:
                out.append(_mk(c, source, body, offs[loose[0]], offs[loose[-1] + 1],
                               heading="<module>", extra={"kind": "module"}))
        return out


CHUNKERS = {
    "fixed": FixedSizeChunker,
    "structure": StructureChunker,
    "ast": PythonASTChunker,
}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def chunk_documents(documents: Iterable[dict], cfg: ChunkConfig,
                    strategy: str = "structure") -> list[Chunk]:
    """`documents`: [{"source": "faq.md", "text": "..."}, ...]

    Python sources are routed to the AST chunker automatically when
    strategy="structure" or "ast"; everything else uses the named strategy.
    """
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown strategy {strategy!r}; pick from {list(CHUNKERS)}")

    generic = CHUNKERS[strategy](cfg)
    code = PythonASTChunker(cfg) if strategy in {"structure", "ast"} else generic

    out: list[Chunk] = []
    for d in documents:
        src = d.get("source") or d.get("path") or "unknown"
        chunker = code if str(src).endswith(".py") else generic
        out.extend(chunker.chunk(str(src), d["text"]))
    return out


def chunk_stats(chunks: Sequence[Chunk]) -> dict:
    lens = [len(c.text) for c in chunks] or [0]
    return {
        "n_chunks": len(chunks),
        "avg_chars": sum(lens) / len(lens),
        "max_chars": max(lens),
        "min_chars": min(lens),
        "total_chars": sum(lens),
    }
