# Project 11 — Retrieval-Augmented Generation (RAG): Concepts & Applications

## Overview

Detecting **malicious source code in PyPI packages** using Retrieval-Augmented Generation (RAG) and LLMs.
This work builds on *"Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?"* (EASE 2025).

### Why This Matters

The reference paper found that RAG achieved only **middling results** (~79% accuracy), underperforming few-shot prompting (97%).
Our approach investigates and improves RAG for this task through:

1. **Diagnosis:** Why does RAG underperform? Retrieval quality? LLM hallucination? Context mismatch?
2. **Modern RAG techniques:** Hybrid retrieval (BM25 + dense vectors, RRF ranking), cross-encoder reranking, 
   cAST-based code chunking for better semantic boundaries
3. **Hallucination mitigation:** Grounding gate (evidence-based verification) to reject LLM answers 
   unsupported by retrieved context

## Installation & Running Tests

```bash
pip install -r requirements.txt
make test        # run unit tests
```

Requires Python 3.11+. The `memory` and `file` backends only need `pyyaml`.
The `qdrant` backend is optional (`pip install qdrant-client`).

## Architecture & Design Principles

### Storage Abstraction

A core requirement: data (knowledge base + embeddings) must be manageable **both in-memory and persisted to disk**, 
with minimal code changes to swap backends. We achieve this via:

- **`KnowledgeRepository` interface:** Common contract for all storage backends
- **Factory pattern:** Single-line configuration swap

```python
from src.config import load_config
from src.repository.factory import build_repository

repo = build_repository(load_config())   # backend from config.yaml, one change to swap
```

Change `storage.backend` in `config.yaml` to `memory | file | qdrant`. Business logic 
(retriever, detector) remains **storage-agnostic** — a key separation of concerns.

## Codebase Structure

```
src/
  config.py            # Load and parse config.yaml
  models.py            # Shared Document dataclass
  repository/          # Storage abstraction (memory / file / qdrant)
  embedding.py         # Text/code → vectors
  retriever.py         # Hybrid retrieval + reranking
  llm_client.py        # LLM inference (Ollama)
  detector.py          # Classification (malicious / benign)
  evaluation.py        # Metrics & diagnostics
```

### Key Components

**`config.py`** — Centralized configuration
- Loads `config.yaml` into a nested-key object: `cfg.get("pipeline.mode")`
- All parameters (backend, model choice, feature toggles) in one place
- Reproducible experiments, no code changes to adjust settings

**`models.py`** — `Document` dataclass
- Common knowledge unit: `doc_id`, `text`, `embedding`, `metadata`, `label`
- `is_labeled` and `preview()` helpers for introspection
- Interface stability across modules

**`repository/`** — Storage layer (in-memory, file-based, Qdrant)
- `base.py`: Abstract `KnowledgeRepository` interface (`add / get / all / search / count / clear`)
- `in_memory.py`: Fast, in-process, ephemeral (dict + cosine similarity)
- `file_based.py`: Persistent JSON storage
- `qdrant_repo.py`: Vector database option (`:memory:` or local disk, same API)
- `_similarity.py`: Shared cosine implementation → identical retrieval results across backends
- Unit tests verify **both backends produce identical outputs** (`test_repository.py`)

**`embedding.py`** — Vectorization
- Interface: `Embedder.embed(text) -> vector`
- Candidates: `bge-m3`, `Qwen3-Embedding`, or code-specialized embedders

**`retriever.py`** — Retrieval pipeline
- Interface: `Retriever.retrieve(query, top_k) -> list[Document]`
- Three progressive modes (configurable):
  1. **Dense-only:** Vector search via repository
  2. **Hybrid:** BM25 (exact tokens: `eval`, `b64decode`) + dense vectors, combined with RRF
  3. **Reranked:** Cross-encoder reranking + confidence threshold (reject below threshold)

**`llm_client.py`** — LLM inference
- Interface: `LLMClient.generate(prompt) -> str`
- Backend: Ollama (`qwen2.5` or configurable)
- **Critical:** Set `max_tokens` generously to avoid truncation (truncated responses confound evaluation)

**`detector.py`** — Malicious code detection
- `MaliciousCodeDetector.detect(code) -> Verdict` (malicious / benign + confidence)
- Flag `use_rag` to toggle RAG vs. zero-shot baseline
- Business logic **storage-agnostic**: depends only on `Retriever` + `LLMClient` interfaces

**`evaluation.py`** — Metrics & diagnostics
- `accuracy`, `balanced_accuracy` (crucial for class imbalance: benign >> malicious)
- Extensible: precision, recall, F1, false-negative rate
- Advanced: **context-sufficiency analysis** (isolate retrieval vs. generation errors) and 
  **evidence localization** (does LLM cite correct lines?)

### Ablation Studies

Feature flags in `config.yaml` (`pipeline.chunking`, `pipeline.retrieval`, `pipeline.rerank`, etc.)
enable progressive evaluation of each component's contribution—both for safe development and 
for reporting in the final paper.

## Team Roles & Contributions

| Role | Owns | Responsibilities |
|------|------|------------------|
| **P1 (Architecture & Persistence)** | `repository/`, `config.py`, `models.py` | Storage abstraction, configuration, integration |
| **P2 (Data & Ingestion)** | `src/ingest.py`, `scripts/fetch_data.py` | PyPI packages (benign & malicious), YARA rules, GHSA advisories |
| **P3 (Embedding & Retrieval)** | `embedding.py`, `retriever.py` | Dense embeddings, hybrid retrieval, reranking, cAST chunking |
| **P4 (LLM & Detection)** | `llm_client.py`, `detector.py` | Ollama integration, classification logic, RAG vs. zero-shot |
| **P5 (Evaluation & Experiments)** | `evaluation.py`, `notebooks/` | Metrics, error analysis, ablation studies, visualization |

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
