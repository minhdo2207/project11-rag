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

## Running the detector

The full chain (ingest → embed → retrieve → LLM → verdict) runs against a local
[Ollama](https://ollama.com):

```bash
ollama pull qwen2.5:7b && ollama pull bge-m3      # models named in config.yaml
python scripts/fetch_data.py                      # knowledge base data (see data/README.md)
python scripts/run_detector.py setup.py --mode rag --ingest
python scripts/run_detector.py setup.py --mode no_rag    # zero-shot baseline, no KB needed
```

`--mode` is `no_rag | rag | rag_fewshot` (overrides `pipeline.mode`). Exit code 2 means the
LLM answered with something unreadable (`label: unknown`).

Two things to know before reading results off a full run:

- `ingest_all` loads both the **train and test** package files into the knowledge base. If you
  evaluate on the test split, retrieval can return the very sample being judged together with
  its label. Ingest train only when measuring accuracy.
- `FileKnowledgeRepository` rewrites its whole JSON file on every `add`. Ingesting the full
  dataset (~190k documents) that way is very slow; use the `memory` or `qdrant` backend for a
  full ingest, or batch writes with `add_many`.

## Architecture & Design Principles

### Storage Abstraction

A core requirement: data (knowledge base + embeddings) must be manageable **both in-memory and persisted to disk**, 
with minimal code changes to swap backends. We achieve this via:

- **`KnowledgeRepository` interface:** Common contract for all storage backends
- **Factory pattern:** Single-line configuration swap

```python
from src.config import load_config
from src.repository.factory import build_repository

repo = build_repository(load_config())  # backend from config.yaml, one change to swap
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
  retriever.py         # Vector retrieval (hybrid + rerank prototyped in experiments/p3_retrieval)
  llm_client.py        # LLM inference (Ollama)
  detector.py          # Classification (malicious / benign)
  evaluation.py        # Metrics & diagnostics
  ingest.py            # Load YARA / PyPI packages / GHSA into the repository
experiments/
  p3_retrieval/        # Seven retrieval improvement scenarios (S1-S7), see its README
scripts/               # fetch_data.py, run_detector.py
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
- `base.py`: Abstract `KnowledgeRepository` interface (`add / add_many / get / all / search / count / clear`)
- `in_memory.py`: Fast, in-process, ephemeral (dict + cosine similarity)
- `file_based.py`: Persistent JSON storage
- `qdrant_repo.py`: Vector database option (`:memory:` or local disk, same API)
- `_similarity.py`: Shared cosine implementation → identical retrieval results across backends
- Unit tests verify **both backends produce identical outputs** (`test_repository.py`)

**`embedding.py`** — Vectorization
- Interface: `Embedder.embed(text) -> vector`, plus `embed_batch`
- `OllamaEmbedder`: real model through Ollama (`bge-m3`, `nomic-embed-text`, ...)
- `HashingEmbedder`: token hashing, offline and stable across runs; for tests and demos only,
  it carries no semantics
- `build_embedder(cfg)` picks one from `embedding.provider`

**`retriever.py`** — Retrieval
- Interface: `Retriever.retrieve(query, top_k) -> list[Document]`
- `VectorRetriever`: embeds the query and searches the repository, optional cosine threshold
- `embed_missing(repo, embedder)` embeds documents that were ingested without a vector
- Hybrid BM25 + vector, reranking and the other P3 experiments live in
  `experiments/p3_retrieval` and are not wired in here yet (`pipeline.retrieval: hybrid` raises)

**`llm_client.py`** — LLM inference
- Interface: `LLMClient.generate(prompt, system) -> str`
- `OllamaClient` calls `/api/chat` (stdlib `urllib`, no extra dependency); `build_llm_client(cfg)`
- Strips `<think>` blocks and logs a warning when the answer hits `max_tokens`
  (truncated answers confound evaluation, so keep `llm.max_tokens` generous)

**`detector.py`** — Malicious code detection
- `MaliciousCodeDetector.detect(code) -> Verdict` with `label` (`malicious | benign | unknown`),
  `reason`, `suspicious_lines`, the `evidence` actually put in the prompt, and the `raw` answer
- Three modes from `pipeline.mode`: `no_rag` (zero-shot baseline), `rag`, `rag_fewshot`
- The LLM is asked for JSON; parsing tolerates code fences and truncated output, and an answer
  that cannot be read becomes `unknown` instead of a guess
- Business logic **storage-agnostic**: depends only on `Retriever` + `LLMClient` interfaces

**`evaluation.py`** — Metrics & diagnostics
- `accuracy`, `balanced_accuracy` for the malicious/benign task (crucial for class imbalance:
  benign >> malicious). Precision, recall, F1 and false-negative rate are not implemented yet
- `three_label_report` / `report_from_file`: accurate / hallucinated / missing / truthfulness
  (the CRAG scoring), used by the CRAG ladder below
- Advanced: **context-sufficiency analysis** (isolate retrieval vs. generation errors) and 
  **evidence localization** (does LLM cite correct lines?)

**`crag/`** — Hallucination experiments on the CRAG benchmark (P5)
- A ladder of configurations B0–B5 (no documents → snippets → hybrid retrieval → reranking →
  citations → refusal gate), each changing exactly one thing, scored with two independent graders
- Separate from the PyPI detector: it measures how much each RAG component reduces hallucination
  on general question answering with `llama3:8b`. Needs a GPU and a large data download
- Full description and how to re-run: [src/crag/README.md](src/crag/README.md)

**`squad/`** — RAG + post-generation guardrail on SQuAD 2.0 (P5)
- Configurations C0–C6 (no RAG → hybrid RAG → one check each → all checks): context check,
  lexical grounding, HHEM grounding and answer relevance decide whether to answer or abstain;
  thresholds are tuned on a dev split and frozen before the test split is scored
- Reports truthfulness together with correct-abstention and over-refusal rates, with bootstrap
  confidence intervals and McNemar tests, for `llama3:8b` and `phi3:mini` (both 4-bit)
- Full description and how to re-run: [src/squad/README.md](src/squad/README.md)

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
