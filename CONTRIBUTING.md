# Contribution rules (read before you commit)

This file lets the whole team work in one repo without stepping on each other. Everyone
reads it once.

## 1. Branches

- `main`: must always work (green tests). Never push directly to `main`.
- Everyone works on their own branch, named `<role>/<task>`:
  - examples: `p3/hybrid-retrieval`, `p5/eval-metrics`, `p1/qdrant-repo`
- When you are done, open a Pull Request into `main`; P1 reviews and merges.

## 2. Commits

- Small, frequent commits. One commit = one meaningful change; don't pile everything into one
  big commit.
- Messages follow Conventional Commits (Vietnamese is fine too):
  ```
  <type>: <short description>

  <more explanation if needed>
  ```
- Common `type` values: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`.
- Examples:
  - `feat: add hybrid retriever (BM25 + vector)`
  - `test: add tests for the repository factory`
  - `fix: handle missing keys when reading config`

## 3. Before you push

Run all three steps and fix whatever is red before pushing:

```bash
make format   # ruff format
make lint     # ruff check
make test     # pytest
```

Or run them by hand:

```bash
ruff format . && ruff check . && python -m pytest -q
```

## 4. Module boundaries (who touches which file)

| Role | May edit | Do not touch |
|------|----------|--------------|
| P1 | `src/repository/`, `src/config.py`, `src/models.py`, CI, architecture docs | other people's ML logic |
| P2 | `data/`, ingest scripts | the interfaces in `repository/` |
| P3 | `src/embedding.py`, `src/retriever.py` | `detector.py` |
| P4 | `src/llm_client.py`, `src/detector.py` | `retriever.py` |
| P5 | `src/evaluation.py`, `notebooks/` | the core modules |

Interfaces (the abstract classes) are decided by P1. If you need to change an interface, tell
P1 instead of editing it yourself, because it will break other people's code.

## 5. Reviewing PRs

- Keep PRs small (< ~300 lines) and easy to read.
- The PR description states what it does, how it was tested, and what is still missing.
- P1 (or whoever is designated) reads the change and runs the tests locally before merging.

## 6. Don't commit junk

Do not push: `__pycache__/`, `.venv/`, large data, downloaded model files, `data/kb/`
(generated index). These are already in `.gitignore`.
