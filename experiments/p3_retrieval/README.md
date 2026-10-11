# P3 — Retrieval improvement scenarios (S1–S7)

Seven self-contained retrieval packages. Each one inherits everything from the
previous scenario and adds exactly one variable, so any change in the numbers can
be attributed to a single component. `s7` is the superset; `s1`–`s6` are kept
because each ships its own ablation runner (`run_sN.py`).

The full write-up (components, algorithms, measured results, recommendations) is
in [`bao-cao-7-kich-ban.pdf`](bao-cao-7-kich-ban.pdf) (Vietnamese).

| Scenario | Adds | Runner |
|---|---|---|
| S1 | dense + BM25 + RRF (hybrid retrieval) | `s1/run_s1.py` |
| S2 | structure-aware chunking, atomic blocks, enrichment header | `s2/run_s2.py` |
| S3 | cross-encoder reranking (`replace` / `rrf` fusion) | `s3/run_s3.py` |
| S4 | metadata pre-filter for scope and version | `s4/run_s4.py` |
| S5 | query understanding (diacritic folding, lexicon, multi-variant) | `s5/run_s5.py` |
| S6 | adaptive-k, MMR, U-shaped ordering | `s6/run_s6.py` |
| S7 | calibration and 4-level answer decision (`may_assert`) | `s7/run_s7.py` |

## Running

```bash
pip install numpy scipy
cd experiments/p3_retrieval/s7
python run_s7.py --demo --cost-ratio 5
```

The scripts use flat imports (`from retriever import ...`), so run them from inside
their own scenario folder. `--demo` needs no GPU and no model download.

## Things to know before relying on the numbers

- **Demo numbers are structural, not quality.** Demo mode uses a hashing embedder and
  a stub reranker on a tiny corpus. Re-measure on real data and real models.
- **Exit code 1 from `run_s1.py --demo` is intentional.** Every runner has a stop
  condition. On the demo set all three modes reach recall@5 = 1.0, so the eval set
  cannot separate them and the script exits non-zero on purpose.
- **S4 and S7 depend on today's date.** The metadata filter uses `date.today()` as
  `as_of`, so the demo output drifts as documents expire. Pass `--as-of` to `run_s4.py`
  for a reproducible run.
- **Numbers in the PDF are not all reproduced by the code as shipped.** For example
  the S2 `structure` variant reaches `span_coverage@5 = 1.000` here versus 0.889 in the
  report, and the S3 and S4 demo figures differ too. The mechanisms and conclusions
  match; the exact figures need a re-run before they go into the final report.
- **Not wired into `src/` yet.** These packages define their own `Document`/`Hit` and
  persist their index with `pickle` + JSONL, so they do not go through
  `KnowledgeRepository` (memory/file/qdrant). Integrating them with the shared
  interface is still open. Do not load a `bm25.pkl` index from an untrusted location,
  because unpickling can execute code.
- The PyPI malware-detection task is not modelled here; the demo corpus is a generic
  Vietnamese Q&A set. Plugging in the PyPI knowledge base (P2) is a follow-up.
