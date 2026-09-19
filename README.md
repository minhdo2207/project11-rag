# Project 11 — Retrieval-Augmented Generation (RAG): Concepts & Applications

Ứng dụng RAG để phát hiện **mã độc trong các gói PyPI**, dựa trên bài tham chiếu
*"Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?"* (EASE 2025).

## Mục tiêu
- Tái hiện & so sánh các phương pháp: **no-RAG / RAG / RAG + few-shot** (và fine-tuning nếu đủ tài nguyên).
- Cải tiến RAG bằng **hybrid retrieval + re-ranking + chunking theo AST** để khắc phục điểm yếu bài gốc.
- Đo mức **giảm hallucination** khi có RAG (task 2 của đề).

## Yêu cầu non-functional (đề bài)
Dữ liệu (knowledge base + embeddings) được quản lý **cả trong bộ nhớ lẫn qua file**, và
việc đổi giữa hai lớp lưu trữ **chỉ sửa 1 dòng khởi tạo**, business logic không đổi.
Điều này được hiện thực qua interface `KnowledgeRepository` (xem `src/repository/`).

```python
# Đổi backend chỉ ở 1 dòng — phần còn lại giữ nguyên:
# repo = InMemoryKnowledgeRepository()
repo = FileKnowledgeRepository(path="data/kb")
```

## Cấu trúc thư mục
```
src/
  repository/        # P1: interface + InMemory + File (persistence hoán đổi)
  models.py          # P1: dataclass Document dùng chung
  embedding.py       # P3: nhúng code/tài liệu (stub)
  retriever.py       # P3: hybrid retrieval + rerank (stub)
  llm_client.py      # P4: gọi Ollama Qwen/Mistral/Llama (stub)
  detector.py        # P4: business logic phán quyết độc/lành (stub)
  evaluation.py      # P5: metrics (stub)
notebooks/demo.ipynb # P5: demo chạy trên Colab/Jupyter
tests/               # unit test (repository)
data/                # dataset, YARA, GHSA
docs/                # báo cáo tìm hiểu tài liệu
```

## Phân công nhóm (5 người)
| Vai | Sở hữu | Nội dung |
|---|---|---|
| **P1** — Kiến trúc & Persistence | `repository/`, `models.py`, integration | interface chung, memory/file, review PR |
| **P2** — Data & Knowledge Base | `data/`, ingest | dataset gói lành/độc, YARA, GHSA |
| **P3** — Embedding & Retrieval | `embedding.py`, `retriever.py` | hybrid (BM25+vector), rerank, AST-chunk |
| **P4** — LLM & Detection | `llm_client.py`, `detector.py` | Ollama, prompt, no-RAG/RAG/few-shot |
| **P5** — Evaluation & Demo | `evaluation.py`, `notebooks/` | metrics, biểu đồ, demo |

## Chạy test
```bash
python -m pytest tests/ -v
```

## Tài liệu
