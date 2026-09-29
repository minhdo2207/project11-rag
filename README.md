# Project 11 — Retrieval-Augmented Generation (RAG): Concepts & Applications

Dùng RAG để **phát hiện mã độc trong các gói PyPI**, dựa trên bài tham chiếu
*"Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?"* (EASE 2025).

Bài gốc kết luận RAG cho kết quả trung bình, thua few-shot (97%). Nhóm mình đi theo hướng:
**(1)** chẩn đoán vì sao RAG kém, **(2)** dựng lại RAG bằng kỹ thuật 2026 (cAST, hybrid, rerank),
**(3)** thêm cổng kiểm chứng để giảm bịa bằng chứng (task "RAG chống hallucination").

## Cài & chạy

```bash
pip install -r requirements.txt
make test        # chay unit test
```

Yêu cầu Python 3.11+. Backend `memory`/`file` không cần cài gì thêm ngoài `pyyaml`.
Backend `qdrant` là tuỳ chọn (`pip install qdrant-client`).

## Yêu cầu non-functional (đề bài)

Dữ liệu (knowledge base + embeddings) phải quản lý được **cả trong RAM lẫn qua file**, và
đổi giữa hai lớp lưu trữ chỉ được sửa **rất ít code**. Bọn mình giải bằng interface
`KnowledgeRepository` + một factory đọc từ `config.yaml`:

```python
from src.config import load_config
from src.repository.factory import build_repository

repo = build_repository(load_config())   # backend lay tu config.yaml, doi 1 dong la xong
```

Đổi `storage.backend` trong `config.yaml` giữa `memory | file | qdrant` — business logic
(retriever, detector) không phải sửa gì.

## Cấu trúc & từng phần implement ra sao

```
src/
  config.py            # doc config.yaml
  models.py            # dataclass Document dung chung
  repository/          # lop luu tru (memory / file / qdrant) - phan cua P1
  embedding.py         # nhung text/code -> vector (P3)
  retriever.py         # truy xuat bang chung (P3)
  llm_client.py        # goi LLM qua Ollama (P4)
  detector.py          # phan quyet doc/lanh (P4)
  evaluation.py        # metrics (P5)
```

### `config.py` — cấu hình tập trung
Đọc `config.yaml` thành object `Config`, cho lấy key kiểu `cfg.get("pipeline.mode")`.
Mọi tham số (backend, model, bật/tắt từng tầng RAG) nằm ở đây → thí nghiệm reproducible,
đổi cấu hình không đụng code. **Đã implement + có test.**

### `models.py` — `Document`
Đơn vị tri thức: `doc_id`, `text`, `embedding`, `metadata`, `label`. Có `preview()` để in log
gọn và `is_labeled`. Mọi module trao đổi qua kiểu này để interface ổn định. **Đã implement.**

### `repository/` — lớp lưu trữ (phần chính của P1)
- `base.py`: interface trừu tượng `KnowledgeRepository` (`add / get / all / search / count / clear`).
- `in_memory.py`: giữ trong `dict`, tìm bằng cosine. Nhanh, mất khi tắt tiến trình.
- `file_based.py`: lưu ra JSON, còn dữ liệu sau khi tắt.
- `qdrant_repo.py`: dùng Qdrant, `:memory:` và local-file **cùng một API**; tuỳ chọn.
- `factory.py`: `build_repository(cfg)` dựng đúng backend từ config.
- Cả `in_memory` và `file` share `_similarity.py` nên **cho kết quả `search` giống hệt nhau**
  — có test chứng minh điều đó (`tests/test_repository.py` chạy song song 2 backend).

**Đã implement + có test.**

### `embedding.py` — (P3, đang là interface)
Interface `Embedder.embed(text) -> vector`. Sẽ hiện thực bằng `bge-m3` hoặc `Qwen3-Embedding`
qua sentence-transformers. Với code thì cân nhắc model embedding chuyên code.

### `retriever.py` — (P3, đang là interface)
Interface `Retriever.retrieve(query, top_k)`. Kế hoạch hiện thực theo 3 mức, bật dần qua config:
1. `vector`: gọi thẳng `repository.search`.
2. `hybrid`: BM25 (bắt token chính xác như `eval`, `b64decode`) + vector, hợp nhất bằng RRF.
3. thêm rerank bằng cross-encoder + ngưỡng từ chối (dưới ngưỡng trả 0 chunk).

### `llm_client.py` — (P4, đang là interface)
Interface `LLMClient.generate(prompt)`. Hiện thực gọi Ollama (`qwen2.5`, đổi model qua config).
**Lưu ý:** đặt `max_tokens` đủ rộng, tránh cắt cụt câu trả lời (bài học từ seminar CRAG —
câu bị cắt dễ bị chấm nhầm là "bịa").

### `detector.py` — (P4)
`MaliciousCodeDetector.detect(code) -> Verdict`. Có cờ `use_rag` để so sánh no-RAG vs RAG.
Business logic chỉ phụ thuộc `Retriever` + `LLMClient` (interface), **không biết** dữ liệu nằm
ở RAM hay file — đó là điểm mấu chốt của thiết kế.

### `evaluation.py` — (P5)
`accuracy` và `balanced_accuracy` (quan trọng khi dữ liệu lệch: gói lành nhiều hơn gói độc).
Sẽ mở rộng thêm precision/recall/F1, false-negative rate, và chẩn đoán context-sufficiency
(tách lỗi retrieval khỏi lỗi generation). **Cơ bản đã implement + có test.**

## Bật/tắt từng tầng (ablation)

Các cờ trong `config.yaml > pipeline` (`chunking`, `retrieval`, `rerank`, `mode`, `verify`)
cho phép bật dần từng tầng và đo đóng góp riêng của mỗi tầng — vừa để phát triển an toàn,
vừa là số liệu cho báo cáo.

## Phân công nhóm (5 người)

| Vai | Sở hữu | Nội dung |
|---|---|---|
| **P1** | `repository/`, `config.py`, `models.py`, CI | interface, persistence, integration |
| **P2** | `data/`, ingest | dataset gói lành/độc, YARA, GHSA |
| **P3** | `embedding.py`, `retriever.py` | hybrid + rerank + cAST |
| **P4** | `llm_client.py`, `detector.py` | Ollama, no-RAG/RAG/few-shot |
| **P5** | `evaluation.py`, `notebooks/` | metrics, chẩn đoán, demo |

Quy tắc làm việc chung: xem [CONTRIBUTING.md](CONTRIBUTING.md).
