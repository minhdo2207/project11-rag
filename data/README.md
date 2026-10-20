# Data Directory

Thư mục này chứa dữ liệu cho đề tài **RAG as a Hallucination Guardrail for a Small LLM**: đo xem RAG và các bước kiểm tra sau khi sinh giảm bịa đặt được bao nhiêu trên model nhỏ, và phải trả giá bao nhiêu câu bị từ chối nhầm.

Guardrail kiểm tra câu trả lời **sau khi sinh** rồi mới quyết định trả lời hay từ chối. Cách chấm theo ba nhãn của CRAG (arXiv 2406.04744):
- ✅ Câu đúng: **+1**
- ⚪ Câu từ chối: **0**
- ❌ Câu bịa: **−1**

## Giới thiệu bộ dữ liệu

**SQuAD 2.0** (Stanford Question Answering Dataset v2) là bộ hỏi-đáp từ Wikipedia, gồm:
- Câu hỏi **có thể trả lời** — đáp án nằm trong đoạn văn
- Câu hỏi **không thể trả lời** (`is_impossible: true`) — dùng để kiểm tra xem model có bịa không

Pipeline chọn 10 bài (seed 42), chia 3 bài dev / 7 bài test, khử đoạn trùng rồi tạo:

| File | Mô tả |
|------|--------|
| `kb.jsonl` | ~350 đoạn văn — Knowledge Base cho RAG |
| `questions_dev.jsonl` | ~350 câu hỏi — dùng để dò ngưỡng guardrail |
| `questions_test.jsonl` | ~2500 câu hỏi — chỉ dùng để báo cáo kết quả |
| `stats.json` | Thống kê số bài, đoạn, câu hỏi |


## Cấu trúc thư mục
 data/squad/
├── raw/
│ └── dev-v2.0.json # file gốc SQuAD 2.0 (tự tải về)
└── processed/
├── kb.jsonl
├── questions_dev.jsonl
├── questions_test.jsonl
└── stats.json 
  
# Data

```bash
# Cài thư viện
pip install -r requirements.txt

# Tải SQuAD 2.0 (~4.4MB) và dựng Knowledge Base
python -m src.squad.build_data
```
## Schema các file

### kb.jsonl — Knowledge Base
Mỗi dòng là một đoạn văn Wikipedia:
| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `pid` | string | ID đoạn văn (SHA1) |
| `title` | string | Tên bài Wikipedia |
| `context` | string | Nội dung đoạn văn |

### questions_dev.jsonl & questions_test.jsonl
Mỗi dòng là một câu hỏi:
| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `qid` | string | ID câu hỏi gốc từ SQuAD |
| `split` | string | `"dev"` hoặc `"test"` |
| `title` | string | Bài Wikipedia liên quan |
| `question` | string | Câu hỏi |
| `answers` | list[string] | Đáp án đúng (rỗng nếu không trả lời được) |
| `is_impossible` | bool | `true` = câu hỏi bẫy, model không được bịa |

### stats.json
Thống kê tổng quan sau khi build:
| Trường | Mô tả |
|--------|-------|
| `seed` | Random seed (42) |
| `kb_paragraphs` | Tổng số đoạn trong KB |
| `splits.dev.articles` | Danh sách bài dùng cho dev |
| `splits.dev.paragraphs` | Số đoạn dev |
| `splits.dev.questions` | Tổng câu hỏi dev |
| `splits.dev.answerable` | Câu có đáp án |
| `splits.dev.unanswerable` | Câu không trả lời được |
| `splits.test.*` | Tương tự cho test set |


## Chạy pipeline tiếp theo

```bash
python -m src.squad.build_data
```

