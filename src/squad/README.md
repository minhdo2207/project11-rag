# RAG + guardrail trên SQuAD 2.0

Luồng thực nghiệm cho đề tài *RAG as a Hallucination Guardrail for a Small LLM*: đo xem RAG và
các bước kiểm tra sau khi sinh giảm bịa đặt được bao nhiêu trên model nhỏ, và phải trả giá bao
nhiêu câu bị từ chối nhầm.

Khác với [`src/crag`](../crag/README.md) (lọc tài liệu **trước** khi sinh), luồng này kiểm tra
**câu trả lời sau khi sinh** rồi mới quyết định trả lời hay từ chối.

## Luồng chạy

```
SQuAD 2.0 dev ──► build_data ──► kb.jsonl (350 đoạn) + questions_{dev,test}.jsonl
                                      │
                                      ▼
                  retrieve ── BM25 + bge-m3 (FAISS) ──► RRF ──► top-10 đoạn / câu
                                      │
                                      ▼
                  generate ── c0: không tài liệu · c1: 5 đoạn đứng đầu
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
       guardrail_scores (chỉ c1)                 label ──► judge ──► label --judge
   ctx_cos · lexical · HHEM · relevance      accurate / hallucination / missing
                 └────────────────────┬────────────────────┘
                                      ▼
                  tune_eval tune   dò ngưỡng trên dev, đóng băng
                  tune_eval eval   bảng C0–C6 trên dev / test
```

| Module | Việc |
|---|---|
| `build_data.py` | Lấy 10 trong 35 bài theo seed 42, chia **theo bài** (3 dev, 7 test), khử đoạn trùng, chèn tên bài vào đầu mỗi đoạn |
| `retrieve.py` | Dựng index một lần; BM25 và bge-m3 mỗi bên 50 ứng viên, gộp bằng RRF (k = 60); đo recall@k, MRR |
| `generate.py` | Gọi Ollama qua `/api/chat`: system prompt + ví dụ mẫu theo lượt, temperature 0, seed 42, tối đa 64 token |
| `guardrail_scores.py` | Tính các điểm guardrail cho câu trả lời của c1, chưa áp ngưỡng |
| `hhem.py` | Nạp `vectara/hallucination_evaluation_model` thủ công (mã đi kèm lỗi với transformers 5.x) |
| `label.py` | Gán nhãn: từ chối → missing; câu unanswerable mà trả lời → hallucination; F1 ≥ 0,5 → accurate; còn lại chờ giám khảo |
| `judge.py` | Giám khảo LLM trả `{"correct": true/false}` cho các câu F1 thấp |
| `analyze_scores.py` | AUC tách câu đúng khỏi câu bịa của từng điểm |
| `tune_eval.py` | Dò ngưỡng (trần từ chối nhầm 0,10), bảng C0–C6, bootstrap CI, McNemar, đường đánh đổi |

## Cấu hình

| Cấu hình | Điểm dùng để chặn | Định nghĩa |
|---|---|---|
| C0 | — | không RAG |
| C1 | — | RAG, 5 đoạn |
| C2 | `ctx_cos` | cosine bge-m3 cao nhất giữa câu hỏi và toàn bộ kho đoạn |
| C3 | `lexical` | tỷ lệ từ nội dung của câu trả lời có trong 5 đoạn |
| C4 | `hhem_qa_max` | HHEM, hypothesis = câu hỏi + câu trả lời, lấy max trên từng đoạn |
| C4p | `hhem_ans_concat` | HHEM theo proposal gốc: hypothesis = câu trả lời, premise = 5 đoạn ghép lại |
| C5 | `relevance` | cosine bge-m3 giữa câu hỏi và câu trả lời |
| C6 | cả bốn điểm của C2, C3, C4, C5 | dò đồng thời bốn ngưỡng |

Điểm dưới ngưỡng thì hệ thống từ chối. C2–C6 dùng lại đúng câu trả lời của C1, chỉ khác bước
quyết định, nên không phải sinh lại.

## Chạy lại

Cần Python 3.11+, GPU NVIDIA, [Ollama](https://ollama.com), và các gói ở mục `src/squad` của
`requirements.txt`.

```bash
ollama pull llama3:8b-instruct-q4_K_M
ollama pull phi3:3.8b-mini-4k-instruct-q4_K_M
ollama pull llama3.3:70b                      # giám khảo

python -m src.squad.build_data                # tự tải SQuAD 2.0 dev (4,4 MB)
python -m src.squad.retrieve
for s in dev test; do for m in llama3 phi3; do for c in c0 c1; do
  python -m src.squad.generate --model $m --config $c --split $s
done; done; done
for s in dev test; do
  python -m src.squad.label --split $s
  python -m src.squad.guardrail_scores --split $s
  python -m src.squad.judge --split $s --judge llama3.3:70b --host http://127.0.0.1:11435
  python -m src.squad.label --split $s --judge llama3.3:70b
done
python -m src.squad.analyze_scores --split dev
python -m src.squad.tune_eval tune
python -m src.squad.tune_eval eval --split dev
python -m src.squad.tune_eval eval --split test
python -m src.squad.tune_eval sweep
```

Dữ liệu nằm ở `data/squad/`, kết quả ở `results/squad/`; cả hai đã có trong `.gitignore`.
Mọi bước sinh và chấm đều chạy tiếp được nếu bị ngắt.

Lưu ý khi chạy:

- Giám khảo 70B cần cả hai card 24 GB. Nếu dịch vụ Ollama của máy bị ghim vào một card, chạy
  một phiên riêng: `OLLAMA_HOST=127.0.0.1:11435 OLLAMA_SCHED_SPREAD=1 ollama serve`.
- Trước khi sinh, kiểm tra `ollama ps`: nếu cột VRAM nhỏ hơn kích thước model thì model đang
  chạy một phần trên CPU và sẽ chậm đi nhiều lần.
- `guardrail_scores` và `retrieve` tự chọn GPU đang trống nhất; đặt `CUDA_VISIBLE_DEVICES` để
  chọn tay.

## Kết quả trên test (2.517 câu, ngưỡng đóng băng từ dev)

| Model | Cấu hình | Acc | Hal | Miss | Truthfulness | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Llama-3-8B (4-bit) | C0 không RAG | 0,210 | 0,656 | 0,135 | −0,446 | 0,176 | 0,097 |
| | C1 RAG | 0,470 | 0,393 | 0,137 | 0,078 | 0,269 | 0,016 |
| | C4 + HHEM | 0,420 | 0,213 | 0,368 | **0,207** | 0,626 | 0,132 |
| Phi-3-mini (4-bit) | C0 không RAG | 0,204 | 0,786 | 0,010 | −0,582 | 0,010 | 0,010 |
| | C1 RAG | 0,430 | 0,506 | 0,064 | −0,075 | 0,116 | 0,017 |
| | C4 + HHEM | 0,406 | 0,334 | 0,260 | **0,072** | 0,425 | 0,110 |

Truthfulness = Acc − Hal. Từ chối đúng = tỷ lệ câu unanswerable được từ chối; từ chối nhầm =
tỷ lệ câu answerable bị từ chối. Giám khảo: `llama3.3:70b`.

Truy hồi (RRF) trên test: recall@1 = 0,785, recall@5 = 0,956, MRR@10 = 0,859.

## Giới hạn

- Giám khảo chưa được đối chứng bằng người chấm tay, và cùng họ Llama với một model bị chấm.
- Trần từ chối nhầm 0,10 do nhóm tự đặt; trên test C4 vượt nhẹ (0,132 và 0,110).
- C4 khác định nghĩa trong proposal (hypothesis có thêm câu hỏi); C4p là bản đúng proposal.
- Sinh ở temperature 0 vẫn có thể lệch vài câu giữa các lần chạy khi nhiều request chạy song
  song trên GPU; các bước còn lại cho kết quả lặp lại được.
- Một bộ dữ liệu, một seed. Luồng này chưa nối vào `KnowledgeRepository` / `LLMClient` của `src/`.
