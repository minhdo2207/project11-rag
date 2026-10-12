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

## Kết quả

Chạy trên **toàn bộ** câu hỏi của hai tập: dev 1.021 câu (dùng để dò ngưỡng) và test 2.517 câu
(ngưỡng đóng băng từ dev). Hai model sinh đều là bản 4-bit qua Ollama; giám khảo là `llama3.3:70b`.

Tóm tắt:

- **RAG là đóng góp lớn nhất.** Trên test, thêm tài liệu làm Truthfulness tăng +0,524 (Llama-3) và
  +0,506 (Phi-3).
- **Chỉ một check có tác dụng rõ:** HHEM với hypothesis là câu hỏi ghép câu trả lời (C4). Trên test,
  Hal của Llama-3 giảm từ 0,393 xuống 0,213 và Truthfulness tăng +0,129; với Phi-3 tăng +0,147.
- **Ba check còn lại và HHEM theo định nghĩa gốc gần như vô ích với Llama-3.** Khi dò cả bốn ngưỡng
  cùng lúc (C6), cách dò tự tắt ba check kia nên C6 trùng C4.
- **Phi-3 có RAG vẫn bịa nhiều hơn đúng** (Truthfulness −0,075 trên test); C4 đưa nó lên 0,072.

### Dữ liệu

| Tập | Bài | Đoạn | Câu hỏi | Answerable | Unanswerable |
|---|---|---|---|---|---|
| dev | 3 | 89 | 1021 | 584 | 437 |
| test | 7 | 261 | 2517 | 1314 | 1203 |

### Truy hồi

Tìm đoạn gốc của câu hỏi trong 350 đoạn, tính trên mọi câu hỏi.

| Tập | Cách truy hồi | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|
| dev | BM25 | 0,832 | 0,921 | 0,945 | 0,974 | 0,883 |
| dev | bge-m3 | 0,777 | 0,899 | 0,928 | 0,960 | 0,844 |
| dev | BM25 + bge-m3 (RRF) | 0,834 | 0,953 | 0,978 | 0,983 | 0,895 |
| test | BM25 | 0,756 | 0,889 | 0,917 | 0,947 | 0,828 |
| test | bge-m3 | 0,730 | 0,892 | 0,935 | 0,971 | 0,819 |
| test | BM25 + bge-m3 (RRF) | 0,785 | 0,922 | 0,956 | 0,979 | 0,859 |

### Bảng C0–C6 trên test (2.517 câu)

**Llama-3-8B-Instruct (4-bit)**

| Cấu hình | Acc | Hal | Miss | Truthfulness | 95% CI | Coverage | Selective risk | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|---|
| C0 không RAG | 0,210 | 0,656 | 0,135 | −0,446 | −0,477 … −0,414 | 0,865 | 0,758 | 0,176 | 0,097 |
| C1 RAG | 0,470 | 0,393 | 0,137 | 0,078 | 0,041 … 0,115 | 0,863 | 0,455 | 0,269 | 0,016 |
| C2 + context check | 0,437 | 0,329 | 0,234 | 0,108 | 0,072 … 0,142 | 0,766 | 0,429 | 0,396 | 0,086 |
| C3 + lexical | 0,464 | 0,377 | 0,158 | 0,087 | 0,050 … 0,124 | 0,842 | 0,448 | 0,294 | 0,033 |
| C4 + HHEM (câu hỏi + câu trả lời) | 0,420 | 0,213 | 0,368 | 0,207 | 0,176 … 0,236 | 0,632 | 0,336 | 0,626 | 0,132 |
| C4p + HHEM theo proposal gốc | 0,426 | 0,346 | 0,228 | 0,079 | 0,043 … 0,114 | 0,772 | 0,449 | 0,356 | 0,112 |
| C5 + relevance | 0,423 | 0,350 | 0,228 | 0,073 | 0,037 … 0,108 | 0,772 | 0,453 | 0,353 | 0,113 |
| C6 đủ bốn check | 0,420 | 0,213 | 0,368 | 0,207 | 0,176 … 0,236 | 0,632 | 0,336 | 0,626 | 0,132 |

| So sánh | Bớt câu bịa | Thêm câu bịa | McNemar p | Δ Truthfulness | 95% CI |
|---|---|---|---|---|---|
| C0 → C1 | 805 | 143 | 1.6e-112 | +0,524 | 0,487 … 0,560 |
| C1 → C2 | 160 | 0 | 1.4e-48 | +0,030 | 0,019 … 0,042 |
| C1 → C3 | 38 | 0 | 7.3e-12 | +0,009 | 0,004 … 0,015 |
| C1 → C4 | 453 | 0 | 8.6e-137 | +0,129 | 0,110 … 0,147 |
| C1 → C4p | 117 | 0 | 1.2e-35 | +0,002 | −0,010 … 0,013 |
| C1 → C5 | 108 | 0 | 6.2e-33 | −0,005 | −0,016 … 0,007 |
| C1 → C6 | 453 | 0 | 8.6e-137 | +0,129 | 0,110 … 0,147 |

**Phi-3-mini-4k-instruct (4-bit)**

| Cấu hình | Acc | Hal | Miss | Truthfulness | 95% CI | Coverage | Selective risk | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|---|
| C0 không RAG | 0,204 | 0,786 | 0,010 | −0,582 | −0,613 … −0,548 | 0,990 | 0,794 | 0,010 | 0,010 |
| C1 RAG | 0,430 | 0,506 | 0,064 | −0,075 | −0,114 … −0,038 | 0,936 | 0,540 | 0,116 | 0,017 |
| C2 + context check | 0,403 | 0,425 | 0,172 | −0,022 | −0,058 … 0,013 | 0,828 | 0,513 | 0,269 | 0,083 |
| C3 + lexical | 0,406 | 0,395 | 0,199 | 0,012 | −0,025 … 0,045 | 0,801 | 0,493 | 0,302 | 0,104 |
| C4 + HHEM (câu hỏi + câu trả lời) | 0,406 | 0,334 | 0,260 | 0,072 | 0,038 … 0,104 | 0,740 | 0,452 | 0,425 | 0,110 |
| C4p + HHEM theo proposal gốc | 0,403 | 0,404 | 0,193 | −0,001 | −0,036 … 0,035 | 0,807 | 0,500 | 0,280 | 0,113 |
| C5 + relevance | 0,391 | 0,446 | 0,163 | −0,054 | −0,091 … −0,019 | 0,837 | 0,533 | 0,219 | 0,111 |
| C6 đủ bốn check | 0,406 | 0,334 | 0,260 | 0,072 | 0,038 … 0,104 | 0,740 | 0,452 | 0,425 | 0,110 |

| So sánh | Bớt câu bịa | Thêm câu bịa | McNemar p | Δ Truthfulness | 95% CI |
|---|---|---|---|---|---|
| C0 → C1 | 771 | 66 | 2.5e-153 | +0,506 | 0,469 … 0,542 |
| C1 → C2 | 203 | 0 | 1.6e-61 | +0,053 | 0,041 … 0,066 |
| C1 → C3 | 279 | 0 | 2.1e-84 | +0,087 | 0,072 … 0,100 |
| C1 → C4 | 432 | 0 | 1.8e-130 | +0,147 | 0,132 … 0,163 |
| C1 → C4p | 256 | 0 | 1.7e-77 | +0,075 | 0,061 … 0,089 |
| C1 → C5 | 151 | 0 | 7.0e-46 | +0,021 | 0,009 … 0,033 |
| C1 → C6 | 432 | 0 | 1.8e-130 | +0,147 | 0,132 … 0,163 |

### Bảng C0–C6 trên dev (1.021 câu, tập dò ngưỡng nên lạc quan)

**Llama-3-8B-Instruct (4-bit)**

| Cấu hình | Acc | Hal | Miss | Truthfulness | 95% CI | Coverage | Selective risk | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|---|
| C0 không RAG | 0,218 | 0,629 | 0,153 | −0,410 | −0,460 … −0,361 | 0,847 | 0,742 | 0,156 | 0,151 |
| C1 RAG | 0,525 | 0,373 | 0,102 | 0,152 | 0,096 … 0,210 | 0,898 | 0,415 | 0,220 | 0,014 |
| C2 + context check | 0,478 | 0,314 | 0,208 | 0,164 | 0,113 … 0,218 | 0,792 | 0,397 | 0,352 | 0,099 |
| C3 + lexical | 0,523 | 0,364 | 0,113 | 0,159 | 0,103 … 0,215 | 0,887 | 0,411 | 0,238 | 0,019 |
| C4 + HHEM (câu hỏi + câu trả lời) | 0,483 | 0,199 | 0,318 | 0,284 | 0,237 … 0,332 | 0,682 | 0,292 | 0,611 | 0,099 |
| C4p + HHEM theo proposal gốc | 0,479 | 0,331 | 0,190 | 0,148 | 0,096 … 0,204 | 0,810 | 0,409 | 0,311 | 0,099 |
| C5 + relevance | 0,483 | 0,336 | 0,181 | 0,147 | 0,094 … 0,202 | 0,819 | 0,410 | 0,291 | 0,099 |
| C6 đủ bốn check | 0,483 | 0,199 | 0,318 | 0,284 | 0,237 … 0,332 | 0,682 | 0,292 | 0,611 | 0,099 |

| So sánh | Bớt câu bịa | Thêm câu bịa | McNemar p | Δ Truthfulness | 95% CI |
|---|---|---|---|---|---|
| C0 → C1 | 316 | 55 | 1.2e-45 | +0,562 | 0,507 … 0,618 |
| C1 → C2 | 60 | 0 | 1.7e-18 | +0,012 | −0,008 … 0,031 |
| C1 → C3 | 9 | 0 | 3.9e-03 | +0,007 | 0,001 … 0,014 |
| C1 → C4 | 178 | 0 | 5.2e-54 | +0,132 | 0,107 … 0,161 |
| C1 → C4p | 43 | 0 | 2.3e-13 | −0,004 | −0,022 … 0,014 |
| C1 → C5 | 38 | 0 | 7.3e-12 | −0,005 | −0,023 … 0,012 |
| C1 → C6 | 178 | 0 | 5.2e-54 | +0,132 | 0,107 … 0,161 |

**Phi-3-mini-4k-instruct (4-bit)**

| Cấu hình | Acc | Hal | Miss | Truthfulness | 95% CI | Coverage | Selective risk | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|---|
| C0 không RAG | 0,214 | 0,761 | 0,025 | −0,548 | −0,595 … −0,498 | 0,975 | 0,781 | 0,016 | 0,033 |
| C1 RAG | 0,487 | 0,454 | 0,059 | 0,032 | −0,025 … 0,093 | 0,941 | 0,483 | 0,114 | 0,017 |
| C2 + context check | 0,447 | 0,383 | 0,170 | 0,064 | 0,012 … 0,119 | 0,830 | 0,462 | 0,265 | 0,099 |
| C3 + lexical | 0,457 | 0,380 | 0,163 | 0,077 | 0,024 … 0,132 | 0,837 | 0,454 | 0,249 | 0,098 |
| C4 + HHEM (câu hỏi + câu trả lời) | 0,454 | 0,339 | 0,207 | 0,116 | 0,065 … 0,168 | 0,793 | 0,427 | 0,350 | 0,099 |
| C4p + HHEM theo proposal gốc | 0,456 | 0,396 | 0,148 | 0,061 | 0,005 … 0,118 | 0,852 | 0,464 | 0,213 | 0,099 |
| C5 + relevance | 0,447 | 0,409 | 0,144 | 0,037 | −0,020 … 0,093 | 0,856 | 0,478 | 0,204 | 0,099 |
| C6 đủ bốn check | 0,454 | 0,339 | 0,207 | 0,116 | 0,065 … 0,168 | 0,793 | 0,427 | 0,350 | 0,099 |

| So sánh | Bớt câu bịa | Thêm câu bịa | McNemar p | Δ Truthfulness | 95% CI |
|---|---|---|---|---|---|
| C0 → C1 | 346 | 33 | 6.1e-67 | +0,580 | 0,520 … 0,640 |
| C1 → C2 | 73 | 0 | 2.1e-22 | +0,031 | 0,012 … 0,051 |
| C1 → C3 | 76 | 0 | 2.6e-23 | +0,045 | 0,025 … 0,064 |
| C1 → C4 | 118 | 0 | 6.0e-36 | +0,083 | 0,061 … 0,107 |
| C1 → C4p | 60 | 0 | 1.7e-18 | +0,028 | 0,010 … 0,046 |
| C1 → C5 | 46 | 0 | 2.8e-14 | +0,005 | −0,013 … 0,024 |
| C1 → C6 | 118 | 0 | 6.0e-36 | +0,083 | 0,061 … 0,107 |

Truthfulness = Acc − Hal; Coverage = 1 − Miss; Selective risk = Hal / (Acc + Hal). Từ chối đúng =
tỷ lệ câu unanswerable được từ chối; từ chối nhầm = tỷ lệ câu answerable bị từ chối. Khoảng tin cậy
là bootstrap 2.000 lần theo câu hỏi. McNemar tính trên biến "câu này có bị bịa không"; guardrail chỉ
chặn chứ không tạo thêm câu bịa, nên khi so C1 với C2–C6 hãy đọc cột Δ Truthfulness thay vì p-value.

### Mức tách câu đúng khỏi câu bịa của từng điểm (AUC, dev)

| Điểm | Llama-3 AUC | Median câu đúng | Median câu bịa | Phi-3 AUC | Median câu đúng | Median câu bịa |
|---|---|---|---|---|---|---|
| `ctx_cos` | 0,639 | 0,628 | 0,582 | 0,645 | 0,629 | 0,582 |
| `lexical` | 0,510 | 1,000 | 1,000 | 0,605 | 1,000 | 1,000 |
| `relevance` | 0,529 | 0,404 | 0,392 | 0,532 | 0,496 | 0,481 |
| `hhem_ans_concat` | 0,542 | 0,826 | 0,802 | 0,579 | 0,895 | 0,853 |
| `hhem_ans_max` | 0,545 | 0,885 | 0,867 | 0,597 | 0,904 | 0,856 |
| `hhem_qa_concat` | 0,771 | 0,939 | 0,837 | 0,742 | 0,944 | 0,830 |
| `hhem_qa_max` | 0,817 | 0,938 | 0,742 | 0,781 | 0,933 | 0,655 |

Số câu có trả lời ở C1: Llama-3 536 đúng và 381 bịa; Phi-3 497 và 464.

### Ngưỡng đã đóng băng

| Cấu hình | Điểm | Llama-3 | Phi-3 |
|---|---|---|---|
| C2 | `ctx_cos` | 0,498 | 0,496 |
| C3 | `lexical` | 1,000 | 0,600 |
| C4 | `hhem_qa_max` | 0,705 | 0,249 |
| C4p | `hhem_ans_concat` | 0,533 | 0,389 |
| C5 | `relevance` | 0,258 | 0,281 |
| C6 | `ctx_cos` | tắt | tắt |
| C6 | `lexical` | tắt | tắt |
| C6 | `hhem_qa_max` | 0,705 | 0,249 |
| C6 | `relevance` | tắt | tắt |

### Đánh đổi của C4 khi nới trần từ chối nhầm (dev)

| Trần | Llama-3: ngưỡng | Acc | Hal | Truthfulness | Từ chối đúng | Phi-3: ngưỡng | Acc | Hal | Truthfulness | Từ chối đúng |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,02 | 0,059 | 0,522 | 0,345 | 0,177 | 0,286 | 0,011 | 0,487 | 0,447 | 0,040 | 0,130 |
| 0,05 | 0,488 | 0,506 | 0,259 | 0,248 | 0,483 | 0,053 | 0,474 | 0,410 | 0,064 | 0,204 |
| 0,10 | 0,705 | 0,483 | 0,199 | 0,284 | 0,611 | 0,249 | 0,454 | 0,339 | 0,116 | 0,350 |
| 0,15 | 0,803 | 0,457 | 0,150 | 0,308 | 0,719 | 0,500 | 0,437 | 0,267 | 0,169 | 0,492 |
| 0,20 | 0,841 | 0,433 | 0,125 | 0,308 | 0,767 | 0,651 | 0,416 | 0,228 | 0,188 | 0,565 |
| 0,30 | 0,891 | 0,379 | 0,090 | 0,289 | 0,840 | 0,833 | 0,370 | 0,142 | 0,228 | 0,739 |

### Giám khảo

Số câu answerable, có trả lời, F1 < 0,5 phải giao giám khảo, và số câu giám khảo cho đúng:

| Tập | Model | Cấu hình | Số câu chấm | Cho đúng | Tỷ lệ |
|---|---|---|---|---|---|
| dev | Llama-3-8B-Instruct | C0 | 395 | 122 | 0,309 |
| dev | Llama-3-8B-Instruct | C1 | 68 | 28 | 0,412 |
| dev | Phi-3-mini-4k-instruct | C0 | 510 | 163 | 0,320 |
| dev | Phi-3-mini-4k-instruct | C1 | 274 | 197 | 0,719 |
| test | Llama-3-8B-Instruct | C0 | 895 | 236 | 0,264 |
| test | Llama-3-8B-Instruct | C1 | 179 | 70 | 0,391 |
| test | Phi-3-mini-4k-instruct | C0 | 1107 | 320 | 0,289 |
| test | Phi-3-mini-4k-instruct | C1 | 569 | 360 | 0,633 |

Số câu giám khảo trả sai định dạng: 0.

## Giới hạn

- Giám khảo chưa được đối chứng bằng người chấm tay, và cùng họ Llama với một model bị chấm.
- Trần từ chối nhầm 0,10 do nhóm tự đặt; trên test C4 vượt nhẹ (0,132 và 0,110).
- C4 khác định nghĩa trong proposal (hypothesis có thêm câu hỏi); C4p là bản đúng proposal.
- Sinh ở temperature 0 vẫn có thể lệch vài câu giữa các lần chạy khi nhiều request chạy song
  song trên GPU; các bước còn lại cho kết quả lặp lại được.
- Một bộ dữ liệu, một seed. Luồng này chưa nối vào `KnowledgeRepository` / `LLMClient` của `src/`.
