# RAG + guardrail trên SQuAD 2.0

Luồng thực nghiệm cho đề tài *RAG as a Hallucination Guardrail for a Small LLM*: đo xem RAG và
các bước kiểm tra sau khi sinh giảm bịa đặt được bao nhiêu trên model nhỏ, và phải trả giá bao
nhiêu câu bị từ chối nhầm.

Guardrail ở đây kiểm tra **câu trả lời sau khi sinh** rồi mới quyết định trả lời hay từ chối.
Cách chấm lấy ba nhãn của CRAG (arXiv 2406.04744): câu đúng +1, câu từ chối 0, câu bịa −1.

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
| `p5_report.py` | Chỉ số bổ sung của P5: điểm +1/−4/0, risk–coverage của HHEM, LettuceDetect làm bước từ chối sau khi sinh |

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

# chỉ số bổ sung của P5; phần detector cần chạy experiments/p3_retrieval/rag_detector trước
python -m src.squad.p5_report export
for f in results/squad/detector_in/*.jsonl; do
  (cd experiments/p3_retrieval/rag_detector && python run_detect.py --scenario S4 --input ../../../$f \
     --output-dir ../../../results/squad/detector_out/$(basename $f .jsonl))
done
python -m src.squad.p5_report report
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
Số liệu dưới đây là lần chạy lại ngày 26/10/2026 trên dữ liệu P2 đã commit (`data/squad/processed`,
`build_data` tái lập giống từng byte); lệch so với lần chạy trước tối đa 0,030 Truthfulness, nằm
trong khoảng tin cậy.

Tóm tắt:

- **RAG là đóng góp lớn nhất.** Trên test, thêm tài liệu làm Truthfulness tăng +0,524 (Llama-3) và
  +0,509 (Phi-3).
- **Chỉ một check có tác dụng rõ:** HHEM với hypothesis là câu hỏi ghép câu trả lời (C4). Trên test,
  Hal của Llama-3 giảm từ 0,394 xuống 0,205 và Truthfulness tăng +0,136; với Phi-3 tăng +0,157.
- **Ba check còn lại và HHEM theo định nghĩa gốc gần như vô ích với Llama-3.** Khi dò cả bốn ngưỡng
  cùng lúc (C6), cách dò gần như chỉ giữ HHEM nên C6 ≈ C4.
- **Phi-3 có RAG vẫn bịa nhiều hơn đúng** (Truthfulness −0,065 trên test); C4 đưa nó lên 0,091.

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
| C0 không RAG | 0,211 | 0,657 | 0,132 | −0,445 | −0,477 … −0,413 | 0,868 | 0,757 | 0,175 | 0,093 |
| C1 RAG | 0,473 | 0,394 | 0,133 | 0,079 | 0,041 … 0,115 | 0,867 | 0,454 | 0,264 | 0,014 |
| C2 + context check | 0,439 | 0,331 | 0,230 | 0,108 | 0,072 … 0,142 | 0,770 | 0,430 | 0,390 | 0,084 |
| C3 + lexical | 0,467 | 0,379 | 0,154 | 0,089 | 0,052 … 0,124 | 0,846 | 0,448 | 0,291 | 0,029 |
| C4 + HHEM (câu hỏi + câu trả lời) | 0,421 | 0,205 | 0,374 | 0,215 | 0,185 … 0,245 | 0,626 | 0,328 | 0,638 | 0,132 |
| C4p + HHEM theo proposal gốc | 0,429 | 0,348 | 0,223 | 0,081 | 0,044 … 0,116 | 0,777 | 0,448 | 0,350 | 0,107 |
| C5 + relevance | 0,425 | 0,351 | 0,224 | 0,074 | 0,039 … 0,108 | 0,776 | 0,452 | 0,347 | 0,112 |
| C6 đủ bốn check | 0,420 | 0,205 | 0,375 | 0,215 | 0,184 … 0,244 | 0,625 | 0,328 | 0,636 | 0,135 |

| So sánh | Bớt câu bịa | Thêm câu bịa | McNemar p | Δ Truthfulness | 95% CI |
|---|---|---|---|---|---|
| C0 → C1 | 801 | 139 | 1.1e-113 | +0,524 | 0,487 … 0,561 |
| C1 → C2 | 158 | 0 | 5.5e-48 | +0,029 | 0,017 … 0,041 |
| C1 → C3 | 38 | 0 | 7.3e-12 | +0,010 | 0,004 … 0,015 |
| C1 → C4 | 474 | 0 | 4.1e-143 | +0,136 | 0,117 … 0,154 |
| C1 → C4p | 115 | 0 | 4.8e-35 | +0,002 | −0,010 … 0,013 |
| C1 → C5 | 108 | 0 | 6.2e-33 | −0,005 | −0,016 … 0,007 |
| C1 → C6 | 474 | 0 | 4.1e-143 | +0,135 | 0,116 … 0,153 |

**Phi-3-mini-4k-instruct (4-bit)**

| Cấu hình | Acc | Hal | Miss | Truthfulness | 95% CI | Coverage | Selective risk | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|---|
| C0 không RAG | 0,208 | 0,781 | 0,011 | −0,574 | −0,606 … −0,542 | 0,989 | 0,790 | 0,012 | 0,010 |
| C1 RAG | 0,435 | 0,500 | 0,066 | −0,065 | −0,103 … −0,028 | 0,934 | 0,535 | 0,120 | 0,016 |
| C2 + context check | 0,404 | 0,415 | 0,181 | −0,012 | −0,047 … 0,023 | 0,819 | 0,507 | 0,283 | 0,088 |
| C3 + lexical | 0,410 | 0,385 | 0,205 | 0,025 | −0,010 … 0,060 | 0,795 | 0,484 | 0,316 | 0,104 |
| C4 + HHEM (câu hỏi + câu trả lời) | 0,405 | 0,314 | 0,281 | 0,091 | 0,060 … 0,123 | 0,719 | 0,436 | 0,457 | 0,119 |
| C4p + HHEM theo proposal gốc | 0,401 | 0,392 | 0,207 | 0,009 | −0,026 … 0,044 | 0,793 | 0,494 | 0,297 | 0,124 |
| C5 + relevance | 0,391 | 0,436 | 0,173 | −0,044 | −0,079 … −0,010 | 0,827 | 0,527 | 0,230 | 0,120 |
| C6 đủ bốn check | 0,405 | 0,314 | 0,281 | 0,091 | 0,060 … 0,123 | 0,719 | 0,436 | 0,457 | 0,119 |

| So sánh | Bớt câu bịa | Thêm câu bịa | McNemar p | Δ Truthfulness | 95% CI |
|---|---|---|---|---|---|
| C0 → C1 | 769 | 60 | 1.1e-157 | +0,509 | 0,472 … 0,544 |
| C1 → C2 | 213 | 0 | 1.5e-64 | +0,054 | 0,041 … 0,066 |
| C1 → C3 | 289 | 0 | 2.0e-87 | +0,090 | 0,076 … 0,104 |
| C1 → C4 | 468 | 0 | 2.6e-141 | +0,157 | 0,139 … 0,174 |
| C1 → C4p | 271 | 0 | 5.3e-82 | +0,074 | 0,059 … 0,089 |
| C1 → C5 | 161 | 0 | 6.8e-49 | +0,021 | 0,008 … 0,033 |
| C1 → C6 | 468 | 0 | 2.6e-141 | +0,157 | 0,139 … 0,174 |

### Bảng C0–C6 trên dev (1.021 câu, tập dò ngưỡng nên lạc quan)

**Llama-3-8B-Instruct (4-bit)**

| Cấu hình | Acc | Hal | Miss | Truthfulness | 95% CI | Coverage | Selective risk | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|---|
| C0 không RAG | 0,220 | 0,629 | 0,151 | −0,408 | −0,457 … −0,359 | 0,849 | 0,740 | 0,153 | 0,149 |
| C1 RAG | 0,528 | 0,372 | 0,100 | 0,156 | 0,101 … 0,214 | 0,900 | 0,413 | 0,215 | 0,014 |
| C2 + context check | 0,480 | 0,314 | 0,206 | 0,166 | 0,116 … 0,220 | 0,794 | 0,396 | 0,348 | 0,099 |
| C3 + lexical | 0,526 | 0,361 | 0,113 | 0,165 | 0,111 … 0,223 | 0,887 | 0,407 | 0,238 | 0,019 |
| C4 + HHEM (câu hỏi + câu trả lời) | 0,484 | 0,196 | 0,320 | 0,288 | 0,243 … 0,335 | 0,680 | 0,288 | 0,616 | 0,099 |
| C4p + HHEM theo proposal gốc | 0,482 | 0,332 | 0,186 | 0,150 | 0,098 … 0,205 | 0,814 | 0,408 | 0,302 | 0,099 |
| C5 + relevance | 0,486 | 0,336 | 0,178 | 0,150 | 0,098 … 0,205 | 0,822 | 0,409 | 0,286 | 0,098 |
| C6 đủ bốn check | 0,485 | 0,196 | 0,319 | 0,289 | 0,245 … 0,336 | 0,681 | 0,288 | 0,613 | 0,099 |

| So sánh | Bớt câu bịa | Thêm câu bịa | McNemar p | Δ Truthfulness | 95% CI |
|---|---|---|---|---|---|
| C0 → C1 | 312 | 50 | 2.1e-47 | +0,564 | 0,509 … 0,620 |
| C1 → C2 | 59 | 0 | 3.5e-18 | +0,010 | −0,010 … 0,029 |
| C1 → C3 | 11 | 0 | 9.8e-04 | +0,009 | 0,002 … 0,017 |
| C1 → C4 | 180 | 0 | 1.3e-54 | +0,132 | 0,106 … 0,160 |
| C1 → C4p | 41 | 0 | 9.1e-13 | −0,006 | −0,024 … 0,012 |
| C1 → C5 | 37 | 0 | 1.5e-11 | −0,006 | −0,024 … 0,011 |
| C1 → C6 | 180 | 0 | 1.3e-54 | +0,133 | 0,108 … 0,162 |

**Phi-3-mini-4k-instruct (4-bit)**

| Cấu hình | Acc | Hal | Miss | Truthfulness | 95% CI | Coverage | Selective risk | Từ chối đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|---|
| C0 không RAG | 0,217 | 0,757 | 0,025 | −0,540 | −0,590 … −0,491 | 0,975 | 0,777 | 0,016 | 0,033 |
| C1 RAG | 0,492 | 0,450 | 0,059 | 0,042 | −0,016 … 0,102 | 0,941 | 0,478 | 0,119 | 0,014 |
| C2 + context check | 0,450 | 0,378 | 0,172 | 0,071 | 0,020 … 0,126 | 0,828 | 0,457 | 0,270 | 0,099 |
| C3 + lexical | 0,458 | 0,377 | 0,165 | 0,081 | 0,029 … 0,136 | 0,835 | 0,451 | 0,252 | 0,099 |
| C4 + HHEM (câu hỏi + câu trả lời) | 0,463 | 0,317 | 0,219 | 0,146 | 0,096 … 0,199 | 0,781 | 0,407 | 0,380 | 0,099 |
| C4p + HHEM theo proposal gốc | 0,462 | 0,388 | 0,150 | 0,074 | 0,018 … 0,131 | 0,850 | 0,456 | 0,217 | 0,099 |
| C5 + relevance | 0,449 | 0,404 | 0,148 | 0,045 | −0,011 … 0,101 | 0,852 | 0,474 | 0,213 | 0,099 |
| C6 đủ bốn check | 0,463 | 0,317 | 0,219 | 0,146 | 0,096 … 0,199 | 0,781 | 0,407 | 0,380 | 0,099 |

| So sánh | Bớt câu bịa | Thêm câu bịa | McNemar p | Δ Truthfulness | 95% CI |
|---|---|---|---|---|---|
| C0 → C1 | 339 | 25 | 1.7e-71 | +0,582 | 0,524 … 0,639 |
| C1 → C2 | 73 | 0 | 2.1e-22 | +0,029 | 0,009 … 0,050 |
| C1 → C3 | 74 | 0 | 1.1e-22 | +0,039 | 0,019 … 0,059 |
| C1 → C4 | 135 | 0 | 4.6e-41 | +0,104 | 0,079 … 0,128 |
| C1 → C4p | 63 | 0 | 2.2e-19 | +0,032 | 0,013 … 0,051 |
| C1 → C5 | 47 | 0 | 1.4e-14 | +0,003 | −0,015 … 0,021 |
| C1 → C6 | 135 | 0 | 4.6e-41 | +0,104 | 0,079 … 0,128 |

Truthfulness = Acc − Hal; Coverage = 1 − Miss; Selective risk = Hal / (Acc + Hal). Từ chối đúng =
tỷ lệ câu unanswerable được từ chối; từ chối nhầm = tỷ lệ câu answerable bị từ chối. Khoảng tin cậy
là bootstrap 2.000 lần theo câu hỏi. McNemar tính trên biến "câu này có bị bịa không"; guardrail chỉ
chặn chứ không tạo thêm câu bịa, nên khi so C1 với C2–C6 hãy đọc cột Δ Truthfulness thay vì p-value.

### Mức tách câu đúng khỏi câu bịa của từng điểm (AUC, dev)

| Điểm | Llama-3 AUC | Median câu đúng | Median câu bịa | Phi-3 AUC | Median câu đúng | Median câu bịa |
|---|---|---|---|---|---|---|
| `ctx_cos` | 0,638 | 0,628 | 0,582 | 0,642 | 0,629 | 0,582 |
| `lexical` | 0,513 | 1,000 | 1,000 | 0,588 | 1,000 | 1,000 |
| `relevance` | 0,532 | 0,405 | 0,392 | 0,542 | 0,496 | 0,478 |
| `hhem_ans_concat` | 0,548 | 0,828 | 0,801 | 0,584 | 0,892 | 0,840 |
| `hhem_ans_max` | 0,552 | 0,885 | 0,867 | 0,607 | 0,904 | 0,843 |
| `hhem_qa_concat` | 0,774 | 0,939 | 0,832 | 0,745 | 0,943 | 0,824 |
| `hhem_qa_max` | 0,818 | 0,938 | 0,740 | 0,796 | 0,933 | 0,627 |

Số câu có trả lời ở C1: Llama-3 539 đúng và 380 bịa; Phi-3 502 và 459.

### Ngưỡng đã đóng băng

| Cấu hình | Điểm | Llama-3 | Phi-3 |
|---|---|---|---|
| C2 | `ctx_cos` | 0,498 | 0,499 |
| C3 | `lexical` | 1,000 | 0,600 |
| C4 | `hhem_qa_max` | 0,715 | 0,291 |
| C4p | `hhem_ans_concat` | 0,533 | 0,438 |
| C5 | `relevance` | 0,259 | 0,287 |
| C6 | `ctx_cos` | tắt | tắt |
| C6 | `lexical` | 0,667 | tắt |
| C6 | `hhem_qa_max` | 0,705 | 0,291 |
| C6 | `relevance` | tắt | tắt |

### Đánh đổi của C4 khi nới trần từ chối nhầm (dev)

| Trần | Llama-3: ngưỡng | Acc | Hal | Truthfulness | Từ chối đúng | Phi-3: ngưỡng | Acc | Hal | Truthfulness | Từ chối đúng |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,02 | 0,052 | 0,525 | 0,346 | 0,179 | 0,277 | 0,015 | 0,490 | 0,438 | 0,052 | 0,144 |
| 0,05 | 0,488 | 0,509 | 0,256 | 0,254 | 0,483 | 0,054 | 0,480 | 0,405 | 0,075 | 0,204 |
| 0,10 | 0,715 | 0,484 | 0,196 | 0,288 | 0,616 | 0,291 | 0,463 | 0,317 | 0,146 | 0,380 |
| 0,15 | 0,804 | 0,458 | 0,150 | 0,309 | 0,716 | 0,502 | 0,443 | 0,259 | 0,184 | 0,499 |
| 0,20 | 0,842 | 0,435 | 0,125 | 0,310 | 0,764 | 0,651 | 0,420 | 0,214 | 0,207 | 0,590 |
| 0,30 | 0,891 | 0,380 | 0,091 | 0,289 | 0,835 | 0,833 | 0,370 | 0,135 | 0,235 | 0,755 |

### Giám khảo

Số câu answerable, có trả lời, F1 < 0,5 phải giao giám khảo, và số câu giám khảo cho đúng:

| Tập | Model | Cấu hình | Số câu chấm | Cho đúng | Tỷ lệ |
|---|---|---|---|---|---|
| dev | Llama-3-8B-Instruct | C0 | 397 | 125 | 0,315 |
| dev | Llama-3-8B-Instruct | C1 | 66 | 29 | 0,439 |
| dev | Phi-3-mini-4k-instruct | C0 | 508 | 165 | 0,325 |
| dev | Phi-3-mini-4k-instruct | C1 | 266 | 192 | 0,722 |
| test | Llama-3-8B-Instruct | C0 | 892 | 232 | 0,260 |
| test | Llama-3-8B-Instruct | C1 | 177 | 71 | 0,401 |
| test | Phi-3-mini-4k-instruct | C0 | 1089 | 311 | 0,286 |
| test | Phi-3-mini-4k-instruct | C1 | 546 | 347 | 0,636 |

Số câu giám khảo trả sai định dạng: 0.

### Chỉ số bổ sung của P5 (`p5_report.py`, test)

Tính lại trên nhãn đã có, không gọi model sinh. Điểm +1/−4/0 phạt câu bịa nặng gấp bốn (đơn vị %).
`C4_p4` dò lại ngưỡng HHEM trên dev để tối đa điểm +1/−4/0. Detector là LettuceDetect
(`experiments/p3_retrieval/rag_detector`) dùng làm bước từ chối sau khi sinh, τ dò trên dev với
cùng trần từ chối nhầm 0,10.

| Model | Cấu hình | Acc | Hal | Miss | Truthfulness | +1/−4/0 | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Llama-3-8B | C0 | 21,1 | 65,7 | 13,2 | −44,5 | −241,6 | 0,093 |
| Llama-3-8B | C1 | 47,3 | 39,4 | 13,3 | 7,9 | −110,2 | 0,014 |
| Llama-3-8B | C4 | 42,1 | 20,5 | 37,4 | 21,5 | −40,1 | 0,132 |
| Llama-3-8B | C4_p4 | 18,3 | 5,8 | 75,8 | 12,5 | −5,0 | 0,633 |
| Llama-3-8B | C1+detector | 46,2 | 38,1 | 15,7 | 8,1 | −106,4 | 0,038 |
| Llama-3-8B | C4+detector | 41,3 | 20,0 | 38,7 | 21,3 | −38,6 | 0,149 |
| Phi-3 | C0 | 20,8 | 78,1 | 1,1 | −57,4 | −291,8 | 0,010 |
| Phi-3 | C1 | 43,5 | 50,0 | 6,6 | −6,5 | −156,5 | 0,016 |
| Phi-3 | C4 | 40,5 | 31,4 | 28,1 | 9,1 | −85,0 | 0,119 |
| Phi-3 | C4_p4 | 17,0 | 5,8 | 77,1 | 11,2 | −6,3 | 0,648 |
| Phi-3 | C1+detector | 41,6 | 44,9 | 13,5 | −3,3 | −138,0 | 0,075 |
| Phi-3 | C4+detector | 39,4 | 30,3 | 30,3 | 9,1 | −81,7 | 0,148 |

AUROC tách câu bịa khỏi câu đúng trên test (Llama-3 / Phi-3): HHEM 0,749 / 0,775, LettuceDetect
0,504 / 0,529. Detector gần như ngẫu nhiên: phần lớn câu bịa nằm ở câu unanswerable, và câu trả lời
là một cụm có thật trong ngữ cảnh — trung thành với ngữ cảnh nên detector không gắn cờ.
Với luật +1/−4/0, mọi cấu hình đều âm; muốn tiến gần 0 phải từ chối khoảng 3/4 số câu.

## Giới hạn

- Giám khảo chưa được đối chứng bằng người chấm tay, và cùng họ Llama với một model bị chấm.
- Trần từ chối nhầm 0,10 do nhóm tự đặt; trên test C4 vượt nhẹ (0,132 và 0,119).
- C4 khác định nghĩa trong proposal (hypothesis có thêm câu hỏi); C4p là bản đúng proposal.
- Sinh ở temperature 0 vẫn có thể lệch vài câu giữa các lần chạy khi nhiều request chạy song
  song trên GPU; các bước còn lại cho kết quả lặp lại được.
- Một bộ dữ liệu, một seed. Luồng này chưa nối vào `KnowledgeRepository` / `LLMClient` của `src/`.
