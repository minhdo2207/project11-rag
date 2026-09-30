# RAG và hiện tượng bịa đặt — bậc thang thực nghiệm trên CRAG

Bài tập lớn **Project 11** môn *Artificial Intelligence for Software Engineering*, HUST 2026–2027.
Đề bài: *Retrieval Augmented Generation (RAG): Concepts and Applications* — khái niệm cơ bản
về RAG, và dùng RAG để giảm bịa đặt.

Nhóm đo trên **CRAG** (Comprehensive RAG Benchmark, NeurIPS 2024 D&B, arXiv 2406.04744),
tập public test **1.335 câu hỏi**, model sinh **llama3:8b-instruct-fp16** chạy cục bộ qua Ollama.

---

## Câu hỏi nghiên cứu

> Thêm tài liệu vào prompt có làm model bớt bịa không, và **thành phần nào** của đường ống RAG
> thực sự mang lại cải thiện?

Để trả lời, mỗi cấu hình chỉ thay **đúng một thứ** so với cấu hình ngay trước nó. Ngân sách
tài liệu giữ nguyên **4.000 token** ở mọi bậc, nên thứ duy nhất thay đổi là *nội dung* lấp
đầy ngân sách đó.

| Bậc | Thay đổi so với bậc trước | Nguồn |
|---|---|---|
| **B0** | không đưa tài liệu | dòng *LLM only* của bài báo |
| **B1** | nối toàn văn 5 trang theo thứ tự gốc | biến thể của nhóm, làm đối chứng |
| **B1.1** | đổi toàn văn → **snippet** | **Task 1** của bài, tái lập Phụ lục A.3.1 |
| **B1.2** | 5 trang → **50 trang** | Task 3 của bài, bỏ phần mock KG |
| **B2** | đổi cách chọn: chia đoạn + BM25 + bge-m3 + **RRF** | nhóm tự làm |
| **B3** | thêm **cross-encoder** xếp lại + cắt ngưỡng 0,05 | nhóm tự làm |
| **B4** | thêm **ràng buộc trích dẫn** `[n]` | nhóm tự làm |
| **B5** | thêm **cổng từ chối** ở mức hệ thống | nhóm tự làm |

---

## Chạy lại

### Yêu cầu

- Python 3.10+, GPU NVIDIA ≥ 16 GB cho model sinh
- Thêm một GPU nữa cho giám khảo thì chạy được sinh và chấm song song — hai script trong
  `scripts/` dựng hai instance Ollama riêng (cổng 11435 sinh, 11436 chấm). Một GPU vẫn chạy
  được, chỉ là tuần tự và lâu gấp đôi.
- [Ollama](https://ollama.com) với `llama3:8b-instruct-fp16` và một model giám khảo **khác họ**
  với model bị chấm, để tránh thiên vị tự khen

```bash
pip install -r requirements.txt
ollama pull llama3:8b-instruct-fp16
ollama pull gemma3:27b            # hoặc model giám khảo khác họ tuỳ máy

# mỗi instance phải thấy được kho model của hệ thống
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11435 ollama serve &   # sinh
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11436 ollama serve &   # chấm
```

### Dữ liệu

CRAG tải công khai từ kho chính thức của Meta, **không cần đăng ký hay đăng nhập**:

```bash
python scripts/download_data.py crag           # 739 MB — Task 1 & 2 (5 trang/câu)
python scripts/download_data.py crag_task3     # 7,7 GB — Task 3, chỉ cần nếu chạy B1.2

python -m src.crag.ingest                      # → data/crag_public_test/questions.jsonl
python -m src.crag.ingest_task3                # → data/crag_task3_public_test/ (cho B1.2)
```

Script tải thẳng từ `github.com/facebookresearch/CRAG`, **kiểm dung lượng từng byte** sau
khi tải và tải tiếp được nếu đứt giữa chừng (`curl -C -`). Task 3 nằm trong bốn phần
`.tar.bz2.part0..3`; bước nạp đọc luồng thẳng qua chúng, không giải nén ra đĩa.

### Luồng chạy

```
                          data/raw/crag/*.bz2
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
              ingest.py                    ingest_task3.py
             5 trang/câu                      50 trang/câu
                  │                               │
                  ▼                               ▼
      crag_public_test/               crag_task3_public_test/
        questions.jsonl                   questions.jsonl
      (1.335 câu, split==1)                      │  chỉ B1.2 dùng
                  │                              │
       ┌──────────┴──────────┐                   │
       │                     ▼                   │
       │              retrieve.py  ── chạy MỘT lần, dùng chung cho B2–B5
       │              ├ chia đoạn 220 từ, chồng lấn 40
       │              ├ BM25 (trùng từ khoá)  ──┐
       │              ├ bge-m3 (gần ngữ nghĩa) ─┴→ RRF(k=60) → top-40
       │              └ cross-encoder bge-reranker-v2-m3 → rerank ∈ [0,1]
       │                     │
       │                     ▼
       │              retrieval.jsonl ──→ calibrate.py ──→ ngưỡng 0,05
       │                     │                                   │
       ▼                     ▼                                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                          generate.py                             │
  │   B0     không đưa tài liệu                                      │
  │   B1.x   nối trang theo THỨ TỰ GỐC tới khi đầy 4.000 token       │
  │   B2     đoạn xếp theo rrf                                       │
  │   B3     đoạn xếp theo rerank, bỏ đoạn dưới ngưỡng               │
  │   B4     như B3 + tài liệu đánh số [n], prompt buộc dẫn nguồn    │
  │   B5     như B4 + cổng: đoạn tốt nhất dưới ngưỡng → I don't know │
  └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                results/crag/gen_<bậc>_<model>_...jsonl
                              │
                              ▼  trim_75 — cắt còn 75 token đầu
                 ┌────────────┴────────────┐
                 ▼                         ▼
         score.py --mode crag      score.py --mode three
      ┌──────────────────────┐   ┌──────────────────────────┐
      │ chứa "i don't know"? │   │ khớp chuỗi đáp án?       │
      │      → missing       │   │      → accurate          │
      │ khớp chuỗi đáp án?   │   │ còn lại → giám khảo      │
      │      → accurate      │   │   chọn 1 trong 3 nhãn    │
      │ còn lại → giám khảo  │   └──────────────────────────┘
      │   True/False         │
      └──────────────────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                          report.py
                              ▼
                   results/crag/metrics.json
```

Hai nhánh chấm dùng chung **một** file sinh: sinh tốn GPU, chấm thì rẻ, nên tách ra để đổi
giám khảo hay đổi định nghĩa nhãn mà không phải gọi lại model.

### Đường ống

```bash
# 1. truy hồi một lần, dùng chung cho B2–B5  (~12 phút, 1.335 câu)
python -m src.crag.retrieve --top-k 40 --rerank-k 40

# 2. chọn ngưỡng từ dữ liệu, không chỉnh theo điểm
python -m src.crag.calibrate

# 3. sinh câu trả lời cho một bậc
python -m src.crag.generate --config B4 --model llama3:8b-instruct-fp16 \
    --ref-source snippet --num-predict 75 --keep-threshold 0.05 --gate-threshold 0.05

# 4. chấm bằng hai bộ độc lập
python -m src.crag.score --gen results/crag/gen_B4_*.jsonl --mode crag  --judge gemma3:27b
python -m src.crag.score --gen results/crag/gen_B4_*.jsonl --mode three --judge gemma3:27b

# 5. tổng hợp thành results/crag/metrics.json
python -m src.crag.report
```

Chạy cả bậc thang một lượt: `scripts/run_all_p75.sh` (sáu bậc đầu) và `scripts/run_b45.sh`
(hai bậc cuối). Cả hai đều **chạy tiếp được** sau khi bị ngắt giữa chừng.

---

## Cấu trúc

```
src/crag/
  ingest.py        đọc bản phát hành CRAG → questions.jsonl (lọc split==1)
  ingest_task3.py  bản 50 trang, lọc theo interaction_id của task 1&2
  retrieve.py      chia đoạn 220 từ · BM25 tự cài · bge-m3 · RRF(k=60) · cross-encoder
  calibrate.py     chọn ngưỡng bằng cách đo đáp án có nằm trong đoạn không
  generate.py      tám cấu hình B0–B5, prompt nguyên văn Phụ lục A.3.1
  score.py         hai bộ chấm độc lập: kiểu bài (nhị phân) và đối chứng ba nhãn
  report.py        chạy cả bậc thang → metrics.json
src/evaluation.py  gộp nhãn thành chỉ số: accurate/hallucinated/missing/truthfulness
tests/             pytest — chạy bằng `make test`
scripts/           driver chạy cả bậc thang, tải dữ liệu, soi dữ liệu
```

