# rag_detector

Khâu **detector hallucination** (dùng [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)) cho 7 kịch bản RAG S1–S7. Detector chạy sau generator và trước evaluation. **Một detector, một cấu hình, dùng chung cho cả 7 kịch bản**, nên chênh lệch metric chỉ đến từ chunking và retrieval.

| Kịch bản | Thành phần thêm |
|---|---|
| S1 | dense + BM25 + RRF |
| S2 | chunk theo cấu trúc, khối nguyên tử, làm giàu tiêu đề |
| S3 | cross-encoder rerank (replace / RRF fuse) |
| S4 | lọc trước theo metadata (phạm vi, phiên bản) |
| S5 | hiểu truy vấn (gấp dấu, từ điển, đa biến thể) |
| S6 | adaptive-k, MMR, sắp xếp U-shape |
| S7 | hiệu chuẩn và quyết định trả lời 4 cấp (`may_assert`) |

Gói này dùng thư viện LettuceDetect **cài từ PyPI** (`pip install lettucedetect`, bản 0.2.3), không nhúng mã nguồn LettuceDetect vào repo.

## Cấu trúc

```
experiments/p3_retrieval/
  s1/ … s7/                 bảy kịch bản retrieval (mỗi gói có run_sN.py riêng)
  rag_detector/             khâu detector cho S1–S7 (thư mục này)
    README.md
    requirements.txt
    run_detect.py           CLI: chạy một kịch bản (--scenario) hoặc cả bảy (--all)
    detection.py            tải detector, chạy LettuceDetect, xử lý câu bị từ chối
    metrics.py              accuracy, hallucination rate, rejection rate, adjusted accuracy, quét tau
    data/                   đầu vào s1.jsonl … s7.jsonl (tự tạo, không commit)
    results/                đầu ra (tự tạo, không commit)
    examples/generations_example.jsonl    dữ liệu mẫu 3 câu
```

## Cài đặt

```bash
cd experiments/p3_retrieval/rag_detector
pip install -r requirements.txt     # lettucedetect và các thư viện nó kéo theo (torch, transformers, ...)
```

Model mặc định `KRLabsOrg/lettucedect-v2-mmbert-base` được tải từ Hugging Face ở lần chạy đầu. Nếu dùng model EuroBERT thì cần `transformers>=4.48.3,<5`. Các script dùng import phẳng (`from detection import ...`); chạy `python run_detect.py ...` trực tiếp, thư mục hiện tại nào cũng được.

## Dữ liệu đầu vào

Mỗi kịch bản là một file JSONL, mặc định `data/s1.jsonl` … `data/s7.jsonl`. Mỗi dòng là một câu hỏi:

| Trường | Bắt buộc | Ý nghĩa |
|---|---|---|
| `id` | có | mã câu hỏi |
| `question` | có | câu hỏi gốc của người dùng |
| `contexts` | có | danh sách chunk **đúng như đã đưa cho LLM**, sau rerank, lọc, MMR và sắp xếp của kịch bản đó |
| `answer` | có | câu trả lời cuối của LLM (có thể rỗng nếu bị từ chối) |
| `correct` | không | `true`/`false`, đúng sai so với gold. Cần cho `accuracy` và `adjusted_accuracy` |
| `refused` / `decision` | không | `decision` thuộc `abstain`, `refuse`, `no_answer` hoặc `refused: true` thì tính là từ chối, detector không chạy. Nếu không có, câu trả lời rỗng hoặc chứa cụm từ chối thông dụng cũng được coi là từ chối |

Xem mẫu ở [examples/generations_example.jsonl](examples/generations_example.jsonl). Thư mục `data/` không có sẵn trong repo: tạo nó và đặt `s1.jsonl` … `s7.jsonl` vào (cả `data/` lẫn `results/` đã nằm trong `.gitignore`).

## Chạy

```bash
# thử nhanh với dữ liệu mẫu
python run_detect.py --scenario S1 --input examples/generations_example.jsonl

# một kịch bản (đọc data/s3.jsonl)
python run_detect.py --scenario S3

# một kịch bản, input ở nơi khác
python run_detect.py --scenario S3 --input /path/to/s3_generations.jsonl

# cả bảy kịch bản, tải model một lần, in bảng so sánh
python run_detect.py --all
python run_detect.py --all --input-dir /path/to/outputs --tau 0.7 --resume
```

`--all` bỏ qua kịch bản nào chưa có file input và báo tên các kịch bản bị bỏ qua.

### Đầu ra (trong `results/`)

- `SN_results.jsonl`: span, confidence cao nhất và trạng thái từ chối của từng câu.
- `SN_summary.json`: metric tại `--tau` và bảng quét `--taus`.
- `comparison.csv` (chỉ với `--all`): metric của mọi kịch bản ở mọi giá trị tau.

### Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--scenario` / `--all` | bắt buộc chọn một | một kịch bản S1–S7, hoặc cả bảy |
| `--input` | | file input, chỉ dùng với `--scenario` |
| `--input-dir` | `data/` | thư mục chứa `s1.jsonl` … `s7.jsonl` |
| `--output-dir` | `results/` | thư mục kết quả |
| `--method` | `transformer` | `transformer` hoặc `llm` (cần `OPENAI_API_KEY`) |
| `--model` | `KRLabsOrg/lettucedect-v2-mmbert-base` | model Hugging Face, hoặc tên LLM nếu `--method llm` |
| `--tau` | 0.5 | câu trả lời bị coi là hallucinated nếu có span nào có confidence ≥ tau |
| `--taus` | 0.5 0.7 0.9 | các ngưỡng để quét |
| `--max-length` | 4096 | độ dài tối đa của detector |
| `--lang` | `en` | ngôn ngữ prompt của detector |
| `--device` | tự chọn | `cuda` hoặc `cpu` |
| `--resume` | tắt | dùng lại kết quả của các `id` đã có trong file kết quả |

## Định nghĩa metric

Theo bài báo tham chiếu (Eq. 11). N là tổng số câu hỏi, các metric tính theo phần trăm. Code nằm ở [metrics.py](metrics.py).

Mỗi câu hỏi rơi vào đúng một nhóm:

- **rejected**: hệ thống từ chối vì context truy xuất không đủ (S7 abstain hoặc câu trả lời từ chối). Detector không chạy trên nhóm này.
- **hallucinated**: có câu trả lời và detector gắn cờ ít nhất một span có confidence ≥ `--tau`.
- **correct**: có câu trả lời, không bị gắn cờ và khớp gold (`correct: true`).
- Câu không bị gắn cờ nhưng sai so với gold được báo riêng ở `incorrect_unflagged`.

| Metric | Công thức | Tốt hơn khi |
|---|---|---|
| `accuracy` | correct / N × 100 | cao hơn |
| `hallucination_rate` | hallucinated / N × 100 | thấp hơn |
| `rejection_rate` | rejected / N × 100 | thấp hơn |
| `adjusted_accuracy` | correct / (correct + hallucinated) × 100 | cao hơn |

`adjusted_accuracy` loại các ca bị từ chối vì thiếu context, nên chỉ đo chất lượng ở các ca mà hệ thống đã cố trả lời.

Giả định cần xác nhận: mẫu số của accuracy và hallucination rate là **tất cả câu hỏi (kể cả câu bị từ chối)**. Nếu bài báo chỉ tính trên các câu đã được sinh, đổi `n` thành số câu đã trả lời trong `metrics_at`.

## Lưu ý trước khi tin vào số liệu

- **Ngôn ngữ.** Repo LettuceDetect không nêu model hỗ trợ tiếng Việt. Hãy gán nhãn tay khoảng 50–100 câu trả lời của bạn để đo precision và recall của detector, và so sánh với `--method llm`.
- **Ý nghĩa của hallucination rate.** Đây là mức không trung thành với context được đưa vào, không phải đúng sai so với gold. Nếu retriever lấy sai tài liệu mà LLM bám theo, câu đó không bị gắn cờ nhưng `correct` vẫn là `false`. Hãy xem `incorrect_unflagged` cạnh hallucination rate.
- **Context dài** (ví dụ k lớn ở S6) vượt `--max-length` sẽ bị chia chunk và lấy max xác suất. Cách này bảo thủ nên có thể làm tăng rate một cách giả tạo.
- **Detector chỉ đo hallucination**, không làm giảm nó.
- Detector LLM không luôn trả `confidence`. Khi thiếu, code coi confidence là 1.0.

## Thay detector khác

Chỉ cần sửa `build_detector` và `detect_records` trong `detection.py`. `run_detect.py` và `metrics.py` không phụ thuộc LettuceDetect.

## Quan hệ với phần còn lại của repo

- **Khác định nghĩa với `src/evaluation.py`.** Ở đây `hallucinated` nghĩa là LettuceDetect gắn cờ một span không được ngữ cảnh hỗ trợ (độ trung thành với context). Trong `three_label_report`, `hallucinated` nghĩa là câu trả lời sai so với đáp án chuẩn. Hai số này không so trực tiếp được.
- **Chưa thay `pipeline.verify`.** Đây là bước đánh giá hàng loạt sau khi sinh, không phải cổng chạy lúc detector trả lời. `pipeline.verify` trong `config.yaml` vẫn chưa được hiện thực.
- **Chưa có khâu tạo đầu vào.** Các runner `s1` … `s7` chỉ làm retrieval; chưa có bước nào sinh câu trả lời của LLM trên các chunk đó để thành `data/sN.jsonl`.
- **Chưa nối vào `src/`.** Giống phần còn lại của `experiments/p3_retrieval`, gói này không đi qua `KnowledgeRepository`.

## Đã kiểm tra gì

`tests/test_p3_rag_detector.py` kiểm tra luồng đọc dữ liệu, nhận diện câu từ chối, công thức metric và CLI bằng một detector giả. **Chưa chạy với model LettuceDetect thật** (cần `torch`, `transformers` và tải model từ Hugging Face).
