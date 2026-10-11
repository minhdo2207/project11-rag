# S4 — Lọc phạm vi và phiên bản

S4 nhắm vào loại hard negative mà **không retriever nào phân biệt được bằng ngữ nghĩa**: chunk đúng chủ đề nhưng sai phiên bản. Chính sách 2025 và 2026 gần như giống hệt nhau về từ ngữ, chỉ khác vài con số. Với dense similarity, bản cũ không phải kết quả tệ — nó là kết quả rất tốt cho một vũ trụ khác.

## Nội dung gói

| File | Vai trò | Trạng thái |
|---|---|---|
| `metadata.py` | Schema phạm vi, trích xuất hiệu lực, biên dịch bộ lọc | **Mới ở S4** |
| `run_s4.py` | Ablation 5 biến thể, đo `stale_rate` | **Mới ở S4** |
| `demo_corpus.py` | Corpus có cặp bản cũ/bản hiện hành | **Mới ở S4** |
| `retriever.py` | + tham số `where`, `filter_mode` (pre/post/off) | **Sửa ở S4** |
| `rerank.py` | Cross-encoder | Giữ nguyên từ S3 |
| `chunking.py` | Chunking theo cấu trúc | Giữ nguyên từ S2 |
| `embedding.py` | Dense encoder | Giữ nguyên từ S1 |
| `requirements.txt` · `README.md` | — | — |

## Chỉ số của kịch bản này: `stale_rate@k`

Tỷ lệ truy vấn có chunk **đã hết hiệu lực** lọt vào ngữ cảnh. Recall và precision đều mù trước nó — chunk cũ đúng chủ đề, viết chuẩn, khớp ngữ nghĩa tốt.

Đây là lỗi truy hồi nguy hiểm nhất trong hệ hỏi đáp người dùng cuối, vì câu trả lời sinh ra trôi chảy, cụ thể, tự tin và **nhìn y hệt một câu trả lời đúng**.

Kèm theo là `stale_outranks_current`: tỷ lệ truy vấn mà bản cũ xếp **trên** bản hiện hành — trường hợp xấu nhất.

## Chạy

```bash
pip install -r requirements.txt
python run_s4.py --demo
python run_s4.py --docs corpus.jsonl --eval eval.jsonl --as-of 2026-10-07 --audience retail
```

```jsonl
corpus.jsonl  {"source": "...", "text": "...", "metadata": {"status": "current", "effective_from": "2026-01-01"}}
eval.jsonl    {"query": "...", "answer_spans": ["bản đúng"], "stale_spans": ["bản cũ"]}
```

## Kết quả demo

```
variant                     rec@5   stale@5  stale>cur  returned   short
V0 no filter                1.000     1.000      0.571     5.000   0.000
V1 no filter + rerank       1.000     1.000      0.571     5.000   0.000
V2 post-filter              1.000     0.000      0.000     2.000   3.000
V3 pre-filter               1.000     0.000      0.000     5.000   0.000
V4 pre-filter + rerank      1.000     0.000      0.000     5.000   0.000
```

Ba điều đọc được, và cả ba đều quan trọng:

**Recall không phát hiện được vấn đề.** `rec@5` bằng 1.000 ở mọi biến thể — cả khi 100% truy vấn có bản cũ trong ngữ cảnh và 57% truy vấn bản cũ xếp trên bản mới. Nếu chỉ đo recall, bạn sẽ kết luận hệ thống hoàn hảo.

**Rerank không cứu được.** V1 giữ nguyên `stale@5` = 1.000. Cross-encoder không có cách nào biết một chunk đã hết hiệu lực — thông tin đó không nằm trong văn bản. Phạm vi là bài toán metadata và chỉ metadata giải được.

**Post-filter âm thầm làm hỏng top-k.** V2 trả về trung bình 2 chunk thay vì 5, thiếu 3, và không có gì trong phản hồi báo điều đó. Caller yêu cầu k và nhận ít hơn mà không biết.

## Vì sao lọc phải chạy trước khi chấm điểm

| `filter_mode` | Cách làm | Hệ quả |
|---|---|---|
| `"pre"` | Thu hẹp tập ứng viên rồi mới chấm điểm | Đúng — top-k vẫn là top-k, lấy từ đúng vũ trụ |
| `"post"` | Chấm điểm rồi loại kết quả không khớp | Sai — top-k co lại âm thầm |
| `"off"` | Không lọc | Baseline để so |

`filter_mode="post"` tồn tại trong code **chỉ để chứng minh nó sai**, không phải để dùng.

Khi bộ lọc không khớp tài liệu nào, retriever trả về rỗng kèm `empty_reason` thay vì rơi về tìm kiếm toàn cục. Fallback âm thầm chính là cách hệ thống trả lời câu hỏi 2026 bằng quy định 2025.

## Bộ lọc

```python
from metadata import as_of_filter, compile_filter

scope = as_of_filter()                                   # quy định đang có hiệu lực hôm nay
scope = as_of_filter("2026-01-15")                       # những gì áp dụng vào ngày đó
scope = as_of_filter(audience="enterprise")              # kèm phân khúc

r.retrieve(q, where=scope, filter_mode="pre")
r.retrieve(q, where={"doc_type": ["faq", "policy"]})     # membership
r.retrieve(q, where={"effective_to": {"exists": False}}) # còn hiệu lực
```

Hai quyết định thiết kế trong `metadata.py` đáng nêu:

**Toán tử lạ thì raise, không bỏ qua.** Một bộ lọc bị phớt lờ âm thầm tệ hơn không lọc, vì hệ thống trông như đã lọc mà thực ra chưa.

**Chunk thiếu giá trị trường bị loại, không được cho qua.** Không chứng minh được là trong phạm vi thì coi như ngoài phạm vi. Thà bỏ sót một chunk còn hơn phục vụ một chunk không rõ phạm vi.

## Kiểm tra độ phủ metadata trước khi tin bộ lọc

`run_s4.py` in báo cáo độ phủ từng trường và **dừng với mã lỗi 2** nếu `status` không phủ 100%. Chunk không có `status` thì không loại ra được — bộ lọc sẽ rò.

```
metadata coverage
  status            100%
  effective_from    100%
  effective_to       46%
  audience          100%
```

Cộng thêm `coverage_report` cho biết bộ lọc giữ lại bao nhiêu phần trăm index. Giữ dưới 5% thì hệ thống sẽ chủ yếu trả về rỗng — biết trước tốt hơn phát hiện sau khi lên production.

## Giới hạn đã biết của S4

Bộ lọc giới hạn **tài liệu nào đủ điều kiện**; nó không phát hiện được rằng **câu hỏi** thuộc phạm vi khác. Hỏi một câu về khách hàng doanh nghiệp trong khi bộ lọc đặt ở `retail` thì retriever vẫn vui vẻ trả về chunk retail — chúng khớp, chúng trong phạm vi, chỉ là sai phân khúc.

Demo cho thấy đúng điều này: 0/2 câu ngoài phạm vi được từ chối. Bịt khoảng trống đó cần nhận diện phạm vi ở phía truy vấn (S5) hoặc ngưỡng tin cậy bắt được mức khớp yếu (S7). S4 chỉ từ chối khi bộ lọc không khớp gì cả.

## Quan hệ với S3

Nên chốt S3 và S4 **cùng nhau**. Rerank có giá trị cao hơn hẳn khi tập ứng viên không còn chứa các chunk sai phiên bản trông rất hợp lý — và ngược lại, một kết luận "rerank không hiệu quả" rút ra từ tập ứng viên bẩn là kết luận không đáng tin.

## Điều kiện để sang S5

- `status` phủ 100% chunk, `effective_from` phủ phần lớn.
- `stale_rate@5` = 0 ở chế độ pre-filter.
- Bộ lọc giữ lại đủ corpus để không gây từ chối hàng loạt.
- Đã ghi nhận khoảng trống "câu hỏi ngoài phạm vi" để S5/S7 xử lý.
