# S5 — Query understanding

S1–S4 đều sửa phía corpus. S5 là kịch bản đầu tiên làm việc ở **phía đầu vào** — nơi hệ hỏi đáp người dùng cuối mất nhiều điểm nhất: index đúng, corpus sạch, mà retrieval vẫn trượt vì câu hỏi đến dưới dạng `lam sao doi mat khau ko`.

## Nội dung gói

| File | Vai trò | Trạng thái |
|---|---|---|
| `query_understanding.py` | Chuẩn hoá, fold dấu, từ điển viết tắt, viết lại, tách câu | **Mới ở S5** |
| `run_s5.py` | Ablation 5 biến thể, chấm theo **nhóm truy vấn** | **Mới ở S5** |
| `demo_corpus.py` | FAQ sạch + 4 nhóm truy vấn người dùng thật | **Mới ở S5** |
| `retriever.py` | + fold dấu phía index, + hợp nhất đa biến thể | **Sửa ở S5** |
| `metadata.py` | Lọc phạm vi | Giữ nguyên từ S4 |
| `rerank.py` · `chunking.py` · `embedding.py` | — | Giữ nguyên |

## Một lỗi tìm được khi làm S5 — ảnh hưởng mọi kịch bản

Tokenizer kế thừa từ S1 dùng regex `[^0-9A-Za-z_]+`, vốn viết cho định danh code ASCII. Trên tiếng Việt nó coi **mọi ký tự có dấu là ký tự ngăn cách**:

```
tokenize("mật khẩu")  ->  ['m', 't', 'kh', 'u']        # trước khi sửa
tokenize("mật khẩu")  ->  ['mật', 'mat', 'khẩu', 'khau']  # sau khi sửa
```

Nghĩa là nhánh BM25 đã **hỏng âm thầm trên toàn bộ corpus tiếng Việt** từ S1. Không crash, không cảnh báo, chỉ là lexical branch gần như không đóng góp gì. Đã sửa thành `\W+` với `re.UNICODE` và **backport sang cả S1–S4** — nếu bạn đã chạy các kịch bản trước, hãy chạy lại, số đo cũ không dùng được.

Bài học rộng hơn: baseline kế thừa từ miền khác mang theo giả định của miền đó. Đáng chạy `tokenize()` trên vài câu tiếng Việt thật trước khi tin bất kỳ con số BM25 nào.

## Bốn cơ chế

| Cơ chế | Làm gì | Ghi chú |
|---|---|---|
| Chuẩn hoá | NFC, lowercase, gộp khoảng trắng, bỏ dấu câu lạ | NFC bắt buộc — tiếng Việt tồn tại ở cả dạng tổ hợp và dựng sẵn |
| **Fold dấu** | `mật khẩu` → cũng index thành `mat khau` | Giá trị cao nhất ở đây; **thay đổi phía index**, cần rebuild |
| Từ điển | `mk`→mật khẩu, `ck`→chuyển khoản, `cắt mạng`→tạm ngưng dịch vụ | Chỉ *thêm* từ, không thay thế |
| Viết lại / tách câu | Tách câu hỏi nhiều ý thành truy vấn con | Chỉ kích hoạt khi phát hiện nhiều ý |

## Quy tắc an toàn: không bao giờ thay thế truy vấn gốc

Truy vấn gốc luôn là biến thể số 0 và luôn có trọng số cao nhất. Các biến thể sau giảm dần theo `variant_decay ** i`. Một bản viết lại lệch hướng vì thế chỉ làm xê dịch thứ hạng chứ không làm mất câu trả lời.

Đây không phải lo xa. Trong demo, `fold diacritics` **làm hỏng** một truy vấn multipart (−0.33 ở nhóm đó) trong khi nâng `noaccent` +0.20 và `abbrev` +0.50. `run_s5.py` in riêng danh sách truy vấn bị từng cơ chế làm hỏng, kèm tên truy vấn.

## Chấm điểm theo nhóm, không chấm gộp

Con số tổng giấu hết mọi thứ ở kịch bản này. Một cơ chế nâng nhóm không dấu 40 điểm mà không ảnh hưởng nhóm khác sẽ hiện ra thành "+6 tổng thể" rồi bị loại oan. Nên mọi biến thể được chấm riêng trên:

| Nhóm | Mô tả |
|---|---|
| `plain` | Câu chuẩn, có dấu, một ý |
| `noaccent` | Gõ không dấu — bình thường, không phải ngoại lệ |
| `abbrev` | Viết tắt và cách nói thông tục |
| `multipart` | Hai câu hỏi trong một |

Chỉ số theo nhóm dùng **recall@1**, không phải @5: trên corpus vài chục chunk, top-5 là một phần lớn index nên recall@5 bão hoà và mọi biến thể trông giống nhau. Script cảnh báo khi corpus dưới 40 chunk.

## Chạy

```bash
pip install -r requirements.txt
python run_s5.py --demo
python run_s5.py --docs corpus.jsonl --eval eval.jsonl
```

```jsonl
eval.jsonl  {"query": "lam sao doi mat khau", "category": "noaccent", "answer_spans": ["..."]}
```

Trường `category` là thứ làm S5 có ý nghĩa. Nếu tập eval của bạn chỉ có câu hỏi chuẩn, S5 sẽ không cho thấy gì — và kết luận đúng không phải "query understanding vô dụng" mà là "tập eval chưa phản ánh đầu vào thật".

## Phân loại truy vấn và trọng số hợp nhất

`classify_query()` trả về `exact | semantic | mixed` bằng luật, dựa trên mã lỗi, tên sản phẩm, con số và từ để hỏi. Kết quả chọn trọng số:

| Loại | Dense | BM25 |
|---|---|---|
| `exact` | 0.4 | 1.0 |
| `semantic` | 1.0 | 0.4 |
| `mixed` | 0.7 | 0.7 |

Dùng luật chứ không dùng model là có chủ đích: kết quả phải kiểm toán được, và khi phân loại sai thì nhìn thấy nguyên nhân ngay.

## Chi phí

Mỗi biến thể là một lần gọi embedding và một lượt BM25. Demo trung bình 2,1 biến thể mỗi truy vấn. `max_variants` giới hạn trần; nếu latency quan trọng, đặt 2–3.

## Giới hạn

`LLMRewriter` có sẵn nhưng **tắt mặc định** (`rewrite=False`). Nó nhận một callable `llm(prompt) -> str` bất kỳ. Prompt được viết hẹp có chủ đích — một rewriter được phép "cải thiện" câu hỏi sẽ âm thầm đổi điều người dùng hỏi, và lỗi đó không hiện ra trong chỉ số retrieval.

Khoảng trống từ S4 — câu hỏi thuộc phạm vi khác — S5 **chưa bịt**. `classify_query` phân loại dạng truy vấn, không phân loại phạm vi. Cần nhận diện phạm vi ở phía truy vấn (mở rộng được từ module này) hoặc ngưỡng tin cậy ở S7.

## Điều kiện để sang S6

- Tập eval có đủ 4 nhóm, mỗi nhóm ít nhất 10 câu từ log thật.
- Đã biết cơ chế nào đóng góp cho nhóm nào.
- Đã xem danh sách truy vấn bị từng cơ chế làm hỏng và chấp nhận được.
- `max_variants` đã chốt theo ngân sách latency.
