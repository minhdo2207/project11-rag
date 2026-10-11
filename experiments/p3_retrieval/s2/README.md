# S2 — Context-preserving chunking

S2 nhắm vào một chế độ lỗi: **phân mảnh ngữ cảnh**. Chunk cắt giữa một định nghĩa, một bảng hay một quy trình tạo ra câu trả lời trôi chảy nhưng sai — model ghép nửa dữ kiện này với nửa dữ kiện kia. Retrieval không sửa được, vì không có chunk nào chứa đủ thông tin ngay từ đầu.

```
chunking.py    3 chiến lược: fixed | structure | ast, + bảo vệ khối nguyên tử
embedding.py   giống S1 (sao chép sang để thư mục tự chứa)
retriever.py   giống S1 — giữ cố định để chỉ đo biến chunking
run_s2.py      ablation 5 biến thể, đánh giá theo span
```

## Nội dung gói

| File | Vai trò | Trạng thái |
|---|---|---|
| `chunking.py` | 3 chiến lược chunking + bảo vệ khối nguyên tử + enrichment header | **Mới ở S2** |
| `run_s2.py` | Ablation 5 biến thể, đánh giá theo span, `fragmentation_rate` | **Mới ở S2** |
| `embedding.py` | Dense encoder | Giữ nguyên từ S1 |
| `retriever.py` | BM25 + dense + RRF | Giữ nguyên từ S1 |
| `requirements.txt` | Phụ thuộc | — |
| `README.md` | Tài liệu này | — |

Gói tự chứa: `embedding.py` và `retriever.py` được **sao chép nguyên vẹn** từ S1, không import chéo thư mục.

### Retrieval vẫn chạy đầy đủ trong S2

Mỗi biến thể chunking dựng index riêng rồi chạy `HybridRetriever(mode="hybrid")` giống hệt S1 — `answer_recall@k` và `span_coverage@k` đều là kết quả retrieval thật. Tầng retrieval được **giữ cố định có chủ đích** để ablation chỉ có một biến; nếu đổi đồng thời cả chunking lẫn tham số retrieval thì không quy được cải thiện về nguyên nhân nào.

Dù code retrieval không đổi một dòng, S2 vẫn tác động mạnh lên tầng này qua dữ liệu: `embed_text` đổi → vector đổi → nhánh dense đổi hoàn toàn; đồng thời vocabulary và phân phối độ dài chunk đổi → IDF và chuẩn hoá độ dài của BM25 đổi theo.


## Bẫy phương pháp: nhãn của S1 không dùng lại được

`gold_ids` của S1 **vô dụng ở đây**. Đổi chiến lược chunking là đổi toàn bộ tập chunk, nên mọi id đều khác. Dùng lại nhãn cũ thì bạn đo một thứ vô nghĩa mà không nhận ra.

S2 đánh giá theo **span trong tài liệu gốc**: một lần truy hồi được tính là trúng khi chunk trả về thực sự *chứa* đoạn văn bản chứa câu trả lời. Nhãn này sống sót qua mọi thay đổi chunking.

```jsonl
corpus.jsonl  {"source": "faq.md", "text": "<toàn văn tài liệu>"}
eval.jsonl    {"query": "...", "answer_spans": ["ngày 15 hàng tháng", "..."]}
```

Script từ chối chạy nếu tập eval không có `answer_spans`.

## Chỉ số quan trọng nhất: `fragmentation_rate`

Tỷ lệ span mà **không chunk nào chứa trọn**. Những span này là *không thể cứu*: không reranker nào, không embedding model lớn hơn nào, không tăng k nào chạm tới được, vì thông tin đã bị phá huỷ ngay lúc index.

Nó tính được **mà không cần chạy retrieval** — chỉ là thuộc tính của index. Đây là chẩn đoán rẻ nhất trong toàn bộ dự án. Đưa nó về 0 trước khi tinh chỉnh bất cứ thứ gì khác.

## Chạy

```bash
pip install numpy scipy
python run_s2.py --demo

python run_s2.py --docs corpus.jsonl --eval eval.jsonl --model Qwen/Qwen3-Embedding-0.6B
```

Kết quả demo:

```
variant                chunks avg_len frag_rate  ans@1  ans@5  cov@5
V0 fixed-700                7     696     0.214  0.444  0.778  0.722
V1 fixed+overlap            9     640     0.000  0.444  0.778  0.778
V2 structure               12     384     0.000  0.444  0.889  0.889
V3 +atomic blocks          12     384     0.000  0.444  0.889  0.889
V4 +enrich header          12     384     0.000  0.556  0.889  0.889
```

Đọc: fixed-700 làm mất 21,4% số span — 3 trong 14 câu trả lời không thể truy hồi dù retrieval hoàn hảo. Structure đưa về 0 và nâng `cov@5` từ 0.722 lên 0.889, đổi lại 1,71 lần số chunk.

## Thang cộng dồn — tách đóng góp từng cơ chế

| Biến thể | Thêm gì |
|---|---|
| V0 | fixed-size 700 ký tự, không overlap |
| V1 | + overlap 120 ký tự |
| V2 | + ranh giới theo cấu trúc (heading, đoạn, AST) |
| V3 | + bảo vệ khối nguyên tử |
| V4 | + enrichment header vào `embed_text` |

`print_mechanism_deltas()` in delta của từng bậc, nên bạn biết lợi ích đến từ cơ chế nào chứ không chỉ biết "cấu hình cuối tốt hơn".

Trong demo, V3 cho delta bằng 0 vì ranh giới heading đã vô tình giữ nguyên bảng rồi. Đó là kết quả thật, không phải lỗi — trên corpus có bảng nằm giữa một section dài thì V3 mới phát huy. Nếu bạn cũng thấy V3 bằng 0, kiểm tra xem corpus có khối nào thực sự bị heading cắt không trước khi kết luận cơ chế vô dụng.

## Khối nguyên tử được bảo vệ

`atomic_spans()` nhận diện và không bao giờ cắt: code fence (``` hoặc ~~~), bảng markdown liền mạch, khối list/quy trình đánh số, và stack trace kể cả toàn bộ dòng `Caused by:`.

Riêng stack trace: nguyên nhân gốc nằm ở `Caused by:` **cuối cùng**, nên cắt phần đuôi là kiểu cắt tệ nhất có thể — model sẽ quy lỗi cho frame sai.

## Ba chiến lược

| Chiến lược | Dùng cho | Cơ chế |
|---|---|---|
| `fixed` | baseline để so | Cắt theo offset ký tự |
| `structure` | văn bản, markdown | Heading > dòng trống > mép khối nguyên tử |
| `ast` | file `.py` | Một chunk mỗi hàm/class, fallback sang `structure` khi file không parse được |

`chunk_documents()` tự động route file `.py` sang AST chunker khi `strategy` là `structure` hoặc `ast`.

## Chuẩn hoá tiếng Việt

Hàm so khớp span dùng NFC + lowercase + gộp khoảng trắng. Không có NFC thì hai chuỗi tiếng Việt nhìn giống hệt nhau vẫn không khớp, do corpus và nhãn dùng cách tổ hợp dấu khác nhau — lỗi này rất khó phát hiện vì mắt thường không thấy.

## Điều kiện để sang S3

- `fragmentation_rate` = 0 trên tập eval thật.
- Có biến thể vượt baseline fixed-size về `span_coverage@5`.
- Chi phí index (số chunk) chấp nhận được — nhiều chunk hơn nghĩa là embed lâu hơn và lưu trữ lớn hơn.

Nếu không biến thể nào thắng baseline, thường do span trong tập eval quá ngắn nên không chunker nào cắt phải. Bổ sung span nhiều câu và câu trả lời dạng quy trình rồi đo lại.
