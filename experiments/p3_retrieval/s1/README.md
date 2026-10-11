# S1 — Hybrid retrieval baseline

Ba thành phần: **dense bi-encoder + BM25 (code-aware tokenizer) + RRF (k=60)**. Không rerank, không adaptive-k, không abstain — những cái đó thuộc S3, S6, S7.

```
embedding.py   dense encoder: asymmetric instruction, L2 normalise, cache theo hash
retriever.py   BM25 + dense + RRF, có switch mode = dense | bm25 | hybrid
run_s1.py      chạy ablation ba chế độ, in bảng so sánh + verdict
```

## Nội dung gói

| File | Vai trò | Trạng thái |
|---|---|---|
| `embedding.py` | Dense encoder: asymmetric instruction, L2 normalise, cache theo hash nội dung | Mới ở S1 |
| `retriever.py` | BM25 + dense + RRF, switch `mode` = dense \| bm25 \| hybrid | Mới ở S1 |
| `run_s1.py` | Ablation ba chế độ, bảng so sánh, verdict, sweep `rrf_k` | Mới ở S1 |
| `requirements.txt` | Phụ thuộc | — |
| `README.md` | Tài liệu này | — |

Gói tự chứa: không import gì từ thư mục kịch bản khác.


## Chạy

```bash
pip install numpy scipy                      # demo chỉ cần hai gói này
python run_s1.py --demo --sweep              # dữ liệu giả, không tải model

pip install sentence-transformers torch      # chạy thật
python run_s1.py --docs corpus.jsonl --eval eval.jsonl --sweep --save ./index
```

Định dạng:

```jsonl
corpus.jsonl  {"id": "d1", "text": "...", "metadata": {"source": "faq.md"}}
eval.jsonl    {"query": "làm sao đổi mật khẩu", "gold_ids": ["d1", "d5"]}
```

## Sản phẩm của S1 là bảng này, không phải cái retriever

```
mode         recall@1   recall@3   recall@5  recall@10    ndcg@10
dense           0.333      0.500      0.667      0.833      0.614
bm25            0.500      0.833      0.833      1.000      0.809
hybrid          0.333      0.667      0.833      0.833      0.665
```

Ba thứ đọc được từ đó:

1. **Hybrid phải thắng cả hai nhánh đơn.** Nếu không, vấn đề nằm ở thượng nguồn (chunking hoặc chính tập eval), không phải ở model — và không nên sang S2.
2. **Nếu dense ≈ bm25**, embedding model gần như không đóng góp gì trên corpus này. Kiểm tra trước khi trả tiền cho model lớn hơn.
3. **Per-query diff** cho biết fusion sửa được câu nào và làm hỏng câu nào. Câu bị hỏng đáng xem hơn con số tổng — thường là một nhánh kéo kết quả tốt của nhánh kia xuống, và đó là tín hiệu về `rrf_k` hoặc trọng số chứ không phải nhiễu.

Script trả exit code 1 khi verdict không đạt, nên cắm vào CI được.

## Hai cơ chế bảo vệ trong verdict

**Kiểm tra bão hoà.** Nếu cả ba chế độ đạt recall@5 ≥ 0.98, bảng vô nghĩa — tập eval quá dễ hoặc corpus quá nhỏ. Script dừng và yêu cầu thêm câu hỏi khó cùng hard negative.

**Hard negative trong tập eval.** Corpus demo có 14 đoạn `n*` cùng chủ đề nhưng không chứa đáp án (ERR_4015, ERR_4021, Combo 60, Combo 120, đổi PIN SIM...). Tập eval thật cũng cần loại này — không có nó thì recall@5 bão hoà và bạn không phân biệt được cấu hình nào tốt hơn.

> Chạy `--demo` sẽ thấy cảnh báo "hybrid does not beat bm25". Đó là **đúng hành vi**: embedder trong demo chỉ là bag-of-token hashing, không có ngữ nghĩa, nên nhánh dense yếu và kéo fusion xuống. Nó minh hoạ điều kiện dừng hoạt động, không phải lỗi code.

## Vì sao RRF chứ không trộn điểm

BM25 cho điểm dương không chặn, cosine bị chặn trong [-1, 1]. Trộn điểm thô mặc nhiên cho BM25 trọng số áp đảo. RRF chỉ thao tác trên thứ hạng nên không cần giả định về thang điểm. `rrf_k` nhỏ ưu ái top đầu mạnh hơn; quét `--sweep` sau khi có tập eval.

## Vì sao tokenizer giữ cả hai dạng

`build_comm_edges` được giữ nguyên **và** tách thành `build`, `comm`, `edges`. Chỉ giữ token đầy đủ thì không khớp câu hỏi diễn đạt tự nhiên; chỉ giữ subtoken thì mất khả năng khớp chính xác tên riêng, mã lỗi, tên gói cước. Giữ cả hai thắng ở cả hai kiểu hỏi, đổi lại vocabulary lớn hơn.

## Ghép vào hệ thống sẵn có

`index()` nhận `Document`, `dict`, hoặc object bất kỳ có `id`/`chunk_id` + `text`/`content`, nên không cần đổi schema. Embedder là tham số truyền vào — chỉ cần có `encode_documents` và `encode_queries`.

```python
from embedding import Embedder, EmbeddingConfig
from retriever import HybridRetriever, RetrieverConfig

r = HybridRetriever(Embedder(EmbeddingConfig(device="cuda")),
                    RetrieverConfig(mode="hybrid", top_k=5)).index(docs)

res = r.retrieve("làm sao đổi mật khẩu")
print(res.context())        # đã kèm citation id
print(res.diagnostics)      # from_dense_only / from_lexical_only / from_both
```

`diagnostics` cho biết mỗi nhánh đóng góp bao nhiêu kết quả vào top-k — ghi vào trace để sau này quy được lỗi về nhánh nào.

## Điều kiện để sang S2

- Hybrid thắng cả hai nhánh đơn trên tập eval thật.
- Tập eval không bão hoã (recall@5 < 0.95) và có hard negative.
- Đã chốt `rrf_k` qua `--sweep`.
