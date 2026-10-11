# S3 — Cross-encoder reranking (thí nghiệm có thể bác bỏ)

S3 thêm tầng thứ hai: cross-encoder chấm lại top-N ứng viên sau fusion. Nhưng đây **không phải kịch bản "bật lên là xong"** — rerank có bằng chứng làm hại trong một số trường hợp, nên S3 được thiết kế để phát hiện điều đó trên corpus của bạn.

## Nội dung gói

| File | Vai trò | Trạng thái |
|---|---|---|
| `rerank.py` | Cross-encoder (bge / Qwen3) + stub offline, đo latency | **Mới ở S3** |
| `run_s3.py` | Ablation rerank, sweep candidate pool, chẩn đoán calibration | **Mới ở S3** |
| `retriever.py` | BM25 + dense + RRF **+ tầng rerank** | **Sửa ở S3** |
| `chunking.py` | 3 chiến lược chunking | Giữ nguyên từ S2 |
| `embedding.py` | Dense encoder | Giữ nguyên từ S1 |
| `demo_corpus.py` | Corpus FAQ tiếng Việt dùng chung | Từ S2 |
| `requirements.txt` · `README.md` | — | — |

Gói tự chứa, không import chéo thư mục.

## Vì sao phải đo chứ không bật mặc định

Hai kết quả đã công bố:

- Cross-encoder huấn luyện trên MS MARCO **giảm** precision ở mọi mức cutoff trên corpus ngoài miền — P@5 rơi từ 0.600 xuống 0.404. Reranker thích ứng miền cần được kiểm tra trước khi triển khai.
- Cross-encoder có thể **chôn** chunk mà tầng một đã xếp đúng ở hạng 1, điển hình là khi BM25 khớp chính xác một thuật ngữ mà cross-encoder tái trọng số bỏ qua.

Nên S3 báo cáo **precision@k, không chỉ recall@k**. Recall có thể giữ nguyên trong khi precision sụp — mà precision mới là thứ quyết định bao nhiêu tài liệu gây nhiễu lọt vào ngữ cảnh.

## Chạy

```bash
pip install -r requirements.txt
python run_s3.py --demo --sweep

python run_s3.py --docs corpus.jsonl --eval eval.jsonl \
    --reranker BAAI/bge-reranker-v2-m3 --sweep --latency-budget 800
```

Nhãn eval dùng `answer_spans` giống S2.

## Ba đầu ra

### 1. Chất lượng

```
variant                       rec@5   prec@5  ndcg@10   churn   p50 ms   p95 ms
no rerank (S1+S2)             0.889    0.178    0.801   0.000    0.254    2.469
rerank replace                0.778    0.156    0.629   2.244    1.310    2.381
rerank rrf-blend              0.889    0.178    0.699   1.200    1.490    2.829
```

Kết quả demo tái hiện đúng chế độ lỗi: `replace` (dùng thẳng thứ tự cross-encoder) làm **tụt cả precision lẫn recall**, còn `rrf-blend` kéo về bằng baseline. Đây là stub reranker nên số tuyệt đối vô nghĩa, nhưng dạng lỗi thì thật.

`rank_churn` = mức dịch chuyển trung bình của thứ hạng. Churn gần 0 nghĩa là reranker gần như không đổi gì — tầng một đã làm hết việc.

### 2. Chi phí

Sweep `rerank_candidates` ∈ {10, 20, 50, 100} kèm latency p95. Tìm điểm gãy: chất lượng thường phẳng sớm hơn latency nhiều. `--latency-budget` làm verdict fail nếu vượt ngân sách.

### 3. Calibration — đầu vào cho S7

```
variant                     gold med  other med     gap   overlap
rerank replace                 0.971      0.947   0.024     0.679
rerank rrf-blend               0.971      0.940   0.031     0.667
```

Đây là phần dễ bị bỏ qua nhất. Đóng góp thật của rerank vào việc giảm hallucination không nằm ở sắp xếp lại, mà ở chỗ nó tạo ra **một đại lượng so sánh được giữa các truy vấn** để S7 đặt ngưỡng từ chối. Điểm RRF phẳng và không so sánh được nên không dùng làm ngưỡng.

`overlap` = tỷ lệ chunk không chứa đáp án nhưng có điểm cao hơn phân vị 10 của nhóm chứa đáp án — chính là mức rò rỉ nếu đặt ngưỡng ở đó. Trên khoảng 0.3 thì điểm quá nhiễu, S7 không dựa vào được.

Trong demo, overlap ≈ 0.67 vì stub reranker không có ngữ nghĩa. Với cross-encoder thật, con số này là thứ quyết định S7 có khả thi hay không — đáng xem hơn cả bảng chất lượng.

## Hai chế độ hợp nhất

| `rerank_fusion` | Cách làm | Khi nào dùng |
|---|---|---|
| `"replace"` | Dùng thẳng thứ tự cross-encoder | Mặc định; đo trước |
| `"rrf"` | Hợp nhất thứ hạng CE với thứ hạng tầng một | Khi `replace` làm tụt precision |

`rrf` tồn tại vì chế độ lỗi đã nêu: nếu tầng một đã đúng thì reranker nên **cải thiện** nó chứ không **ghi đè** nó. Trộn theo thứ hạng giữ được phần tăng precision mà không cho cross-encoder quyền phủ quyết.

## Sigmoid chứ không softmax

Điểm được ép về [0,1] bằng sigmoid trên từng cặp (query, doc), không softmax trên cả danh sách. Softmax chuẩn hoá theo từng truy vấn nên điểm mất khả năng so sánh giữa các truy vấn — đúng thứ S7 cần.

## Ba kết luận có thể của S3

| Kết quả | Hành động |
|---|---|
| precision@5 tăng rõ | Nhận, chốt `rerank_candidates` theo sweep |
| precision@5 tụt | Thử `fusion="rrf"` → thử reranker khác → nếu vẫn tụt thì **giữ reranker chỉ để lấy điểm cho S7**, không dùng thứ tự |
| precision gần như không đổi, churn thấp | Tầng một đã đủ tốt; cân nhắc bỏ rerank trừ khi cần điểm cho S7 |

Lưu ý quan trọng: **"bác bỏ rerank" không có nghĩa là xoá reranker khỏi pipeline.** Nó vẫn có thể là nguồn điểm duy nhất đủ tin cậy cho cơ chế từ chối ở S7.

## Nhánh Qwen3-Reranker

`--rerank-kind qwen3` dùng định dạng prompt yes/no của Qwen3-Reranker. Hãy đối chiếu với model card trước khi tin số đo từ nhánh này; nhánh mặc định `seq-cls` dùng API chuẩn của `AutoModelForSequenceClassification` nên ít rủi ro hơn.

## Điều kiện để sang S4

- Đã có kết luận rõ ràng: nhận, bác bỏ, hay giữ-chỉ-để-lấy-điểm.
- `rerank_candidates` đã chốt qua sweep, p95 trong ngân sách.
- Đã ghi lại `overlap` để S7 biết điểm có dùng được làm ngưỡng không.
