# S6 — Chọn ngữ cảnh

S1–S5 quyết định **chunk nào có sẵn**. S6 quyết định chunk nào thực sự tới được model — và đây là nhóm tác động trực tiếp nhất lên hallucination, vì một lý do dễ hiểu ngược.

## Nội dung gói

| File | Vai trò | Trạng thái |
|---|---|---|
| `context_selection.py` | adaptive-k, MMR, U-shaped ordering | **Mới ở S6** |
| `run_s6.py` | Ablation 4 biến thể, sweep `gap_ratio` | **Mới ở S6** |
| `demo_corpus.py` | FAQ 21 mục, nhiều mục cận kề sinh distractor | **Mới ở S6** |
| `retriever.py` | + tầng `selection` sau rerank | **Sửa ở S6** |
| `query_understanding.py` · `metadata.py` · `rerank.py` · `chunking.py` · `embedding.py` | — | Giữ nguyên |

## Đại lượng cần tối thiểu hoá

Không phải "nội dung không liên quan" nói chung, mà **`avg_distractors`** — số chunk đúng chủ đề nhưng không chứa đáp án, trong ngữ cảnh cuối cùng.

Lý do: tài liệu liên quan nhưng không chứa đáp án gây hại **hơn hẳn** tài liệu hoàn toàn không liên quan. Chỉ một tài liệu gây nhiễu đã làm giảm độ chính xác tới 25% ở một số mô hình, và mức suy giảm tăng dần theo số lượng, vượt 67% trong một số trường hợp. Trong khi đó thêm tài liệu *ngẫu nhiên* lại cải thiện độ chính xác tới 35%.

Hệ quả thiết kế: **mỗi chunk đưa vào prompt là một rủi ro, không phải một tài sản.** Recall tăng mà `avg_distractors` cũng tăng thì đó không phải cải thiện.

## Kết quả demo

```
variant              recall   avg_k  distract   redund  edge_dist  reorder
V0 fixed k=6          1.000   6.000     5.000    0.397      0.200    0.000
V1 adaptive-k         1.000   4.300     3.300    0.363      0.200    0.000
V2 +MMR               1.000   4.300     3.300    0.363      0.100    0.000
V3 +U-shape           1.000   4.300     3.300    0.363      0.000    0.500
```

**adaptive-k bỏ 1,70 distractor mỗi truy vấn với recall không đổi.** Đây là loại cải thiện mà recall@k hoàn toàn không nhìn thấy — cả hai biến thể đều recall 1.000.

Sweep `gap_ratio` cho thấy toàn bộ đường cong đánh đổi:

```
gap_ratio   recall   avg_k  distract
     0.00    1.000    6.00      5.00    (= k cố định)
     0.30    1.000    5.40      4.40
     0.55    1.000    4.30      3.30
     0.75    0.900    2.40      1.50
     0.90    0.900    1.70      0.80
```

Đọc như một đường cong, không phải tìm người thắng: chọn điểm mà recall còn chấp nhận được và số distractor giảm nhiều nhất.

## Hai tương tác được kiểm tra tự động

**MMR có thể vô dụng.** Trong demo, `redundancy_drop` = 0 — ngữ cảnh vốn đã không trùng lặp. Script báo thẳng: hoặc corpus ít trùng lặp, hoặc chunking đã gộp sẵn các đoạn gần giống nhau, và khuyên tắt MMR để đỡ phức tạp. MMR chồng lấn với khử trùng lặp ở tầng chunking.

**adaptive-k và U-shape triệt tiêu nhau.** Reordering cho cải thiện không đáng kể trên tập truy hồi nhỏ, chỉ vượt trội khi tập lớn. Mà adaptive-k tồn tại chính để làm tập nhỏ. Trong demo U-shape chỉ kích hoạt 50% truy vấn vì k thực tế là 4,3 còn ngưỡng là 6.

Phải chọn một:
- Muốn ngữ cảnh nhỏ và sạch → giữ adaptive-k, tắt ordering.
- Muốn ordering → nâng `k_max`, nới `gap_ratio`, chấp nhận nhiều distractor hơn.

Bật cả hai như cấu hình mặc định thì ordering gần như trơ.

## Chỉ số vị trí: khoảng cách tới mép, không phải vị trí thô

`edge_dist` = `min(pos, n-1-pos)` của chunk chứa đáp án. U-shape đẩy bằng chứng mạnh ra **hai đầu**, nên vị trí thô tăng theo thiết kế và sẽ bị đọc nhầm thành suy giảm.

Dù vậy đây vẫn chỉ là **proxy**. Hiệu quả của ordering không đo được từ tầng retrieval — nó chỉ hiện ra ở chỉ số generation. Cần xác nhận với người làm tầng sau trước khi giữ.

## Chẩn đoán `cut_reason`

```
gap    = tìm thấy vách điểm số (lành mạnh)
k_max  = không có vách, điểm phẳng nên nhồi đến trần
floor  = mọi thứ dưới ngưỡng tuyệt đối
```

Nếu `k_max` chiếm đa số, adaptive-k đang không làm gì — thường vì điểm số chưa hiệu chuẩn.

## Phụ thuộc vào S3

adaptive-k cắt theo **độ dốc tương đối** của điểm số. Điểm RRF phẳng và không so sánh được giữa các truy vấn, nên trên điểm RRF thô thì phép cắt này kích hoạt gần như ngẫu nhiên. **S6 cần điểm reranker từ S3.**

Nếu S3 đã bị bác bỏ về precision, vẫn giữ reranker làm nguồn điểm — đây chính là lý do đã nêu ở README của S3.

## Chạy

```bash
pip install -r requirements.txt
python run_s6.py --demo --sweep
python run_s6.py --docs corpus.jsonl --eval eval.jsonl --sweep
```

```python
from context_selection import SelectionConfig
sel = SelectionConfig(use_adaptive_k=True, gap_ratio=0.55,
                      use_mmr=False, use_u_shape=False, k_max=6)
res = r.retrieve(q, rerank=True, selection=sel)
res.diagnostics["selection"]   # k_selected, cut_reason, mmr_dropped, redundancy
```

Bỏ `selection` thì retriever hành xử đúng như S5.

## Điều kiện để sang S7

- Đã chốt `gap_ratio` từ đường cong sweep, có lý do cho điểm đã chọn.
- Đã quyết định MMR bật hay tắt dựa trên `redundancy_drop` thực đo.
- Đã quyết định ordering bật hay tắt dựa trên k thực tế, không theo mặc định.
- `cut_reason` chủ yếu là `gap`, không phải `k_max`.
