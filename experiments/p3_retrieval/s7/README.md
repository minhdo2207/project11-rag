# S7 — Hiệu chuẩn độ tin cậy và quyết định trả lời

S1–S6 cải thiện **bằng chứng**. S7 quyết định **làm gì khi bằng chứng yếu** — điểm duy nhất trong tầng retrieval có thể chặn một câu trả lời sai tự tin.

## Nội dung gói

| File | Vai trò | Trạng thái |
|---|---|---|
| `confidence.py` | Hiệu chuẩn, đường cong đánh đổi, thang 4 mức quyết định | **Mới ở S7** |
| `run_s7.py` | Hiệu chuẩn + đo quyết định trên cả hai nhóm câu hỏi | **Mới ở S7** |
| `demo_corpus.py` | 12 câu trả lời được + 11 câu không trả lời được, 3 loại | **Mới ở S7** |
| `retriever.py` | + `decision`, + thuộc tính `may_assert` | **Sửa ở S7** |
| 6 module còn lại | — | Giữ nguyên từ S1–S6 |

## Vì sao không dùng từ chối nhị phân

Người dùng đến hệ hỏi đáp để **được cung cấp thông tin**. Hệ thống cứ không chắc là từ chối thì an toàn nhưng vô dụng. Nên quyết định được chia bốn mức:

| Mức | Nghĩa | `may_assert` |
|---|---|---|
| `answer` | Bằng chứng mạnh, trả lời bình thường | ✓ |
| `answer_conditional` | Bằng chứng dùng được, nêu kèm điều kiện và trích dẫn | ✓ |
| `evidence_only` | Chưa đủ để kết luận, nhưng đây là các đoạn liên quan nhất | ✗ |
| `abstain` | Không có gì đáng hiển thị | ✗ |

`evidence_only` là mức quan trọng nhất cho hệ phục vụ người dùng cuối: **chi phí hallucination bằng 0** vì không khẳng định điều gì, trong khi người dùng vẫn nhận được thứ để dùng.

`decision.may_assert` là hợp đồng với tầng generation. `False` nghĩa là không được phát biểu khẳng định nào, bất kể trong `hits` có gì.

## Tham số bạn phải tự chốt

`--cost-ratio` = chi phí của một câu trả lời sai tự tin chia cho chi phí của một lần từ chối không cần thiết. **Không có giá trị đúng phổ quát.** Một câu trả lời sai về chính sách hay cước phí có thể tốn một khiếu nại và một khách hàng; một lần từ chối tốn thêm một cú nhấp.

Tác động của tham số này trên dữ liệu demo:

| cost_ratio | ngưỡng | false_answer | over_refusal | coverage |
|---|---|---|---|---|
| 1 | 0.937 | 0.091 | 0.333 | 0.667 |
| 5 | 0.937 | 0.091 | 0.333 | 0.667 |
| 20 | 0.998 | 0.000 | 0.917 | **0.083** |

Ở mức 20, hệ thống gần như không trả lời gì — đúng về mặt toán học, vô dụng về mặt sản phẩm. Chọn tham số này có chủ đích và ghi vào báo cáo, vì mọi ngưỡng phía sau kế thừa nó.

## Hiệu chuẩn phải chạy trước

```
calibration — can the score tell answerable from unanswerable?
  answerable    median=0.961  p25=0.937  p10=0.824  n=12
  unanswerable  median=0.824  p90=0.971  n=11
  separation    0.455
```

`separation` = tỷ lệ câu không trả lời được có điểm dưới phân vị 10 của nhóm trả lời được.

| separation | Ý nghĩa |
|---|---|
| < 0.3 | Điểm số gần như không mang thông tin về khả năng trả lời. **Không ngưỡng nào cứu được** — quay lại S3 |
| 0.3 – 0.6 | Dùng được, nhưng hai tầng giữa sẽ gánh nhiều lưu lượng — đó chính là việc của chúng |
| > 0.6 | Tốt, ngưỡng phân tách sạch |

Đây chính là thứ chỉ số `overlap` ở S3 đo trước. Nếu S3 báo `overlap` cao, S7 sẽ không chạy tốt và bạn biết điều đó từ trước.

## Ba ngưỡng được suy ra, không đặt tay

```
t_high = median của nhóm trả lời được     -> trả lời tự tin
t_mid  = ngưỡng tối ưu chi phí            -> dưới mức này, không khẳng định
t_low  = median của nhóm không trả lời được -> dưới mức này, không hiển thị gì
```

Điểm dễ sai: ngưỡng tối ưu chi phí là ranh giới **ngừng khẳng định**, không phải ranh giới **ngừng hiển thị**. Đặt `t_low = t_mid` làm tầng `evidence_only` rỗng và biến cả cơ chế trở lại thành từ chối nhị phân — đúng thứ S7 sinh ra để tránh.

## Ba loại câu không trả lời được

Nhóm này không phải kiểm thử phụ — nó là nửa mà toàn bộ kịch bản được đo trên đó. Ba loại thất bại khác nhau:

| Loại | Mô tả | Kết quả demo |
|---|---|---|
| `absent` | Chủ đề không có trong corpus | 0.00 bị khẳng định |
| `adjacent` | Chủ đề **có**, nhưng dữ kiện được hỏi thì không | **0.20** |
| `oos` | Câu trả lời tồn tại nhưng ngoài phạm vi bộ lọc | 0.00 |

`adjacent` là loại khó nhất, đúng như dự đoán: corpus có phủ chủ đề nên retrieval trả về các chunk trông rất thuyết phục, chỉ là không chứa dữ kiện được hỏi. Giữa đó và một câu bịa tự tin chỉ còn mỗi cái ngưỡng.

**Khoảng trống của S4 đã được bịt:** nhóm `oos` giờ 0.00 bị khẳng định. S4 không phát hiện được câu hỏi thuộc phạm vi khác, nhưng ngưỡng tin cậy của S7 bắt được mức khớp yếu.

## Kết quả

```
                     answer  conditional  evidence  abstain
  answerable           0.42         0.25      0.00     0.33
  unanswerable         0.00         0.09      0.45     0.45

  false_answer_rate        0.091
  over_refusal             0.333
  coverage                 0.667
  precision_when_answered  1.000
```

So với không có S7: `false_answer_rate` từ **1.000 xuống 0.091**, đổi lại coverage từ 1.000 xuống 0.667.

Lưu ý con số `coverage` ở demo thấp do stub reranker nhiễu. Với cross-encoder thật, `separation` cao hơn nên cùng một mức `false_answer_rate` sẽ giữ được coverage tốt hơn nhiều.

## Tín hiệu thứ hai: margin

Ngoài điểm tuyệt đối, S7 kiểm tra `top1 − top2`. Đỉnh bảng xếp hạng phẳng nghĩa là retriever tìm được vài ứng viên ngang nhau và không phân biệt được — đúng trạng thái sinh ra câu trả lời sai tự tin, kể cả khi điểm tuyệt đối cao. Margin dưới `min_margin` thì hạ một bậc.

## Chạy

```bash
pip install -r requirements.txt
python run_s7.py --demo --cost-ratio 5
python run_s7.py --docs corpus.jsonl --eval eval.jsonl --cost-ratio 3
```

```jsonl
{"query": "...", "answer_spans": ["..."]}                         # trả lời được
{"query": "...", "answerable": false, "kind": "adjacent"}         # không trả lời được
```

Script **dừng với mã lỗi 2** nếu tập eval không có câu không trả lời được: `false_answer_rate` sẽ không xác định và ngưỡng sẽ được fit trên một lớp duy nhất. Mục tiêu 20–30% tập eval.

```python
from confidence import ConfidenceConfig
res = r.retrieve(q, rerank=True, confidence=ConfidenceConfig(t_high=.8, t_mid=.5, t_low=.25))
if not res.may_assert:
    ...   # hiển thị res.decision.message + res.hits, KHÔNG sinh câu trả lời
```

## Điều kiện để sang S8

- `separation` ≥ 0.3, lý tưởng ≥ 0.6.
- Đã chốt `cost_ratio` và ghi lý do vào báo cáo.
- Tầng generation đã xác nhận tôn trọng `may_assert`.
- Tầng `evidence_only` không rỗng — nếu rỗng thì ba ngưỡng đang bị gộp.
