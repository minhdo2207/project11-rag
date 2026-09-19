# Báo cáo tìm hiểu tài liệu — Project 11: RAG (Concepts & Applications)

**Chủ đề:** Retrieval-Augmented Generation và ứng dụng phát hiện mã độc trong hệ sinh thái PyPI
**Người thực hiện:** (điền tên) — vai trò: Software Engineer / Kiến trúc & Persistence (P1)
**Ngày:** 2026-09-15

---

## 1. Mục tiêu báo cáo

Tổng hợp 3 bài báo liên quan trực tiếp tới đề tài, rút ra:
- Các phương pháp phát hiện mã độc bằng LLM/ML hiện nay, điểm mạnh – điểm yếu.
- Vì sao RAG trong bài tham chiếu cho kết quả kém, và kỹ thuật cải tiến RAG 2025.
- Những gì áp dụng được vào project của nhóm và phần đóng góp của vai Software Engineer.

Ba bài được chọn bổ trợ nhau: (1) **bài tham chiếu gốc** đặt vấn đề RAG; (2) một **hướng thay thế** (phân tích động bằng ML) để so sánh; (3) một **khảo sát kỹ thuật cải tiến RAG** làm công cụ nâng cấp.

---

## 2. Tóm tắt 3 bài báo

### Bài 1 (bài tham chiếu chính) — *Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?*
- **Tác giả / nguồn:** Ibiyo, Louangdy, Nguyen, Di Sipio, Di Ruscio — EASE 2025. arXiv:2504.13769.
- **Vấn đề:** Gói độc trên PyPI được **thiết kế để đánh lừa người dùng**, tiến hoá liên tục, thiếu dataset có cấu trúc → khó phát hiện bằng công cụ quét lỗ hổng truyền thống.
- **Phương pháp so sánh:**
  1. **Fine-tuning** LLM trên dataset đã tuyển chọn.
  2. **RAG** với nguồn tri thức: **YARA rules + GitHub Security Advisories (GHSA) + mẫu mã độc**.
  3. **Few-shot learning** (đưa vài ví dụ mẫu vào prompt).
- **Kết quả (bất ngờ):**
  - RAG: độ chính xác **ở mức trung bình**, thấp hơn kỳ vọng.
  - Few-shot: **97% accuracy, 95% balanced accuracy** — tốt nhất.
- **Hạn chế & hướng phát triển (tác giả tự nêu):** cần **mở rộng knowledge base có cấu trúc**, **tinh chỉnh mô hình truy xuất (retrieval)**, và hướng tới **giải pháp lai (hybrid)**.
- **Bài học:** RAG không "vô dụng" mà **bị triển khai chưa tới** — đúng khe hở để nhóm cải tiến.

### Bài 2 (hướng thay thế) — *DySec: A Machine Learning-based Dynamic Analysis for Detecting Malicious Packages in PyPI*
- **Tác giả / nguồn:** Sk Tanzir Mehedi, Chadni Islam, Gowri Ramachandran, Raja Jurdak. arXiv:2503.00324.
- **Ý tưởng:** Thay vì đọc code tĩnh, **quan sát hành vi lúc cài đặt** gói bằng **eBPF** (probe ở mức kernel và user).
- **Đặc trưng & mô hình:** theo dõi **36 đặc trưng thời gian thực** — system calls, network traffic, tiêu thụ tài nguyên, truy cập thư mục, hành vi cài đặt — rồi phân loại bằng học máy.
- **Dataset:** **14.271 gói** (trong đó **7.127 mẫu độc**), chạy trong môi trường cách ly.
- **Kết quả:** accuracy **95,99%**, độ trễ **<0,5s**; giảm false-negative **78,65%** so với phân tích tĩnh và **82,24%** so với phân tích metadata; phát hiện được gói độc thật mà PyPI sau đó gỡ bỏ.
- **Điểm mạnh:** bắt được payload sinh động lúc runtime, typosquatting, remote access. **Điểm yếu:** cần môi trường sandbox, tốn hạ tầng, không giải thích ngữ nghĩa code.

### Bài 3 (bộ công cụ cải tiến) — *Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers*
- **Nguồn:** arXiv:2506.00054.
- **Phân loại kỹ thuật nâng cấp RAG:**
  1. **Hybrid retrieval** — kết hợp dense (vector) + sparse (BM25) + knowledge graph → tăng recall.
  2. **Reranking** — chấm lại độ liên quan sau truy xuất, lọc ngữ cảnh nhiễu → tăng precision.
  3. **Chunking** — cắt tài liệu hợp lý (khuyến nghị thử **256–512 token**) tránh vỡ ngữ cảnh.
  4. **Contextual retrieval** — truy xuất thích ứng theo độ phức tạp câu hỏi, multi-hop.
  5. **Robustness / chống hallucination** — kiểm chứng câu trả lời dựa trên bằng chứng đã truy xuất.
- **Khuyến nghị hành động:** truy xuất **đa tầng** (lexical để recall → neural rerank để precision); tối ưu kích thước chunk; **đối chiếu câu trả lời với nguồn**; buộc mô hình chỉ ra passage nào làm căn cứ → **giảm hallucination** bằng cách "neo" vào nguồn kiểm chứng được.

---

## 3. Bảng so sánh nhanh

| Tiêu chí | Few-shot (Bài 1) | RAG (Bài 1) | Fine-tuning (Bài 1) | Dynamic ML — DySec (Bài 2) |
|---|---|---|---|---|
| Cách hoạt động | Vài ví dụ trong prompt | Truy xuất tri thức + LLM | Train lại trọng số | Quan sát hành vi runtime |
| Độ chính xác | **97% / 95% balanced** | Trung bình | Cao nếu đủ data | **95,99%** |
| Cập nhật mã độc mới | Đổi ví dụ, tức thì | Cập nhật KB, không train | Phải train lại | Thu thêm hành vi |
| Giải thích được | Kém | **Tốt (truy nguồn)** | Kém | Trung bình |
| Chi phí hạ tầng | Thấp | Trung bình | Cao (GPU) | Cao (sandbox/eBPF) |
| Điểm yếu chính | Phụ thuộc chọn ví dụ | KB hẹp, retrieval yếu | Overfit, tốn data | Cần môi trường chạy |

---

## 4. Ứng dụng vào project của nhóm

### 4.1. Định hướng nghiên cứu (câu chuyện xuyên suốt)
Bài 1 kết luận RAG kém **vì KB hẹp + retrieval yếu**. Nhóm sẽ:
1. Tái hiện **few-shot làm baseline mạnh** (mục tiêu chạm mức ~97% của bài gốc).
2. **Cải tiến RAG** bằng kỹ thuật từ Bài 3 (hybrid + rerank + chunking hợp lý) và chứng minh RAG-cải-tiến thu hẹp/vượt khoảng cách với few-shot.
3. Đo **mức giảm hallucination** khi có/không RAG (đúng task 2 của đề).

### 4.2. Áp dụng cụ thể theo module (nhóm 5 người)
- **P1 – Kiến trúc & Persistence (vai Software Engineer):**
  - Interface `KnowledgeRepository` + hai bản `InMemory` / `File` (đáp ứng yêu cầu non-functional "quản lý cả memory lẫn file, đổi backend sửa tối thiểu code").
  - Định nghĩa interface chung (Embedder, Retriever, LLMClient, Detector) để nhóm code song song.
  - Benchmark **memory vs file** (thời gian nạp/tìm kiếm, RAM khi KB lớn dần) — trục nghiên cứu riêng.
- **P2 – Data & Knowledge Base:** thu thập gói lành/độc; nạp **YARA + GHSA + mẫu mã độc** (từ Bài 1); có thể bổ sung nhãn từ dataset kiểu DySec.
- **P3 – Embedding & Retrieval:** hiện thực **hybrid retrieval (BM25 + vector) + cross-encoder rerank + chunking theo AST/hàm** (từ Bài 3) — nơi cải tiến chính.
- **P4 – LLM & Detection:** chạy Qwen/Mistral/Llama qua Ollama; ba chế độ **no-RAG / RAG / RAG+few-shot**; buộc model xuất **bằng chứng + dòng code đáng ngờ** (chống hallucination).
- **P5 – Evaluation & Demo:** accuracy, **balanced accuracy**, recall, **false-negative rate**; biểu đồ so sánh model & chế độ; notebook demo trên Colab.

### 4.3. Các cải tiến rút ra trực tiếp từ tài liệu
1. **Hybrid retrieval (BM25 + vector):** mã độc lộ qua từ khoá chính xác (`eval`, `base64.b64decode`, `exec`, `os.system`, URL lạ) — BM25 bắt tốt phần vector bỏ sót. (Bài 1 nêu retrieval yếu; Bài 3 đưa giải pháp.)
2. **Re-ranking bằng cross-encoder** trước khi đưa vào prompt → lọc nhiễu.
3. **Chunking theo AST/hàm** thay vì cắt thô; thử dải 256–512 token.
4. **Mở rộng & làm sạch KB** (YARA/GHSA đầy đủ) — đúng khuyến nghị Bài 1.
5. **RAG + few-shot kết hợp** (Bài 1 tách riêng — nhóm ghép lại).
6. **Hướng lai với tín hiệu hành vi kiểu DySec** (mở rộng nâng cao): dùng metadata/hành vi làm tầng lọc nhanh, LLM+RAG xử lý ca nghi ngờ.

### 4.4. Rủi ro & lưu ý
- Không có GPU → **ưu tiên few-shot + RAG**, hạn chế fine-tuning (chỉ làm baseline nếu tài nguyên cho phép).
- **False-negative** (bỏ sót mã độc) nguy hiểm nhất trong bảo mật → báo cáo cần nhấn chỉ số này, không chỉ accuracy.
- Dataset mất cân bằng → dùng **balanced accuracy**, không chỉ accuracy thô.

---

## 5. Kết luận

- Few-shot hiện là baseline mạnh nhất (Bài 1); RAG chưa phát huy do **KB hẹp + retrieval yếu** — có thể khắc phục bằng kỹ thuật của Bài 3.
- DySec (Bài 2) cho thấy hướng **phân tích động** là bổ sung giá trị, mở đường cho giải pháp **lai**.
- Đóng góp của vai Software Engineer tập trung ở **kiến trúc modular, lớp persistence memory/file hoán đổi, chất lượng code + test, quy trình Git và benchmark hạ tầng** — vừa khó nhất, vừa khớp tiêu chí chấm điểm.

---

## 6. Tài liệu tham khảo

1. M. Ibiyo, T. Louangdy, P. T. Nguyen, C. Di Sipio, D. Di Ruscio. *Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?* EASE 2025. arXiv:2504.13769.
2. S. T. Mehedi, C. Islam, G. Ramachandran, R. Jurdak. *DySec: A Machine Learning-based Dynamic Analysis for Detecting Malicious Packages in PyPI Ecosystem.* arXiv:2503.00324.
3. *Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers.* arXiv:2506.00054.
