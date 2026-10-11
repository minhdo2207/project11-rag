"""S4 demo corpus — the same policy in two versions.

Built so the 2025 and 2026 documents are near-identical in wording and differ
only in the numbers. That is exactly the shape that defeats semantic retrieval:
the superseded chunk is not a bad match, it is an excellent match to the wrong
universe. Only metadata separates them.

Every eval row carries `answer_spans` (the current, correct text) and
`stale_spans` (the superseded text that must NOT be served).
"""

DEMO_DOCS = [
    # ---------------- superseded ---------------- #
    {
        "source": "chinh_sach_thanh_toan_2025.md",
        "metadata": {"doc_type": "policy", "version": "2025.1", "status": "superseded",
                     "effective_from": "2025-01-01", "effective_to": "2025-12-31",
                     "audience": "retail"},
        "text": """\
# Chính sách thanh toán 2025

Hiệu lực từ ngày 01/01/2025. Văn bản này đã hết hiệu lực và được thay thế bởi
Chính sách thanh toán 2026.

## Thời hạn thanh toán
Hóa đơn phát hành ngày 1 hàng tháng. Thời hạn thanh toán là ngày 20 hàng tháng.
Quá hạn 10 ngày, dịch vụ bị tạm ngưng một chiều. Quá hạn 20 ngày, dịch vụ bị
tạm ngưng hai chiều.

## Phí khôi phục
Phí khôi phục dịch vụ sau tạm ngưng là 15.000 đồng.

## Thanh toán tự động
Hệ thống tự động trừ tiền vào ngày 18 hàng tháng, thử lại tối đa 2 lần.

## Hạn mức thanh toán qua ứng dụng
Hạn mức mỗi lần giao dịch qua ứng dụng là 10.000.000 đồng.
""",
    },
    # ---------------- current ---------------- #
    {
        "source": "chinh_sach_thanh_toan_2026.md",
        "metadata": {"doc_type": "policy", "version": "2026.1", "status": "current",
                     "effective_from": "2026-01-01", "effective_to": None,
                     "audience": "retail"},
        "text": """\
# Chính sách thanh toán 2026

Hiệu lực từ ngày 01/01/2026, thay thế Chính sách thanh toán 2025.

## Thời hạn thanh toán
Hóa đơn phát hành ngày 1 hàng tháng. Thời hạn thanh toán là ngày 15 hàng tháng.
Quá hạn 7 ngày, dịch vụ bị tạm ngưng một chiều. Quá hạn 15 ngày, dịch vụ bị
tạm ngưng hai chiều.

## Phí khôi phục
Phí khôi phục dịch vụ sau tạm ngưng là 20.000 đồng.

## Thanh toán tự động
Hệ thống tự động trừ tiền vào ngày 10 hàng tháng, thử lại tối đa 3 lần.

## Hạn mức thanh toán qua ứng dụng
Hạn mức mỗi lần giao dịch qua ứng dụng là 20.000.000 đồng.
""",
    },
    # ---------------- superseded tariff ---------------- #
    {
        "source": "goi_cuoc_combo_2025.md",
        "metadata": {"doc_type": "tariff", "version": "2025.2", "status": "superseded",
                     "effective_from": "2025-06-01", "effective_to": "2026-02-28",
                     "audience": "retail"},
        "text": """\
# Gói cước Combo 2025

Áp dụng từ ngày 01/06/2025. Các gói dưới đây đã ngừng cung cấp cho thuê bao mới.

## Combo 90
Gói Combo 90 có 60GB data tốc độ cao mỗi tháng, miễn phí gọi nội mạng dưới
10 phút. Giá 90.000 đồng mỗi tháng.

## Combo 120
Gói Combo 120 có 90GB data, miễn phí gọi ngoại mạng 30 phút.
""",
    },
    # ---------------- current tariff ---------------- #
    {
        "source": "goi_cuoc_combo_2026.md",
        "metadata": {"doc_type": "tariff", "version": "2026.1", "status": "current",
                     "effective_from": "2026-03-01", "effective_to": None,
                     "audience": "retail"},
        "text": """\
# Gói cước Combo 2026

Áp dụng từ ngày 01/03/2026.

## Combo 90
Gói Combo 90 có 90GB data tốc độ cao mỗi tháng, miễn phí gọi nội mạng dưới
20 phút. Giá 90.000 đồng mỗi tháng.

## Combo 120
Gói Combo 120 có 120GB data, miễn phí gọi ngoại mạng 50 phút.
""",
    },
    # ---------------- different audience ---------------- #
    {
        "source": "chinh_sach_doanh_nghiep_2026.md",
        "metadata": {"doc_type": "policy", "version": "2026.1", "status": "current",
                     "effective_from": "2026-01-01", "effective_to": None,
                     "audience": "enterprise"},
        "text": """\
# Chính sách thanh toán doanh nghiệp 2026

Hiệu lực từ ngày 01/01/2026, áp dụng cho khách hàng doanh nghiệp.

## Thời hạn thanh toán
Thời hạn thanh toán là ngày 25 hàng tháng. Quá hạn 30 ngày mới tạm ngưng dịch vụ.

## Phí khôi phục
Khách hàng doanh nghiệp được miễn phí khôi phục dịch vụ.
""",
    },
]

DEMO_EVAL = [
    {"query": "thời hạn thanh toán hóa đơn là ngày nào",
     "answer_spans": ["Thời hạn thanh toán là ngày 15 hàng tháng"],
     "stale_spans": ["Thời hạn thanh toán là ngày 20 hàng tháng"]},
    {"query": "chậm đóng tiền bao lâu thì bị cắt dịch vụ",
     "answer_spans": ["Quá hạn 7 ngày, dịch vụ bị tạm ngưng một chiều"],
     "stale_spans": ["Quá hạn 10 ngày, dịch vụ bị tạm ngưng một chiều"]},
    {"query": "phí khôi phục dịch vụ là bao nhiêu",
     "answer_spans": ["Phí khôi phục dịch vụ sau tạm ngưng là 20.000 đồng"],
     "stale_spans": ["Phí khôi phục dịch vụ sau tạm ngưng là 15.000 đồng"]},
    {"query": "thanh toán tự động trừ tiền ngày mấy",
     "answer_spans": ["tự động trừ tiền vào ngày 10 hàng tháng, thử lại tối đa 3 lần"],
     "stale_spans": ["tự động trừ tiền vào ngày 18 hàng tháng, thử lại tối đa 2 lần"]},
    {"query": "gói Combo 90 có bao nhiêu data",
     "answer_spans": ["Combo 90 có 90GB data tốc độ cao mỗi tháng"],
     "stale_spans": ["Combo 90 có 60GB data tốc độ cao mỗi tháng"]},
    {"query": "Combo 120 miễn phí gọi ngoại mạng bao nhiêu phút",
     "answer_spans": ["Combo 120 có 120GB data, miễn phí gọi ngoại mạng 50 phút"],
     "stale_spans": ["Combo 120 có 90GB data, miễn phí gọi ngoại mạng 30 phút"]},
    {"query": "hạn mức giao dịch qua ứng dụng mỗi lần",
     "answer_spans": ["Hạn mức mỗi lần giao dịch qua ứng dụng là 20.000.000 đồng"],
     "stale_spans": ["Hạn mức mỗi lần giao dịch qua ứng dụng là 10.000.000 đồng"]},
]

# Queries whose answer lives outside the retail scope — the filter must abstain
# rather than quietly serve the enterprise document.
DEMO_OUT_OF_SCOPE = [
    "khách hàng doanh nghiệp được miễn phí khôi phục không",
    "doanh nghiệp thanh toán hạn chót ngày nào",
]
