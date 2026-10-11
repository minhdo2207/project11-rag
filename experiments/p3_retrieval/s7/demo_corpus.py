"""S7 demo corpus — answerable and unanswerable questions in one set.

The unanswerable group is not a side case here; it is the half the whole
scenario is measured on. Three kinds, because they fail differently:

  absent     the topic is simply not in the corpus
  adjacent   the topic IS in the corpus but the specific fact asked for is not
             — the hardest kind, because retrieval returns good-looking chunks
  oos        the answer exists but belongs to a scope the filter excludes
             (the gap S4 left open)
"""

DEMO_DOCS = [{
    "source": "faq_tong_hop.md",
    "metadata": {"status": "current", "audience": "retail",
                 "effective_from": "2026-01-01"},
    "text": """\
# Hỏi đáp tổng hợp

## Đổi mật khẩu tài khoản
Vào Cài đặt > Bảo mật > Đổi mật khẩu. Hệ thống gửi mã OTP về số điện thoại đã
đăng ký. Mật khẩu mới phải có tối thiểu 8 ký tự, gồm cả chữ và số.

## Quên mật khẩu
Chọn Quên mật khẩu ở màn hình đăng nhập, nhập số điện thoại và xác thực OTP.

## Mã OTP khi đăng nhập
Mã OTP đăng nhập có hiệu lực 5 phút và chỉ dùng được một lần.

## Bảo mật hai lớp
Bật trong Cài đặt > Bảo mật > Xác thực hai lớp.

## Thời hạn thanh toán
Thời hạn thanh toán là ngày 15 hàng tháng. Hóa đơn phát hành ngày 1 hàng tháng.

## Tạm ngưng do chậm thanh toán
Quá hạn 7 ngày, dịch vụ bị tạm ngưng một chiều. Quá hạn 15 ngày, tạm ngưng hai
chiều và phát sinh phí khôi phục 20.000 đồng.

## Hình thức thanh toán
Thanh toán qua ứng dụng, chuyển khoản ngân hàng, điểm giao dịch hoặc ví điện tử.
Chuyển khoản ngân hàng ghi nhận sau 1 đến 2 giờ làm việc.

## Thanh toán tự động
Khi đã liên kết ngân hàng, hệ thống tự động trừ tiền vào ngày 10 hàng tháng.

## Tra cứu hóa đơn
Lịch sử hóa đơn nằm trong mục Hóa đơn của ứng dụng, lưu 24 kỳ gần nhất.

## Gói cước Combo 90
Gói Combo 90 có 90GB data tốc độ cao mỗi tháng, miễn phí gọi nội mạng dưới
20 phút. Giá 90.000 đồng mỗi tháng.

## Gói cước Combo 120
Gói Combo 120 có 120GB data, miễn phí gọi ngoại mạng 50 phút. Giá 120.000 đồng.

## Đăng ký đổi gói cước
Đổi gói trong mục Gói cước của ứng dụng, có hiệu lực từ chu kỳ tiếp theo.

## Thiết bị tin cậy
Mỗi tài khoản đăng ký tối đa 5 thiết bị tin cậy.
"""
}, {
    "source": "chinh_sach_doanh_nghiep.md",
    "metadata": {"status": "current", "audience": "enterprise",
                 "effective_from": "2026-01-01"},
    "text": """\
# Chính sách khách hàng doanh nghiệp

## Thời hạn thanh toán doanh nghiệp
Thời hạn thanh toán là ngày 25 hàng tháng, quá hạn 30 ngày mới tạm ngưng.

## Miễn phí khôi phục
Khách hàng doanh nghiệp được miễn phí khôi phục dịch vụ.
"""
}]

# --- answerable ------------------------------------------------------------ #
DEMO_ANSWERABLE = [
    {"query": "làm sao để đổi mật khẩu tài khoản",
     "answer_spans": ["Cài đặt > Bảo mật > Đổi mật khẩu"]},
    {"query": "mã OTP đăng nhập có hiệu lực bao lâu",
     "answer_spans": ["OTP đăng nhập có hiệu lực 5 phút"]},
    {"query": "thời hạn thanh toán là ngày nào",
     "answer_spans": ["Thời hạn thanh toán là ngày 15 hàng tháng"]},
    {"query": "chậm thanh toán bao lâu thì bị tạm ngưng",
     "answer_spans": ["Quá hạn 7 ngày, dịch vụ bị tạm ngưng một chiều"]},
    {"query": "phí khôi phục dịch vụ bao nhiêu",
     "answer_spans": ["phí khôi phục 20.000 đồng"]},
    {"query": "gói Combo 90 có bao nhiêu data",
     "answer_spans": ["Combo 90 có 90GB data tốc độ cao mỗi tháng"]},
    {"query": "Combo 120 miễn phí gọi ngoại mạng mấy phút",
     "answer_spans": ["Combo 120 có 120GB data, miễn phí gọi ngoại mạng 50 phút"]},
    {"query": "chuyển khoản ngân hàng bao lâu thì ghi nhận",
     "answer_spans": ["Chuyển khoản ngân hàng ghi nhận sau 1 đến 2 giờ làm việc"]},
    {"query": "thanh toán tự động trừ tiền ngày mấy",
     "answer_spans": ["tự động trừ tiền vào ngày 10 hàng tháng"]},
    {"query": "được đăng ký mấy thiết bị tin cậy",
     "answer_spans": ["tối đa 5 thiết bị tin cậy"]},
    {"query": "đổi gói cước có hiệu lực khi nào",
     "answer_spans": ["có hiệu lực từ chu kỳ tiếp theo"]},
    {"query": "tra cứu hóa đơn ở đâu",
     "answer_spans": ["Lịch sử hóa đơn nằm trong mục Hóa đơn của ứng dụng"]},
]

# --- unanswerable ---------------------------------------------------------- #
DEMO_UNANSWERABLE = [
    # absent: chủ đề không có trong corpus
    {"query": "chính sách bảo hành thiết bị đầu cuối thế nào", "kind": "absent"},
    {"query": "làm sao đăng ký dịch vụ truyền hình", "kind": "absent"},
    {"query": "tổng đài chăm sóc khách hàng số mấy", "kind": "absent"},
    {"query": "chuyển vùng quốc tế tính cước ra sao", "kind": "absent"},

    # adjacent: chủ đề CÓ trong corpus, nhưng dữ kiện được hỏi thì không
    {"query": "mã OTP thanh toán có hiệu lực bao lâu", "kind": "adjacent"},
    {"query": "gói Combo 90 có bao nhiêu phút gọi ngoại mạng", "kind": "adjacent"},
    {"query": "phí đổi gói cước giữa chu kỳ là bao nhiêu", "kind": "adjacent"},
    {"query": "hóa đơn lưu được bao nhiêu năm", "kind": "adjacent"},
    {"query": "mật khẩu có hạn sử dụng bao lâu thì phải đổi", "kind": "adjacent"},

    # oos: câu trả lời tồn tại nhưng thuộc phạm vi bộ lọc loại ra (khoảng trống S4)
    {"query": "doanh nghiệp thanh toán hạn chót ngày nào", "kind": "oos"},
    {"query": "khách hàng doanh nghiệp có mất phí khôi phục không", "kind": "oos"},
]
