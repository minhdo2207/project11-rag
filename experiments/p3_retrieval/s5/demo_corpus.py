"""S5 demo corpus — the documents are clean; the queries are not.

S1–S4 used messy corpora and tidy questions. S5 inverts that: a well-formed FAQ
and the kind of input a real Vietnamese user types on a phone. The four query
categories exist because an aggregate recall number hides which mechanism is
doing the work.
"""

DEMO_DOCS = [{
    "source": "faq_dich_vu.md",
    "text": """\
# Hỏi đáp dịch vụ

## Đổi mật khẩu tài khoản
Vào Cài đặt > Bảo mật > Đổi mật khẩu. Hệ thống gửi mã OTP về số điện thoại đã
đăng ký, mã có hiệu lực trong 5 phút. Mật khẩu mới phải có tối thiểu 8 ký tự,
gồm cả chữ và số.

## Quên mật khẩu
Chọn Quên mật khẩu ở màn hình đăng nhập, nhập số điện thoại và xác thực OTP.
Nếu số điện thoại không còn dùng được, khách hàng cần đến điểm giao dịch.

## Thời hạn thanh toán và tạm ngưng dịch vụ
Thời hạn thanh toán là ngày 15 hàng tháng. Quá hạn 7 ngày, dịch vụ bị tạm ngưng
một chiều. Quá hạn 15 ngày, dịch vụ bị tạm ngưng hai chiều và phát sinh phí
khôi phục 20.000 đồng.

## Hình thức thanh toán
Khách hàng có thể thanh toán qua ứng dụng, chuyển khoản ngân hàng, điểm giao
dịch hoặc ví điện tử. Thanh toán qua chuyển khoản ngân hàng được ghi nhận sau
1 đến 2 giờ làm việc.

## Tra cứu hóa đơn
Lịch sử hóa đơn nằm trong mục Hóa đơn của ứng dụng, lưu trữ 24 kỳ gần nhất.
Hóa đơn điện tử được gửi qua email vào ngày 1 hàng tháng.

## Gói cước Combo 90
Gói Combo 90 có 90GB data tốc độ cao mỗi tháng, miễn phí gọi nội mạng dưới
20 phút. Giá 90.000 đồng mỗi tháng.

## Mã lỗi đăng nhập
Mã lỗi ERR_4012 nghĩa là phiên đăng nhập đã hết hạn, cần đăng nhập lại.
Mã lỗi ERR_4021 nghĩa là tài khoản bị khóa tạm thời, chờ 30 phút.

## Bảo mật hai lớp
Bật trong Cài đặt > Bảo mật > Xác thực hai lớp. Khi bật, mỗi lần đăng nhập trên
thiết bị mới đều cần mã OTP.

## Thiết bị tin cậy
Mỗi tài khoản được đăng ký tối đa 5 thiết bị tin cậy. Khi thêm thiết bị thứ 6,
thiết bị ít hoạt động nhất sẽ bị loại khỏi danh sách.
"""
}]

DEMO_EVAL = [
    # --- plain: câu hỏi chuẩn, có dấu, một ý ---
    {"query": "làm sao để đổi mật khẩu", "category": "plain",
     "answer_spans": ["Cài đặt > Bảo mật > Đổi mật khẩu"]},
    {"query": "thời hạn thanh toán là ngày nào", "category": "plain",
     "answer_spans": ["Thời hạn thanh toán là ngày 15 hàng tháng"]},
    {"query": "gói Combo 90 có bao nhiêu data", "category": "plain",
     "answer_spans": ["Combo 90 có 90GB data tốc độ cao mỗi tháng"]},
    {"query": "ERR_4021 là lỗi gì", "category": "plain",
     "answer_spans": ["ERR_4021 nghĩa là tài khoản bị khóa tạm thời"]},

    # --- noaccent: gõ không dấu, rất phổ biến trên điện thoại ---
    {"query": "lam sao doi mat khau", "category": "noaccent",
     "answer_spans": ["Cài đặt > Bảo mật > Đổi mật khẩu"]},
    {"query": "quen mat khau thi lam the nao", "category": "noaccent",
     "answer_spans": ["Chọn Quên mật khẩu ở màn hình đăng nhập"]},
    {"query": "thoi han thanh toan ngay nao", "category": "noaccent",
     "answer_spans": ["Thời hạn thanh toán là ngày 15 hàng tháng"]},
    {"query": "tra cuu hoa don o dau", "category": "noaccent",
     "answer_spans": ["Lịch sử hóa đơn nằm trong mục Hóa đơn của ứng dụng"]},
    {"query": "bao mat hai lop bat o dau", "category": "noaccent",
     "answer_spans": ["Cài đặt > Bảo mật > Xác thực hai lớp"]},

    # --- abbrev: viết tắt và cách nói thông tục ---
    {"query": "doi mk tk kieu gi", "category": "abbrev",
     "answer_spans": ["Cài đặt > Bảo mật > Đổi mật khẩu"]},
    {"query": "chậm đóng tiền bao lâu thì bị cắt mạng", "category": "abbrev",
     "answer_spans": ["Quá hạn 7 ngày, dịch vụ bị tạm ngưng"]},
    {"query": "xem hd o muc nao", "category": "abbrev",
     "answer_spans": ["Lịch sử hóa đơn nằm trong mục Hóa đơn của ứng dụng"]},
    {"query": "ck ngan hang bao lau thi vao", "category": "abbrev",
     "answer_spans": ["chuyển khoản ngân hàng được ghi nhận sau"]},

    # --- multipart: hai câu hỏi trong một ---
    {"query": "đổi mật khẩu thế nào và mật khẩu mới cần mấy ký tự",
     "category": "multipart",
     "answer_spans": ["Cài đặt > Bảo mật > Đổi mật khẩu",
                      "tối thiểu 8 ký tự"]},
    {"query": "thời hạn thanh toán ngày nào, còn phí khôi phục bao nhiêu",
     "category": "multipart",
     "answer_spans": ["Thời hạn thanh toán là ngày 15 hàng tháng",
                      "phí\nkhôi phục 20.000 đồng"]},
    {"query": "được đăng ký mấy thiết bị tin cậy và khi vượt thì sao",
     "category": "multipart",
     "answer_spans": ["tối đa 5 thiết bị tin cậy",
                      "thiết bị ít hoạt động nhất sẽ bị loại khỏi danh sách"]},
]
