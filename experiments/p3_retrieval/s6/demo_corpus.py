"""S6 demo corpus — built to produce distractors on purpose.

Every topic has several neighbouring sections that are on-topic, well written
and do NOT answer the question: OTP for login vs OTP for payment, suspension for
non-payment vs suspension on request, Combo 90 vs Combo 90 Plus vs Combo 120.

That is the shape that matters for S6. A corpus of unrelated documents would
make context selection look pointless, because the top-k would be clean by
accident.
"""

DEMO_DOCS = [{
    "source": "faq_tong_hop.md",
    "text": """\
# Hỏi đáp tổng hợp

## Đổi mật khẩu tài khoản
Vào Cài đặt > Bảo mật > Đổi mật khẩu. Hệ thống gửi mã OTP về số điện thoại đã
đăng ký. Mật khẩu mới phải có tối thiểu 8 ký tự, gồm cả chữ và số.

## Quên mật khẩu
Chọn Quên mật khẩu ở màn hình đăng nhập, nhập số điện thoại và xác thực OTP.
Nếu số điện thoại không còn dùng được, cần đến điểm giao dịch.

## Đổi mã PIN SIM
Mã PIN SIM đổi ở mục Cài đặt > SIM, khác hoàn toàn với mật khẩu tài khoản ứng
dụng. Nhập PIN cũ rồi nhập PIN mới hai lần.

## Mã OTP khi đăng nhập
Mã OTP đăng nhập có hiệu lực 5 phút và chỉ dùng được một lần.

## Mã OTP khi thanh toán
Mã OTP thanh toán có hiệu lực 3 phút, gửi riêng cho từng giao dịch.

## Bảo mật hai lớp
Bật trong Cài đặt > Bảo mật > Xác thực hai lớp. Khi bật, mỗi lần đăng nhập trên
thiết bị mới đều cần mã OTP.

## Thời hạn thanh toán
Thời hạn thanh toán là ngày 15 hàng tháng. Hóa đơn phát hành ngày 1 hàng tháng.

## Tạm ngưng do chậm thanh toán
Quá hạn 7 ngày, dịch vụ bị tạm ngưng một chiều. Quá hạn 15 ngày, tạm ngưng hai
chiều và phát sinh phí khôi phục 20.000 đồng.

## Tạm ngưng theo yêu cầu khách hàng
Khách hàng có thể yêu cầu tạm ngưng dịch vụ tối đa 3 tháng mỗi năm, không mất
phí khôi phục.

## Khôi phục dịch vụ
Sau khi thanh toán toàn bộ dư nợ, dịch vụ được khôi phục trong vòng 2 giờ.

## Hình thức thanh toán
Thanh toán qua ứng dụng, chuyển khoản ngân hàng, điểm giao dịch hoặc ví điện tử.
Chuyển khoản ngân hàng ghi nhận sau 1 đến 2 giờ làm việc.

## Thanh toán tự động
Khi đã liên kết ngân hàng, hệ thống tự động trừ tiền vào ngày 10 hàng tháng.

## Tra cứu hóa đơn
Lịch sử hóa đơn nằm trong mục Hóa đơn của ứng dụng, lưu 24 kỳ gần nhất.

## Hóa đơn điện tử
Hóa đơn điện tử gửi qua email vào ngày 1 hàng tháng, định dạng PDF có chữ ký số.

## Khiếu nại hóa đơn
Gửi khiếu nại trong mục Hóa đơn. Hệ thống cấp mã khiếu nại trong 24 giờ làm việc.

## Gói cước Combo 90
Gói Combo 90 có 90GB data tốc độ cao mỗi tháng, miễn phí gọi nội mạng dưới
20 phút. Giá 90.000 đồng mỗi tháng.

## Gói cước Combo 90 Plus
Gói Combo 90 Plus có 120GB data và thêm 30 phút gọi ngoại mạng. Giá 120.000 đồng.

## Gói cước Combo 120
Gói Combo 120 có 120GB data, miễn phí gọi ngoại mạng 50 phút. Giá 120.000 đồng.

## Đăng ký đổi gói cước
Đổi gói trong mục Gói cước của ứng dụng, có hiệu lực từ chu kỳ tiếp theo.

## Thiết bị tin cậy
Mỗi tài khoản đăng ký tối đa 5 thiết bị tin cậy. Thêm thiết bị thứ 6 thì thiết
bị ít hoạt động nhất bị loại khỏi danh sách.

## Đăng xuất từ xa
Có thể đăng xuất thiết bị khác trong mục Cài đặt > Bảo mật > Thiết bị.
"""
}]

DEMO_EVAL = [
    {"query": "làm sao để đổi mật khẩu tài khoản",
     "answer_spans": ["Cài đặt > Bảo mật > Đổi mật khẩu"]},
    {"query": "mã OTP đăng nhập có hiệu lực bao lâu",
     "answer_spans": ["OTP đăng nhập có hiệu lực 5 phút"]},
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
    {"query": "được đăng ký mấy thiết bị tin cậy",
     "answer_spans": ["tối đa 5 thiết bị tin cậy"]},
    {"query": "hóa đơn điện tử gửi khi nào",
     "answer_spans": ["Hóa đơn điện tử gửi qua email vào ngày 1 hàng tháng"]},
    {"query": "thanh toán tự động trừ tiền ngày mấy",
     "answer_spans": ["tự động trừ tiền vào ngày 10 hàng tháng"]},
]
