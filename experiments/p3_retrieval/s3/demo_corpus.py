"""Shared demo corpus (lifted from S2) — Vietnamese FAQ documents with
tables, numbered procedures and near-miss sections that make a hard-negative
heavy eval possible without any download."""

DEMO_DOCS = [{
    "source": "faq_thanh_toan.md",
    "text": """\
# Thanh toán và hóa đơn

## Giới thiệu chung
Tài liệu này mô tả toàn bộ quy định về thanh toán, hóa đơn và xử lý khiếu nại
dành cho khách hàng cá nhân. Các quy định áp dụng từ ngày 01/01/2026 và thay thế
cho các văn bản trước đó. Khách hàng doanh nghiệp tham khảo tài liệu riêng.
Mọi thay đổi về chính sách được thông báo trước 30 ngày qua SMS và ứng dụng.
Trường hợp có khác biệt giữa tài liệu này và hợp đồng đã ký, hợp đồng có hiệu lực
cao hơn. Khách hàng nên kiểm tra phiên bản mới nhất trong mục Trợ giúp.

## Thời hạn thanh toán
Hóa đơn được phát hành vào ngày 1 hàng tháng cho kỳ sử dụng của tháng trước.
Thời hạn thanh toán là ngày 15 hàng tháng. Nếu quá hạn 7 ngày, dịch vụ sẽ bị
tạm ngưng một chiều, nghĩa là khách hàng vẫn nhận được cuộc gọi nhưng không gọi
đi được. Quá hạn 15 ngày, dịch vụ bị tạm ngưng hai chiều và phát sinh phí khôi
phục 20.000 đồng. Quá hạn 60 ngày, hợp đồng bị chấm dứt và số điện thoại được
thu hồi về kho số sau 30 ngày tiếp theo.

## Các hình thức thanh toán
| Hình thức | Phí giao dịch | Thời gian ghi nhận | Hạn mức mỗi lần |
| --- | --- | --- | --- |
| Ứng dụng My Viettel | Miễn phí | Tức thì | 20.000.000 đồng |
| Chuyển khoản ngân hàng | Theo ngân hàng | 1-2 giờ | Theo ngân hàng |
| Điểm giao dịch Viettel | Miễn phí | Tức thì | Không giới hạn |
| Ví điện tử liên kết | Miễn phí | Tức thì | 10.000.000 đồng |
| Thẻ tín dụng quốc tế | 1,1% giá trị | Tức thì | 50.000.000 đồng |
| Ủy nhiệm thu ngân hàng | Miễn phí | Trong ngày | Không giới hạn |
| Cửa hàng tiện lợi đối tác | 3.000 đồng | Tức thì | 5.000.000 đồng |
| Bưu cục VNPost | 5.000 đồng | 1 ngày làm việc | 10.000.000 đồng |

## Quy trình khiếu nại hóa đơn
1. Đăng nhập ứng dụng My Viettel và vào mục Hóa đơn, chọn kỳ hóa đơn cần khiếu nại.
2. Chọn Khiếu nại, điền lý do cụ thể và đính kèm ảnh chứng từ nếu có.
3. Hệ thống cấp mã khiếu nại trong vòng 24 giờ làm việc và gửi qua SMS.
4. Bộ phận chăm sóc khách hàng liên hệ xác minh trong vòng 2 ngày làm việc.
5. Kết quả xử lý được gửi qua SMS trong vòng 5 ngày làm việc kể từ ngày xác minh.
6. Nếu khiếu nại được chấp nhận, số tiền được hoàn vào kỳ hóa đơn tiếp theo.
7. Nếu không đồng ý với kết quả, khách hàng có thể yêu cầu phúc tra một lần duy nhất.
8. Kết quả phúc tra là kết quả cuối cùng và được gửi trong vòng 10 ngày làm việc.

## Thanh toán tự động
Khi đã liên kết ngân hàng, hệ thống tự động trừ tiền vào ngày 10 hàng tháng.
Nếu số dư không đủ, hệ thống thử lại tối đa 3 lần trong 3 ngày liên tiếp.
Sau 3 lần thất bại, thanh toán tự động bị tạm dừng và khách hàng cần thanh toán
thủ công. Để bật lại, vào Cài đặt > Thanh toán > Liên kết ngân hàng.

## Hoàn tiền
Trường hợp thanh toán trùng, số tiền thừa được giữ lại dưới dạng số dư và tự động
trừ vào kỳ hóa đơn tiếp theo. Khách hàng muốn nhận lại bằng tiền mặt cần đến điểm
giao dịch kèm giấy tờ tùy thân và chứng từ thanh toán.
"""
}, {
    "source": "faq_tai_khoan.md",
    "text": """\
# Tài khoản và bảo mật

## Phạm vi áp dụng
Phần này áp dụng cho tài khoản ứng dụng My Viettel trên cả nền tảng iOS và
Android, cũng như cổng web tự phục vụ. Một số tính năng bảo mật nâng cao chỉ có
trên ứng dụng di động do yêu cầu về thiết bị tin cậy. Khách hàng sử dụng nhiều
thiết bị cần xác thực riêng cho từng thiết bị trong lần đăng nhập đầu tiên.

## Đổi mật khẩu
Vào Cài đặt > Bảo mật > Đổi mật khẩu. Hệ thống gửi mã OTP về số điện thoại đã
đăng ký, mã có hiệu lực trong 5 phút. Mật khẩu mới phải có tối thiểu 8 ký tự,
gồm chữ và số, và không được trùng với 3 mật khẩu gần nhất. Sau khi đổi thành
công, toàn bộ phiên đăng nhập trên các thiết bị khác sẽ bị đăng xuất.

## Quên mật khẩu
Chọn Quên mật khẩu ở màn hình đăng nhập, nhập số điện thoại và xác thực OTP.
Nếu số điện thoại không còn sử dụng được, khách hàng cần đến điểm giao dịch kèm
giấy tờ tùy thân. Lưu ý: việc này khác với đổi mã PIN SIM, vốn thực hiện ở mục
Cài đặt > SIM và không liên quan đến mật khẩu tài khoản ứng dụng.

## Mã lỗi đăng nhập
| Mã lỗi | Ý nghĩa | Cách xử lý | Thời gian chờ |
| --- | --- | --- | --- |
| ERR_4010 | Sai mật khẩu | Nhập lại mật khẩu | Không |
| ERR_4012 | Phiên đăng nhập hết hạn | Đăng nhập lại | Không |
| ERR_4015 | Thiết bị không được tin cậy | Xác thực OTP | Không |
| ERR_4018 | Sai mã OTP quá 5 lần | Yêu cầu mã mới | 10 phút |
| ERR_4021 | Tài khoản bị khóa tạm thời | Chờ rồi thử lại | 30 phút |
| ERR_4030 | Tài khoản bị khóa vĩnh viễn | Đến điểm giao dịch | Không |
| ERR_5003 | Máy chủ quá tải | Thử lại sau | 5 phút |
| ERR_5010 | Lỗi kết nối hệ thống cước | Thử lại sau | 15 phút |

## Bảo mật hai lớp
Bật trong Cài đặt > Bảo mật > Xác thực hai lớp. Khi bật, mỗi lần đăng nhập trên
thiết bị mới đều cần mã OTP, kể cả khi mật khẩu đúng. Khách hàng nên bật tính
năng này nếu tài khoản có liên kết ví điện tử hoặc thanh toán tự động.

## Thiết bị tin cậy
Danh sách thiết bị tin cậy nằm ở Cài đặt > Bảo mật > Thiết bị. Mỗi tài khoản
được đăng ký tối đa 5 thiết bị tin cậy. Khi thêm thiết bị thứ 6, thiết bị ít
hoạt động nhất sẽ tự động bị loại bỏ khỏi danh sách.
"""
}]

DEMO_EVAL = [
    # span nhiều câu, nằm vắt qua ranh giới cửa sổ 700 ký tự
    {"query": "khi nào bị cắt dịch vụ nếu chậm đóng tiền",
     "answer_spans": ["quá hạn 7 ngày, dịch vụ sẽ bị tạm ngưng một chiều",
                      "Quá hạn 15 ngày, dịch vụ bị tạm ngưng hai chiều và phát sinh phí khôi"]},
    # span trong bảng dài — fixed-size gần như chắc chắn cắt giữa bảng
    {"query": "các cách thanh toán và phí giao dịch",
     "answer_spans": ["Thẻ tín dụng quốc tế | 1,1% giá trị | Tức thì | 50.000.000 đồng",
                      "Cửa hàng tiện lợi đối tác | 3.000 đồng | Tức thì | 5.000.000 đồng"]},
    # quy trình 8 bước — bị chia đôi thì mất nửa câu trả lời
    {"query": "quy trình khiếu nại hóa đơn gồm những bước nào",
     "answer_spans": ["Hệ thống cấp mã khiếu nại trong vòng 24 giờ làm việc",
                      "Kết quả phúc tra là kết quả cuối cùng và được gửi trong vòng 10 ngày làm việc"]},
    {"query": "thanh toán tự động trừ tiền ngày nào và thử lại mấy lần",
     "answer_spans": ["tự động trừ tiền vào ngày 10 hàng tháng",
                      "thử lại tối đa 3 lần trong 3 ngày liên tiếp"]},
    {"query": "thanh toán trùng thì xử lý thế nào",
     "answer_spans": ["số tiền thừa được giữ lại dưới dạng số dư và tự động"]},
    {"query": "làm sao đổi mật khẩu và điều kiện mật khẩu mới",
     "answer_spans": ["Cài đặt > Bảo mật > Đổi mật khẩu",
                      "tối thiểu 8 ký tự,\ngồm chữ và số, và không được trùng với 3 mật khẩu gần nhất"]},
    {"query": "ERR_4021 nghĩa là gì",
     "answer_spans": ["ERR_4021 | Tài khoản bị khóa tạm thời | Chờ rồi thử lại | 30 phút"]},
    {"query": "sai OTP nhiều lần thì chờ bao lâu",
     "answer_spans": ["ERR_4018 | Sai mã OTP quá 5 lần | Yêu cầu mã mới | 10 phút"]},
    {"query": "được đăng ký tối đa bao nhiêu thiết bị tin cậy",
     "answer_spans": ["tối đa 5 thiết bị tin cậy"]},
]
