# Data Directory

Thư mục này chứa dữ liệu cho Knowledge Base của hệ thống RAG phát hiện PyPI packages độc hại.
Các file data không được lưu trong git do kích thước lớn.

## Cấu trúc thư mục
  data/
├── malicious/
│ ├── train_malicious_packages_final.json
│ └── test_malicious_packages_final.json
├── benign/
│ ├── train_benign_packages_final.json
│ └── test_benign_packages_final.json
├── yara/
│ └── sample-yara-rules-core.yar
└── ghsa_all.json 
  
# Data

Thư mục này không được lưu trong git. Chạy script sau để tải về:

```bash
pip install requests
python scripts/fetch_data.py
```

Điền GitHub Token vào dòng `GITHUB_TOKEN` trong script trước khi chạy.
