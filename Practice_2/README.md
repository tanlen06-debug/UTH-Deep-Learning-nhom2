# Practice 2 — Pre-trained Neural Network Architectures

Thư mục này chứa toàn bộ nội dung Practice 2. Nhóm đã có branch riêng cho từng thành viên nên không cần tạo lại branch.

## Cấu trúc

```text
Practice_2/
├── configs/       # Cấu hình chung và cấu hình thí nghiệm
├── data/          # Dữ liệu cục bộ, không đưa lên GitHub
├── src/           # Mã nguồn dùng chung
├── notebooks/     # Notebook chính thức của từng phần
├── members/       # Khu vực ghi chú/output riêng của 6 thành viên
├── results/       # Bảng và hình kết quả dùng chung
├── checkpoints/   # Trọng số mô hình, không đưa lên GitHub
├── runs/          # TensorBoard logs, không đưa lên GitHub
└── docs/          # Phân công và quy tắc làm việc
```

## Quy tắc

1. Mỗi thành viên chỉ chỉnh sửa file được phân công.
2. Không đổi tên thư mục hoặc di chuyển file khi chưa báo nhóm trưởng.
3. Không dùng đường dẫn tuyệt đối theo máy cá nhân.
4. Không commit dữ liệu, checkpoint hoặc TensorBoard log lớn.
5. Notebook phải chạy lại được từ đầu đến cuối trước khi tạo Pull Request.
6. Kết quả cuối cùng được tổng hợp vào `results/tables/metrics.csv`.
