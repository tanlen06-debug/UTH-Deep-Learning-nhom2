# Member 02 – Dataset & Preprocessing

## Trách nhiệm
- Chuẩn bị NIH ChestX-ray14 metadata.
- Liên kết `Image Index` với `Patient Age`.
- Kiểm tra missing values, giá trị tuổi bất thường và ảnh lỗi.
- Thực hiện split theo **Patient ID** để tránh data leakage.
- Tạo train/validation/test metadata.
- Xây dựng Dataset/DataLoader và image transforms cơ bản.

## Deliverables
- Notebook khám phá và preprocessing dữ liệu.
- File metadata đã xử lý cho train/validation/test.
- Code Dataset/DataLoader dùng lại được.
- Thống kê phân bố tuổi và số lượng mẫu.

## Acceptance criteria
- Không có patient xuất hiện ở nhiều tập.
- Input image có kích thước thống nhất.
- Target Age là numeric và hợp lệ.
