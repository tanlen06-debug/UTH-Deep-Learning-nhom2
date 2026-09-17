# Trạng thái kết quả Member 05

**Chưa có kết quả huấn luyện/đánh giá NIH.** Không có archive.zip trong repository được truy cập. Môi trường không cài được PyTorch và sau đó mất kết nối; Python tests, forward pass và notebook chưa được chạy.

## Đã xác minh thực tế

Đọc đầy đủ ba Git blobs CSV từ main commit 4210d4f58de1edb14fc61f2105cebdec73e16d2f qua GitHub và kiểm tra bằng JavaScript trong phiên làm việc:

| Split | Số ảnh | Số bệnh nhân | Tuổi min–max |
|---|---:|---:|---:|
| Train | 78.831 | 21.561 | 1–95 |
| Validation | 16.383 | 4.620 | 1–91 |
| Test | 16.890 | 4.621 | 1–93 |

Không có giá trị bắt buộc rỗng, tuổi ngoài [1,100], tên ảnh/zip_member không khớp, ảnh/zip_member trùng trong mỗi split. Cả ba cặp split có patient/image/zip_member overlap bằng 0. Audit này chỉ chứng minh tính nhất quán metadata; không chứng minh ảnh tồn tại, đọc được hoặc đúng nội dung.

metadata_audit.json lưu Git blob SHA của từng CSV và các số đếm. Đây không phải kết quả chạy tests/test_augmentation.py. Khi chạy code, validate_splits sẽ kiểm tra lại file local và tạo SHA-256.

## Chưa xác minh

- Tests Python/PyTorch và notebook chạy từ đầu đến cuối.
- Đường dẫn ảnh trong ZIP, ảnh bị lỗi và chất lượng augmentation trên ảnh thật.
- MAE/MSE/RMSE của E1–E4, checkpoint và thời gian training.
- Hiệu quả tổng quát hóa của augmentation.

Không tạo bảng metric rỗng mang nhãn “completed”, không điền số liệu ước đoán hoặc kết quả dữ liệu tổng hợp vào báo cáo NIH. Sau khi có ảnh, lệnh trong README sẽ tạo đầy đủ artifacts trong một thư mục run mới.
