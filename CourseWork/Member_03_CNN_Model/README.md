# Member 03 – CNN Model

## Trách nhiệm
- Thiết kế CNN cho bài toán age regression.
- Bám sát yêu cầu: **CNN + Global Average Pooling + Linear output**.
- Xác định số convolution blocks, channels, activation, pooling và normalization.
- Kiểm tra tensor shapes từ input đến output.
- Viết model PyTorch có output đúng dạng `[batch_size, 1]`.

## Deliverables
- `model.py` hoặc notebook xây dựng model.
- Sơ đồ kiến trúc CNN.
- Model summary và số lượng parameters.
- Kiểm tra forward pass bằng dữ liệu mẫu.

## Acceptance criteria
`Input X-ray → CNN → Global Average Pooling → Linear(1) → Predicted Age`.
