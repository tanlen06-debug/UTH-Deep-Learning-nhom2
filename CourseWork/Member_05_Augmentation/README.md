# Member 05 – Data Augmentation

## Trách nhiệm
- Thiết kế augmentation cho ảnh chest X-ray.
- Tạo pipeline transform cho training và giữ validation/test không augmentation ngẫu nhiên.
- Thử các augmentation hợp lý như rotation nhẹ, horizontal flip hoặc affine nhẹ.
- So sánh kết quả có và không có augmentation.
- Kiểm tra augmentation không tạo ảnh phi thực tế.

## Deliverables
- Notebook augmentation experiments.
- Code transforms.
- Một số ảnh trước/sau augmentation để minh họa.
- Bảng so sánh performance với baseline.

## Acceptance criteria
Augmentation phải hợp lý đối với medical imaging và chỉ áp dụng ngẫu nhiên cho training set.
