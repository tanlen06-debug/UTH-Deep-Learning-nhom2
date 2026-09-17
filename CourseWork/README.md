# CourseWork – CNN Medical Image Regression

## Đề tài
**CNN-Based Patient Age Regression from NIH ChestX-ray14**

## Yêu cầu chính
- Model: CNN + Global Average Pooling + Linear output
- Task: Regression
- Dataset: NIH ChestX-ray14
- Target: Patient Age
- Extension 1: So sánh MSE Loss và MAE Loss
- Extension 2: Thử Data Augmentation
- Metrics: MAE, MSE, RMSE

## Phân công 6 thành viên

| Thành viên | Vai trò | Thư mục |
|---|---|---|
| Member 01 | Nhóm trưởng – tích hợp, quản lý repo, thiết kế pipeline, kiểm tra cuối | `Member_01_Group_Leader/` |
| Member 02 | Dataset & preprocessing – metadata, cleaning, patient-level split, DataLoader | `Member_02_Data/` |
| Member 03 | CNN architecture – CNN + GAP + Linear regression model | `Member_03_CNN_Model/` |
| Member 04 | Loss experiments – MSE vs MAE, training comparison | `Member_04_Loss_Comparison/` |
| Member 05 | Data augmentation – thiết kế và đánh giá augmentation | `Member_05_Augmentation/` |
| Member 06 | Evaluation & report – metrics, plots, analysis, report/presentation | `Member_06_Evaluation_Report/` |

## Quy tắc làm việc
1. Mỗi thành viên làm việc chủ yếu trong thư mục được phân công.
2. Không commit dataset NIH ChestX-ray14 dung lượng lớn vào repository.
3. Dùng Git branch riêng cho từng thành viên khi phát triển code.
4. Commit rõ ràng, ví dụ: `feat: add patient-level data split`.
5. Thành viên hoàn thành tạo Pull Request về `main` để nhóm trưởng kiểm tra.
6. Nhóm trưởng chịu trách nhiệm tích hợp và kiểm tra notebook cuối cùng.

## Pipeline chung
`NIH ChestX-ray14 → Metadata/Cleaning → Patient-level Split → Image Transform → CNN → Global Average Pooling → Linear(1) → Age Prediction → MAE/MSE/RMSE → Experiments → Analysis`
