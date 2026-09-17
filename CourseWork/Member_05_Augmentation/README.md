# Member 05 – Data Augmentation

Phần này kiểm tra ảnh hưởng của **data augmentation** đối với bài toán dự đoán tuổi bệnh nhân từ ảnh NIH ChestX-ray14.

- Input: ảnh X-quang grayscale `[B, 1, 224, 224]`
- Target: `Patient Age`
- Task: hồi quy
- Model chung: CNN → Global Average Pooling → Linear(1)
- Metric chính: MAE (năm)
- Seed mặc định: 42

## Cấu trúc thư mục

```text
Member_05_Augmentation/
├── README.md
├── requirements.txt
├── notebook/
│   └── 05_augmentation.ipynb
├── src/
│   ├── __init__.py
│   ├── transforms.py
│   ├── contracts.py
│   ├── integration.py
│   ├── reference_model.py
│   ├── experiments.py
│   ├── reporting.py
│   └── evaluate.py
├── figures/
│   └── README.md
├── results/
│   ├── README.md
│   ├── metadata_audit.json
│   └── experiment_results_template.csv
└── tests/
    └── test_augmentation.py
```

Cấu trúc này dùng cùng quy ước `notebook/` với Member 01, Member 02 và Member 03. Toàn bộ mã, kết quả và hình của Member 05 nằm trong thư mục này.

## Nội dung đã triển khai

### 1. Transform

Baseline giữ nguyên preprocessing của Member 02:

```text
Resize(224, 224) → ToTensor() → Normalize([0.5], [0.5])
```

Augmentation của E3/E4 chỉ áp dụng cho tập train:

```text
Resize(224, 224)
→ RandomAffine(rotation ±5°, translation tối đa 2%)
→ ToTensor()
→ Normalize([0.5], [0.5])
```

Không dùng vertical flip, crop mạnh hoặc biến dạng đàn hồi. Validation và test luôn dùng transform cố định, kể cả khi tham số `augment=True` được truyền nhầm.

### 2. Ma trận thí nghiệm

| ID | Model | Loss | Train augmentation | So sánh |
|---|---|---|---|---|
| E1 | CNN + GAP + Linear(1) | MSE | Không | Baseline cho E3 |
| E2 | Cùng model | MAE/L1 | Không | Baseline cho E4 |
| E3 | Cùng model | MSE | Có | E3 − E1 |
| E4 | Cùng model | MAE/L1 | Có | E4 − E2 |

Bốn thí nghiệm dùng cùng:

- patient-level train/validation/test split của Member 02;
- trọng số khởi tạo;
- thứ tự minibatch;
- optimizer Adam;
- learning rate, batch size, số epoch và seed;
- quy tắc chọn checkpoint bằng validation MAE thấp nhất.

Test set không được dùng để chọn augmentation hoặc checkpoint.

### 3. Kết quả đầu ra

Sau khi chạy đủ E1–E4, chương trình tạo:

```text
results/run_seed42/
├── manifest.json
├── validation_results.csv
├── augmentation_comparison.csv
├── discussion.md
├── E1/
│   ├── history.csv
│   ├── result.json
│   └── best_model.pt
├── E2/ ...
├── E3/ ...
├── E4/ ...
└── figures/
    ├── augmentation_examples.png
    ├── train_validation_loss.png
    ├── validation_mae.png
    └── validation_comparison.png
```

`ΔMAE = MAE_augmented − MAE_baseline`. Giá trị âm cho thấy augmentation làm validation MAE giảm. MAE và RMSE có đơn vị năm; MSE có đơn vị năm².

## Liên kết với phần của các thành viên khác

### Member 02 – Dataset

Chương trình đọc trực tiếp:

- `../Member_02_Data/data/processed/train.csv`
- `../Member_02_Data/data/processed/val.csv`
- `../Member_02_Data/data/processed/test.csv`
- file NIH `archive.zip`

Dataset được dùng lại từ `Member_02_Data/src/dataset.py`. Chương trình kiểm tra cột bắt buộc, tuổi hợp lệ, ảnh trùng và giao bệnh nhân giữa các split. Không tự chia lại dữ liệu.

Audit metadata hiện có:

| Split | Ảnh | Bệnh nhân | Tuổi |
|---|---:|---:|---:|
| Train | 78.831 | 21.561 | 1–95 |
| Validation | 16.383 | 4.620 | 1–91 |
| Test | 16.890 | 4.621 | 1–93 |

Patient ID, Image Index và zip_member không giao nhau giữa ba tập.

### Member 03 – CNN

Chương trình ưu tiên `../Member_03_CNN_Model/src/model.py`. File này chưa có trên `main` tại commit đã đối chiếu, nên `src/reference_model.py` lưu bản model từ nhánh `linh-brand`, commit `736cbbc34de4287c8e44dea14731a6834f715ad2`.

Model gồm bốn ConvBlock, Global Average Pooling và Linear(256,1), tổng cộng 389.057 tham số. Khi Member 03 bổ sung model chung, pipeline tự ưu tiên model đó và lưu fingerprint source vào manifest.

### Member 04 – Loss comparison

Notebook Member 04 hiện dùng dữ liệu hồi quy tổng hợp và MLP. Các metric đó không phải baseline NIH. Vì vậy Member 05 chạy lại E1 và E2 trên cùng NIH split, CNN và cấu hình với E3/E4 để bảo đảm so sánh công bằng.

### Member 06 – Evaluation

Member 05 chỉ chọn checkpoint bằng validation MAE. Sau khi cấu hình được chốt, Member 06 có thể dùng `src/evaluate.py` để tính test MAE, MSE và RMSE. File này kiểm tra fingerprint của split, model và checkpoint trước khi đọc ảnh test.

## Cài đặt

Từ thư mục gốc repository:

```bash
python -m pip install -r CourseWork/Member_05_Augmentation/requirements.txt
```

Cần Python 3.10+ và cặp phiên bản PyTorch/torchvision tương thích với máy đang chạy.

## Kiểm thử

```bash
python -m unittest discover \
  -s CourseWork/Member_05_Augmentation/tests \
  -v
```

Các test kiểm tra:

- không rò rỉ bệnh nhân/ảnh giữa các split;
- baseline khớp transform Member 02;
- validation/test deterministic;
- augmentation train có seed tái lập;
- output model là `[B,1]`;
- metric được cộng theo số mẫu;
- luồng E1–E4 chạy được trên fixture nhỏ.

Nếu thiếu PyTorch/torchvision, runtime tests sẽ hiển thị `SKIP`; trạng thái này chưa đạt yêu cầu kiểm thử cuối.

## Chạy notebook

Mở:

```text
CourseWork/Member_05_Augmentation/notebook/05_augmentation.ipynb
```

Chỉnh `ZIP_PATH` tới file NIH `archive.zip`, sau đó chạy từ trên xuống. Notebook sẽ:

1. xác minh split;
2. kiểm tra transform;
3. tạo hình trước/sau từ ảnh train thật;
4. kiểm tra model;
5. chạy E1–E4;
6. hiển thị bảng và biểu đồ so sánh.

Nếu không có `archive.zip`, notebook ghi rõ phần chưa chạy và không tạo metric giả.

## Chạy bằng dòng lệnh

```bash
python -m CourseWork.Member_05_Augmentation.src.experiments \
  --zip-path "/duong/dan/archive.zip" \
  --processed-dir CourseWork/Member_02_Data/data/processed \
  --output-dir CourseWork/Member_05_Augmentation/results/run_seed42
```

Smoke test có thể dùng `--epochs 1`; kết quả chính thức dùng cấu hình chung 10 epoch. Mỗi lượt chạy phải dùng một output directory mới để tránh trộn checkpoint và metric.

## Bàn giao test cho Member 06

Sau khi chốt recipe và checkpoint:

```bash
python -m CourseWork.Member_05_Augmentation.src.evaluate \
  --run-dir CourseWork/Member_05_Augmentation/results/run_seed42 \
  --zip-path "/duong/dan/archive.zip" \
  --processed-dir CourseWork/Member_02_Data/data/processed \
  --device cpu
```

Kết quả test được ghi vào `test_evaluation/test_results.csv`. Không thay đổi augmentation sau khi xem test.

## Trạng thái hiện tại

- Mã nguồn, notebook, cấu trúc output và kiểm thử đã được chuẩn bị.
- Metadata split thật đã được audit.
- Chưa có `archive.zip` trong repository nên chưa tạo hình augmentation, checkpoint hoặc metric NIH.
- Không có số liệu tổng hợp được trình bày như kết quả NIH.

Để hoàn tất phần thực nghiệm, cần chạy tests không còn runtime test bị skip, xem hình augmentation trên ảnh thật, chạy đủ E1–E4 và cập nhật bảng kết quả từ output thực tế.
