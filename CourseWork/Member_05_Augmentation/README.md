# Member 05 – Data Augmentation

Hoàn thiện phần thiết kế và triển khai augmentation cho **CNN-Based Patient Age Regression from NIH ChestX-ray14**, theo [master plan](../00_COURSEWORK_PLAN.md) và [member tasks](../01_MEMBER_TASKS.md).

**Trạng thái:** đã chuẩn bị code, notebook, kiểm thử và hướng dẫn; đã kiểm tra metadata thật. **Chưa chạy huấn luyện NIH, chưa có checkpoint/ảnh augmentation/số liệu MAE thực nghiệm.** Không có archive.zip trong repository được cung cấp; môi trường soạn thảo không cài được PyTorch và sau đó mất kết nối, nên Python/runtime tests chưa được thực thi. Không coi phần thực nghiệm là hoàn thành trước khi chạy trên dữ liệu thật.

## Đối chiếu với coursework

| Yêu cầu | Triển khai |
|---|---|
| Target Patient Age, regression | Giữ nhãn tuổi bằng năm, tensor [B,1] |
| Dữ liệu Member 02 | Đọc nguyên train.csv, val.csv, test.csv; dùng ZipChestXrayAgeDataset |
| Không rò rỉ bệnh nhân | Kiểm tra Patient ID, tên ảnh và zip_member giữa mọi cặp split |
| CNN → GAP → Linear(1) | Dùng model Member 03, đầu vào [B,1,224,224] |
| Train-only augmentation | Rotation ±5°, translation tối đa 2%; không lật ảnh |
| Validation/test cố định | Resize → ToTensor → Normalize; không random transform |
| E3/E4 so với E1/E2 | Một runner chạy đủ bốn thí nghiệm cùng recipe |
| Hình trước/sau | Tự sinh từ ảnh train thật khi có ZIP |
| Kết quả và bàn giao | History, validation metrics, paired comparisons, figures, checkpoint, manifest |

Các file mới chỉ thuộc thư mục Member 05.

## Cấu trúc

- notebooks/05_augmentation.ipynb: giải thích bằng tiếng Việt, cấu hình, kiểm tra, minh họa, chạy và đọc kết quả.
- src/transforms.py: baseline và augmentation.
- src/contracts.py: kiểm tra split và fingerprint CSV.
- src/integration.py: kết nối Dataset Member 02 và CNN Member 03.
- src/reference_model.py: bản sao nguyên văn model Member 03, có nguồn cụ thể bên dưới.
- src/experiments.py: chạy E1–E4 công bằng, checkpoint theo validation MAE.
- src/reporting.py: hình trước/sau, learning curves, bảng chênh lệch, nhận xét theo kết quả thực.
- src/evaluate.py: tiện ích tùy chọn để Member 06 đánh giá các checkpoint đã chốt.
- tests/test_augmentation.py: 15 kiểm thử, gồm kiểm tra luồng với ảnh tổng hợp trong thư mục tạm.
- results/metadata_audit.json: audit thực tế của CSV trên main.
- results/README.md: trạng thái kết quả và định dạng đầu ra.

## Tích hợp với các thành viên

Đã đối chiếu main tại commit **4210d4f58de1edb14fc61f2105cebdec73e16d2f**.

**Member 02:** CSV có các cột Image Index, Patient ID, Patient Age, zip_member. Ảnh được đọc trực tiếp từ ZIP và chuyển grayscale bởi Dataset hiện có. Không tự chia lại dữ liệu hoặc lấy một subset để làm kết quả chính thức.

**Member 03:** notebook trên main import src/model.py nhưng file đó chưa có trên main ở commit đã kiểm tra. Bản model đầy đủ nằm tại [linh-brand, commit 736cbbc3](https://github.com/tanlen06-debug/UTH-Deep-Learning-nhom2/blob/736cbbc34de4287c8e44dea14731a6834f715ad2/CourseWork/Member_03_CNN_Model/src/model.py), Git blob **f1d3cefec6808d9b4c70fa664c17744b3dcc9bca**. reference_model.py giữ nguyên nội dung đó: bốn ConvBlock 1→32→64→128→256, GAP, Linear(256,1), 389.057 tham số. Khi file chung xuất hiện, integration.py ưu tiên file chung. Khi dùng snapshot, chương trình cảnh báo rõ và lưu hash của source; không âm thầm đổi kiến trúc. Nhóm cần chốt model trước chạy cuối.

**Member 04:** notebook hiện tại trên main minh họa hồi quy y=3x+2+noise với MLP. Số liệu đó không phải baseline NIH. Runner ở đây chạy lại E1/E2 trên cùng NIH split, CNN và recipe với E3/E4. Không nhập hay trộn bảng kết quả giả lập với kết quả X-quang. Khi nhóm có baseline NIH chính thức, cần thống nhất recipe trước khi chạy ma trận này.

**Member 01:** mặc định seed=42, image_size=224, batch_size=32, learning_rate=0.001, num_epochs=10, weight_decay=0.0001 theo CONFIG trong notebook project overview. Chọn Adam, không scheduler và không early stopping; áp dụng như nhau cho cả bốn thí nghiệm. Không coi các lựa chọn này đã được cả nhóm phê duyệt chỉ vì code có giá trị mặc định.

## Thiết kế augmentation

Baseline khớp default_transform của Member 02:

1. Resize về 224×224.
2. ToTensor, pixel về [0,1].
3. Normalize(mean=[0.5], std=[0.5]), đầu ra [-1,1].

E3/E4 thêm RandomAffine sau Resize: xoay ±5°, dịch chuyển tối đa 2% theo mỗi trục, bilinear, fill=0. Không đổi nhãn tuổi, không vertical/horizontal flip, không crop mạnh, không biến dạng đàn hồi. Cấu hình này nhằm thay đổi hình học ở mức nhỏ; cần xem ảnh thật để xác nhận vùng phổi không bị cắt bất hợp lý. Không khẳng định mọi ảnh sẽ giữ nguyên thông tin hữu ích chỉ dựa vào biên độ biến đổi.

build_transform("val", augment=True) và build_transform("test", augment=True) vẫn trả về baseline deterministic.

## Tái lập

Chạy lệnh từ **repository root**, dùng Python 3.10+ và một cặp PyTorch/torchvision tương thích. requirements kế thừa requirements.txt của nhóm; manifest lưu phiên bản runtime khi chạy. Với GPU, cài bản PyTorch phù hợp môi trường trước nếu cần.

~~~bash
python -m pip install -r CourseWork/Member_05_Augmentation/requirements.txt
python -m unittest discover -s CourseWork/Member_05_Augmentation/tests -v
~~~

Nếu thiếu torch/torchvision, 7 runtime tests sẽ được đánh dấu SKIP, không phải PASS. 8 metadata tests chỉ cần Python chuẩn. Kiểm thử luồng dùng ảnh tổng hợp chỉ để kiểm tra code; các ảnh và metric đó bị xóa cùng thư mục tạm, không đưa vào kết quả NIH.

Đặt archive.zip đúng vị trí Member 02 hoặc truyền đường dẫn thật:

~~~bash
python -m CourseWork.Member_05_Augmentation.src.experiments \
  --zip-path "/duong/dan/archive.zip" \
  --output-dir CourseWork/Member_05_Augmentation/results/run_seed42
~~~

Hoặc mở notebooks/05_augmentation.ipynb, chỉnh ZIP_PATH và chạy lần lượt các cell. Notebook tìm repository root từ vị trí hiện tại nên dùng được khi mở ở root hoặc thư mục notebooks. Khi thiếu ZIP/thư viện, notebook ghi rõ các bước chưa chạy; không phát sinh số liệu NIH giả.

Có thể thêm --epochs 1 để kiểm tra luồng trên dữ liệu thật, nhưng đó chưa phải lượt chạy chính thức 10 epochs. Cần một output directory mới cho mỗi lượt chạy; runner từ chối ghi đè hoặc trộn kết quả cũ. Mặc định num_workers=0 phù hợp cách đọc ZIP của Member 02.

Chạy đủ bốn mô hình trên toàn bộ dữ liệu có thể tốn nhiều thời gian; nên dùng GPU. Không có số liệu thời gian huấn luyện ước đoán trong bài nộp này.

## Điều kiện so sánh công bằng

| Thuộc tính | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| Loss | MSE | MAE/L1 | MSE | MAE/L1 |
| Augmentation train | Không | Không | Có | Có |
| CNN, state_dict khởi tạo | Giống nhau | Giống nhau | Giống nhau | Giống nhau |
| Split, optimizer, LR, batch, epochs | Giống nhau | Giống nhau | Giống nhau | Giống nhau |
| Validation/test transform | Baseline | Baseline | Baseline | Baseline |

Sampler có generator riêng cùng seed, không bị lệch thứ tự do random affine. Reset seed trước mỗi thí nghiệm, lưu chung initial-state hash. Metric tích lũy theo số mẫu, kể cả minibatch cuối không đủ batch size. Kiểm tra output và target đều [B,1] để tránh broadcasting sai.

Checkpoint tốt nhất chọn bằng validation MAE; hòa thì giữ epoch sớm hơn. Train vẫn chạy đủ epochs. Cùng seed không bảo đảm bitwise giống nhau giữa mọi phiên bản PyTorch, GPU hoặc nền tảng; cần lưu manifest và môi trường để đối chiếu.

## Kết quả được tạo khi chạy thật

- manifest.json: cấu hình, version, fingerprint CSV/model/trọng số khởi tạo, trạng thái chạy.
- E1…E4/history.csv: epoch, train_loss/mae/mse/rmse, val_loss/mae/mse/rmse.
- E1…E4/best_model.pt: plain state_dict của checkpoint có validation MAE thấp nhất.
- E1…E4/result.json: best_epoch, validation metrics, thời gian chạy, checkpoint hash.
- validation_results.csv: bảng bốn thí nghiệm.
- augmentation_comparison.csv: E3–E1 và E4–E2, ΔMAE và cải thiện tương đối.
- figures/augmentation_examples.png: ảnh train trước/sau.
- figures/train_validation_loss.png, validation_mae.png, validation_comparison.png.
- discussion.md: nhận xét tự điền từ kết quả thật, nêu rõ giới hạn một seed.

ΔMAE = MAE_augmented − MAE_baseline; số âm là cải thiện trên validation. MAE/RMSE tính bằng năm, MSE bằng năm². Không vẽ MSE và L1 như cùng một đơn vị, không dùng classification accuracy.

## Bàn giao và đánh giá cuối

Member 05 không dùng test để chọn augmentation. Sau khi chốt toàn bộ recipe/checkpoint, Member 06 có thể chạy:

~~~bash
python -m CourseWork.Member_05_Augmentation.src.evaluate \
  --run-dir CourseWork/Member_05_Augmentation/results/run_seed42 \
  --zip-path "/duong/dan/archive.zip" \
  --processed-dir CourseWork/Member_02_Data/data/processed \
  --device cpu
~~~

Tiện ích kiểm tra source model, CSV và checkpoint chưa thay đổi, chỉ đọc checkpoint đã chốt và xuất test_evaluation/test_results.csv (MAE, MSE, RMSE). Không tự gọi từ notebook huấn luyện. Nếu thư mục test_evaluation đã tồn tại, từ chối chạy lại âm thầm; nếu lần chạy bị ngắt thì kiểm tra status.json và nguyên nhân trước khi dọn thư mục đó để phục hồi cùng checkpoint đã chốt. Không thay recipe theo kết quả test.

Bàn giao nguyên thư mục run cùng hình minh họa. Member 06 tiếp tục actual-vs-predicted và residual plots; Member 01 tích hợp final notebook. Checkpoint lớn và dữ liệu NIH được ignore, không commit lên repo; chọn commit các CSV/ảnh/báo cáo thật cần nộp sau khi kiểm tra (results/run*/ hiện được ignore để tránh đẩy toàn bộ lượt chạy).

## Còn cần hoàn tất trước khi nộp

- [ ] Chạy 15 tests; bảo đảm không có runtime tests bị skip.
- [ ] Nhóm chốt CNN/recipe và kiểm tra hình augmentation trên ảnh NIH thật.
- [ ] Chạy E1–E4, giữ lại logs/checkpoints và cập nhật bảng kết quả thật.
- [ ] Viết nhận xét dựa trên số liệu, không mặc định augmentation tốt hơn.
- [ ] Member 06 đánh giá test sau khi chốt mô hình.
- [ ] Member 01 review PR và tích hợp.
