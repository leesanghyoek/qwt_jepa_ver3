# Dataset Kaggle: TartanAir V2 — đối chiếu với loader

Nguồn số liệu: bản audit bạn chạy trên chính dataset Kaggle của mình.
Phần "Đối chiếu" bên dưới là kiểm chứng bằng code trong repo này.

## 1. Kết luận

**Dataset tương thích với loader hiện tại, không cần sửa code loader.** Layout,
tên file IMU, tần số và kích thước ảnh đều khớp thứ mà `qjepa/data/tartanair.py`
đang tìm. Đã dựng một bản sao đúng cấu trúc này ở local và chạy trọn
`build-manifest → train-phase1 → train-phase2 → evaluate` thành công.

Dùng [`configs/kaggle_tartanair_v2.yaml`](configs/kaggle_tartanair_v2.yaml) —
config này chỉ đổi đường dẫn/thiết bị/worker so với `pipeline_v3.yaml`.

## 2. Quy mô

| Hạng mục | Giá trị |
| --- | --- |
| Environment | 14 |
| Trajectory | 166 (`Data_easy` 83, `Data_hard` 83) |
| Ảnh | 256×256, RGB |
| IMU | 100 Hz, `acc`+`gyro`, mỗi file `[N,3]` |
| Camera | 10 Hz, `cam_time` khớp số ảnh |
| Trọng lực | **Còn trong `acc.npy`** (\|mean acc\| ≈ 9,14 m/s²); không dùng `acc_nograv` |

Environment: AmericanDiner, ArchVizTinyHouseDay, CoalMine, CountryHouse,
DesertGasStation, HQWesternSaloon, HongKong, House, Office, OldTownNight,
OldTownSummer, RetroOffice, VictorianStreet, WaterMillDay.

Ba trajectory đọc thử, cả ba đều `images == len(cam_time)`:

| Trajectory | IMU rows | Ảnh | IMU Hz | Cam Hz |
| --- | --- | --- | --- | --- |
| AmericanDiner/Data_easy/P000 | 2830 | 284 | 100 | 10 |
| DesertGasStation/Data_hard/P002 | 5140 | 515 | 100 | 10 |
| Office/Data_hard/P003 | 3900 | 391 | 100 | 10 |

## 3. Đối chiếu từng điểm với loader

| Loader cần | Dataset có | Kết quả |
| --- | --- | --- |
| `<root>/<env>/<difficulty>/<Pxxx>/` (3 tầng) | `AmericanDiner/Data_easy/P000` | Khớp |
| `<Pxxx>/image_lcam_front/*_lcam_front.png` | Có | Khớp |
| `<Pxxx>/imu/{acc,gyro,imu_time,cam_time}` `.npy` ưu tiên hơn `.txt` | `.npy` đủ 166/166 | Khớp |
| `acc`/`gyro` dạng `[N,3]`, timestamp tăng nghiêm ngặt | Đúng | Khớp |
| `len(ảnh) == len(cam_time)` mỗi trajectory | Đúng ở 3 mẫu đọc thử | Khớp |
| Ảnh 256×256 cho `run_kind: main` | 256×256 | Khớp |
| `imu_window: 128` cần ≥128 hàng IMU | Thấp nhất 2830 | Khớp |

Loader **bỏ qua** các file TartanAir V2 khác trong `imu/` (`acc_nograv.npy`,
`vel_body.npy`, …) nên chúng không gây lỗi.

## 4. Split được sinh thế nào

Không có thư mục `train/valid/test` trong dataset, nên manifest tự chia theo
`(environment, trajectory_id)` — tức **83 motion key**, không phải 166 trajectory.
`Data_easy` và `Data_hard` của cùng một chuyển động **luôn nằm cùng split**, nên
không rò rỉ giữa train và test. Với tỉ lệ mặc định 0,8/0,1/0,1 trên 83 key:

- train ≈ 67 motion key (≈ 134 trajectory)
- valid ≈ 8 motion key (≈ 16 trajectory)
- test ≈ 8 motion key (≈ 16 trajectory)

Số **sample** (mỗi sample = 1 ảnh + 128 hàng IMU) do cell 5 in ra; ước lượng theo
ba trajectory đọc thử là khoảng 6 vạn sample, trong đó train khoảng 5 vạn.

## 5. `rejected_centre ≈ 13` mỗi trajectory là BÌNH THƯỜNG

Window 128 mẫu ở 100 Hz dài 1,27 s và phải **bao quanh** thời điểm chụp ảnh. Vì
vậy khoảng 6–7 ảnh đầu và 6–7 ảnh cuối mỗi trajectory không có đủ IMU hai bên và
bị loại — tổng ≈ 13 ảnh/trajectory, **không phụ thuộc độ dài trajectory**.

Con số này đã được kiểm chứng: bản sao local dựng đúng 100 Hz/10 Hz cho ra đúng
13 `rejected_centre` mỗi trajectory. Với dữ liệu thật:

- 284 ảnh → mất ≈ 4,6 %
- 515 ảnh → mất ≈ 2,5 %

Chỉ cần lo khi `rejected_nonuniform` hoặc `rejected_outside` lớn, hoặc
`rejected_centre` vượt xa 13/trajectory — khi đó timestamp mới thực sự có vấn đề.

## 6. Chọn DATA_ROOT — chỗ dễ sai nhất

Bên cạnh `tartanair-v2` còn có `tartanair-v2-jepa/train/` và `norm_stats.yaml`.
Loader **không dùng** hai thứ đó. Nhưng nếu `--data-root` trỏ cao hơn một bậc,
`os.walk` sẽ nhặt luôn bản sao trong `tartanair-v2-jepa/train/`; vì `train` là tên
split nên một nửa trajectory có split hint còn một nửa không, và build-manifest
dừng với:

```text
Mixed explicit and missing split directories; use a consistent dataset root.
4 trajectory(ies) sit under a train/valid/test directory, for example ...
```

Quy tắc: **trỏ vào đúng thư mục mà con trực tiếp của nó là các environment.**

Đường dẫn bạn audit được là
`/kaggle/input/datasets/buidinhkhoi/tartanairkhoi/tartanair-v2`. Mount thật trong
notebook Kaggle thường có dạng `/kaggle/input/<slug>/tartanair-v2`, nên **đừng
hardcode**: Cell 3 tự dò và ghi đè `data.root`.

## 7. Lệnh đúng

Lệnh trong bản audit cũ (`--root`, `--out`, `--window`) **không phải flag của CLI
này** và sẽ lỗi. Dùng:

```bash
python3 -m qjepa build-manifest \
  --config configs/kaggle_tartanair_v2.yaml \
  --data-root /kaggle/input/<slug>/tartanair-v2 \
  --output /kaggle/working/outputs/manifest
```

`imu_window` lấy từ config (`data.imu_window: 128`), không truyền qua dòng lệnh.

Các bước còn lại nằm trong [KAGGLE_TRAIN_CELLS.md](KAGGLE_TRAIN_CELLS.md).

## 8. Chưa kiểm chứng

- Mới đọc thử **3/166** trajectory. Cell 5 kiểm tra toàn bộ; nếu có trajectory
  nào lệch `len(ảnh) != len(cam_time)` thì build-manifest dừng và nêu tên.
- Chưa chạy trên GPU thật nên chưa có số VRAM/tốc độ; Cell 7 đo trước khi train.
- `norm_stats.yaml` và `tartanair-v2-jepa/train/manifest.csv` của bạn **không**
  được dùng: normalization được tính lại từ IMU sạch của đúng split train.
