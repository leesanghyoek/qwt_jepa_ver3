# Báo cáo phase 2 — khôi phục từ latent

## Kết quả đã chạy

Smoke test đã tạo decoder mới, freeze backbone phase 1 và hoàn tất một optimizer
update từ `ZI/ZU`. Hash backbone trước/sau update giống nhau; decoder có thay đổi.

- Loss tổng smoke: `1.235841989517212`.
- Image L1: `0.5476357340812683`.
- IMU accel SmoothL1: `0.610332727432251`.
- IMU gyro SmoothL1: `0.7660796046257019`.

Đây là smoke trên model nhỏ, initialization ngẫu nhiên; không phải chỉ số chất
lượng khôi phục.

## Pilot chính

Trạng thái: **NOT_RUN** vì chưa có checkpoint phase 1 đạt latent gates và chưa có
dataset held-out. Chưa có MAE/PSNR/SSIM ảnh, RMSE/bias/variation error IMU hoặc
panel clean/noisy/restored trên trajectory test.

Protocol đã có trong CLI qua `evaluate --protocol`, bao gồm clean/clean,
noisy-image, noisy-IMU, noisy/noisy, low-light-only, blur-only, sensor-noise-only
và ba nhóm lỗi IMU.

