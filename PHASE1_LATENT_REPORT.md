# Báo cáo phase 1 — latent pretraining

## Kết quả đã chạy

Smoke test trên batch tổng hợp B2, ảnh 32×32 và IMU 32×6 đã hoàn tất một optimizer
update với clean-online branch, teacher target, variance/covariance và encoder
sensitivity. Decoder không tồn tại trong model phase 1.

- QWT round-trip max absolute error FP32: `2.384185791015625e-07`.
- Loss tổng smoke: `1.591516137123108`.
- JEPA latent loss: `0.6546900272369385`.
- Variance loss: `0.7718220353126526`.
- Covariance loss: `2.060617208480835`.
- Successful updates: `1`.

Các giá trị này chỉ chứng minh forward/backward hữu hạn. Không dùng chúng để kết
luận representation đã học được nội dung ảnh hay động học IMU.

## Pilot chính

Trạng thái: **NOT_RUN**. Chưa có manifest TartanAir và chưa chạy 10.000 update B8
trên GPU. Vì vậy chưa có checkpoint latent chính, collapse curves, effective rank
trên bank 64 sample hoặc so sánh control/treatment Jacobian.

Recipe đã khóa tại `configs/pipeline_v3.yaml`: B8 thật, 10.000 update, teacher EMA
0.99→0.999, sensitivity tắt 500 update rồi ramp 1.000 update tới `1e-4`.

