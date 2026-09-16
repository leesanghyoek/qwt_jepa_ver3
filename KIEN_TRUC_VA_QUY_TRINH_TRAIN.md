# Kiến trúc thực tế và quy trình train ảnh/IMU

Tài liệu mô tả source `qjepa/` và cấu hình chính `configs/pipeline_v3.yaml` sau
kiểm tra ngày 15/09/2026. Hai tài liệu thiết kế đầu vào vẫn giữ nguyên:
[migration guide](QWT_JEPA_JACOBIAN_MIGRATION_GUIDE.md) và
[architecture diagrams](QWT_JEPA_V3_DETAILED_ARCHITECTURE_DIAGRAMS.md).
Lệnh chạy từng cell: [KAGGLE_TRAIN_CELLS.md](KAGGLE_TRAIN_CELLS.md).

**Ý chính:** phase 1 học biểu diễn ảnh/IMU sạch trong latent space; phase 2 mới
học chuyển latent thành ảnh và chuỗi IMU. Không có decoder hoặc loss khôi phục
trong phase 1. Đây là biến thể JEPA dự đoán latent clean từ input bị corruption,
không triển khai patch masking kiểu I-JEPA.

**Giới hạn transform:** tên lớp `QuaternionWaveletTransform2D` và backend
`qwt_dualtree_db4` hiện trỏ tới bốn cây db4 dùng offset 0/1, pack 48 kênh thực.
Chưa có kiểm chứng filter Hilbert pair hoặc đối chiếu QWT reference. Vì vậy tài
liệu gọi nó là **transform ảnh hiện tại**, không kết luận đã đạt yêu cầu QWT
chuẩn của đặc tả. Encoder/fusion là Conv/MLP số thực; không có Hamilton layer.

## 1. Dữ liệu đi vào model

```mermaid
flowchart TD
    D[Trajectory camera và IMU sạch] --> S[Chia train / valid / test theo environment và motion ID]
    S --> N[Mean và std IMU chỉ từ các trajectory train sạch]
    S --> P[Ghép timestamp: 1 ảnh với 128 hàng IMU liên tiếp]
    P --> IC[Resize và center crop ảnh RGB 256 x 256]
    P --> UC[IMU sạch: ax ay az gx gy gz]
    IC --> IN[Blur, giảm sáng, noise cảm biến, codec]
    UC --> UN[Corruption cả trajectory rồi cắt window]
    N --> Z[Normalizer cố định]
    UN --> Z
    IN --> B[Backbone online]
    Z --> B
    IC --> T[Target sạch]
    UC --> T
```

| Đại lượng | Shape | Đơn vị/quy ước |
| --- | --- | --- |
| Ảnh clean/noisy | `[B,3,256,256]` | float32 RGB, `[0,1]` |
| IMU clean/noisy vật lý | `[B,6,128]` | `ax,ay,az` m/s²; `gx,gy,gz` rad/s |
| IMU normalized | `[B,6,128]` | `(u - mean_train) / scale_train` |
| Timestamp ảnh | `[B]` | giây, float64 |
| Timestamp IMU | `[B,128]` | giây, float64, tăng nghiêm ngặt |
| Metadata thời gian | `[B,3]` | tính chênh lệch bằng float64 rồi mới cast float32 |

`scale_train = max(std_train, floor)` theo từng kênh; floor accel=`1e-3`,
gyro=`1e-4`. Không chuẩn hóa riêng từng
window vì sẽ mất bias/biên độ cần phục hồi. Train normalization tính từ toàn bộ
hàng IMU của trajectory train, loại trùng các stream giống hoàn toàn.

Ảnh được ghép với cửa sổ có **tâm gần timestamp ảnh nhất**, sai số tối đa một
chu kỳ IMU; reject window thiếu hàng hoặc dt lệch quá 2% so với median dt.
Với IMU 100 Hz, 128 mẫu phủ 1,27 giây. Cửa sổ nằm hai phía thời điểm ảnh nên
pipeline hiện tại xử lý offline hoặc cần chờ mẫu tương lai khoảng 0,635 giây.
Không padding, không nối window qua hai trajectory.

Hai bản easy/hard cùng `(environment, trajectory_id)` nằm cùng split. Sampler
train hiện cân bằng tương đối các `trajectory_key`, bảo đảm số trajectory tối
thiểu trong mỗi batch; có thể lấy lại trajectory ít mẫu. Easy/hard vẫn là hai
key sampler khác nhau: điều này không bảo đảm hai chuyển động độc lập. Sampler
không bảo đảm khoảng cách thời gian tối thiểu giữa các frame trong batch.

## 2. Corruption mô phỏng camera kém và IMU

Clean target được giữ nguyên sau resize/crop. Chỉ input bị làm xấu.

| Nhóm | Thiết lập mặc định |
| --- | --- |
| Defocus | xác suất 0,80; sigma 0,4–3 px |
| Motion blur | xác suất 0,65; chiều dài 3–17 px |
| Giảm độ phân giải | xác suất 0,35; scale 0,45–0,90 |
| Giảm sáng | exposure gain 0,10–0,55, tone curve gamma 0,50–0,90 |
| Màu và tối góc | white balance 0,82–1,18; vignette 0–0,45 |
| Nhiễu cảm biến | shot noise 250–1800 photon; read noise 1/255–5/255; row noise và hot pixel |
| Lưu ảnh | 6–8 bit; JPEG xác suất 0,35, quality 35–80 |
| Nhiễu accel | white sigma 0,05–0,30 m/s², bias, random walk |
| Nhiễu gyro | white sigma 0,002–0,015 rad/s, bias, random walk |
| Lỗi IMU khác | low-pass, sai scale, trộn trục, spike, dropout giữ mẫu, lượng tử |

Trình tự ảnh: blur/giảm phân giải → exposure/màu/gamma → noise → lượng tử/JPEG.
Thông số camera ổn định theo đoạn 1 giây; noise cảm biến thay đổi theo frame.
IMU được tạo nhiễu **trên cả trajectory** để hai window chồng lấn có cùng giá trị
input tại cùng timestamp. Không cộng nhiễu riêng cho từng window.

Train có 5% xác suất giữ clean ở mỗi corruptor. Validation/test dùng realization
cố định và `clean_probability=0`; scenario `clean` được yêu cầu riêng. `full`
nghĩa là hỗn hợp lỗi theo các xác suất trên, không phải mọi frame đều có mọi loại
blur. Train đổi realization mỗi epoch; seeds riêng cho corruption, sampler,
vị trí regularization, probe Jacobian và initialization.

Đây là mô phỏng. Cần hiệu chỉnh khoảng corruption theo camera/IMU thực; độ sáng
mất hoàn toàn hoặc chi tiết bị blur mất không thể được bảo đảm khôi phục chính
xác. Pipeline train hiện cần clean reference đồng bộ, chưa hỗ trợ chỉ có dữ
liệu noisy mà không có target sạch.

## 3. Backbone và kích thước từng tầng

```mermaid
flowchart LR
    I[RGB B x 3 x 256 x 256] --> WI[Transform ảnh: 48 x 128 x 128]
    U[IMU chuẩn hóa B x 6 x 128] --> WU[Haar: 12 x 64]
    WI --> EI[CNN2D: 32, 64, 96, 128]
    WU --> EU[CNN1D: 32, 64, 96, 128]
    EI --> FI[FI: B x 128 x 16 x 16]
    EU --> FU[FU: B x 128 x 8]
    FI --> F[Gated multimodal fusion]
    FU --> F
    M[Metadata B x 3] --> F
    F --> ZI[ZI: B x 128 x 16 x 16]
    F --> ZU[ZU: B x 128 x 8]
```

| Tầng | Nhánh ảnh | Nhánh IMU |
| --- | --- | --- |
| Transform | `48×128×128` | `12×64` |
| Stage 0, stride 1 | `32×128×128` | `32×64` |
| Stage 1, stride 2 | `64×64×64` | `64×32` |
| Stage 2, stride 2 | `96×32×32` | `96×16` |
| Stage 3, stride 2 | `128×16×16` | `128×8` |

Mỗi Stage: Conv kernel 3 + GroupNorm(8) + SiLU → residual block gồm hai Conv
kernel 3 và GroupNorm. Không dropout/BatchNorm. Không xuất các feature tầng sớm
cho decoder. Shape bảng không ghi chiều batch.

Transform ảnh xử lý từng kênh RGB, bốn band và bốn component-slot:
`3 × 4 × 4 = 48` kênh, periodic boundary. Synthesis trung bình bốn cây nghịch
đảo. Haar tách `(u_even+u_odd)/sqrt(2)` và `(u_even-u_odd)/sqrt(2)` thành 12 kênh.
Các bộ lọc cố định, không có trọng số trainable.

## 4. Fusion giữ lại lưới ảnh và trục thời gian

```mermaid
flowchart TD
    FI[FI] --> PI[Mean không gian rồi LayerNorm: 128]
    FU[FU] --> PU[Adaptive pool 4 bins rồi Linear 512 tới 128 và LayerNorm]
    PI --> CAT[Concat image summary, IMU summary, metadata: 259]
    PU --> CAT
    MT[3 metadata thời gian] --> CAT
    CAT --> SH[MLP 259 tới 256 tới 128, SiLU, LayerNorm]
    CAT --> GI[Sigmoid gate ảnh 128]
    CAT --> GU[Sigmoid gate IMU 128]
    SH --> BI[Broadcast trên lưới 16 x 16]
    SH --> BU[Broadcast trên 8 vị trí thời gian]
    FI --> DI[Concat FI và shared rồi Conv1x1, SiLU, Conv1x1]
    BI --> DI
    FU --> DU[Concat FU và shared rồi Conv1x1, SiLU, Conv1x1]
    BU --> DU
    DI --> ZI[ZI = FI + gateI nhân deltaI]
    GI --> ZI
    DU --> ZU[ZU = FU + gateU nhân deltaU]
    GU --> ZU
```

Metadata là `(t_image - tâm_window)/span`, `log(span)`,
`log(median_dt / 0.01)`. Gate weight khởi tạo 0, bias −2 nên sigmoid ban đầu
khoảng 0,119. Model bắt đầu bằng hiệu chỉnh nhỏ từ thông tin chung, vẫn giữ
feature của từng modality. Đây là fusion qua summary, không phải cross-attention
theo từng pixel/mẫu IMU. Giữ trật tự thời gian trong FU/ZU và vị trí không gian
trong FI/ZI cho bước khôi phục về sau.

## 5. Phase 1: JEPA trong latent space

```mermaid
flowchart TD
    N[Ảnh và IMU noisy] --> O[Online encoder và fusion, có gradient]
    O --> Z[ZI và ZU]
    Z --> P[Hai predictor theo token]
    C[Ảnh và IMU clean] --> T[Teacher encoder, no_grad]
    T --> TT[TI và TU detached, trước fusion]
    P --> J[JEPA: SmoothL1 giữa latent đã LayerNorm]
    TT --> J
    C --> OC[Online encoder và fusion clean, có gradient]
    O --> R[Variance và covariance của 8 raw feature maps]
    OC --> R
    N --> FD[Probe hữu hạn trước fusion, luân phiên ảnh và IMU]
    FD --> S[Sensitivity loss]
    J --> L[Loss phase 1]
    R --> L
    S --> L
    L --> OPT[Backward và AdamW online, fusion, predictors]
    OPT --> EMA[EMA cập nhật teacher sau update thành công]
    EMA --> T
```

Teacher chỉ gồm hai encoder, deepcopy online lúc khởi tạo. Teacher không có
fusion, predictor hoặc decoder. Nhánh clean-online chia sẻ trọng số với online
noisy, vẫn giữ gradient để loss chống collapse điều chỉnh representation clean.

Predictor mỗi modality: token `[B,K,128]` → LN → Linear(128,256) → GELU →
Linear(256,128). Ảnh có `K=256`; IMU có `K=8`.

```text
L_JEPA = 0.5 × [SmoothL1(LN(P_I(ZI)), LN(TI))
               + SmoothL1(LN(P_U(ZU)), LN(TU))]
L_phase1 = L_JEPA + L_variance + 0.01 × L_covariance + lambda(s) × L_FD
```

LN trong phép đo loss không có affine. Variance/covariance tính trên raw
`FI,FU,ZI,ZU` của **cả noisy và clean**, rồi lấy trung bình tám map. Chọn 16 vị
trí ảnh và 8 vị trí IMU; tại mỗi vị trí thống kê qua chiều batch, mẫu số `B−1`.
Không gộp vị trí không gian vào batch. Vì thế phase 1 cần batch vật lý ≥8;
accumulation B4 hai lần không tương đương B8 cho loss này.

Sensitivity dùng probe dấu ±1, ảnh epsilon=1/255 và clamp `[0,1]`, IMU epsilon
=0,01 trong đơn vị normalized. Năng lượng input tính **sau clamp thực tế**.

```text
L_FD = mean_b [mean((LN(F(x+delta)) - LN(F(x)))²)
                / mean(delta_thực_tế²)]
```

F là encoder dense trước fusion; gradient đi qua cả nhánh gốc và nhánh probe.
Không đo Jacobian của decoder hay raw reconstruction. Nguồn probe luân phiên
ảnh/IMU theo số update thành công. Đồ thị ghi `encoder_sensitivity` là gain đã
chuẩn hóa feature; chưa có gain raw riêng.

### Trình tự và lịch tối ưu

1. Lấy một batch noisy/clean ghép đúng timestamp; chuẩn hóa IMU đúng một lần.
2. Forward noisy online, clean online và clean teacher.
3. Tính JEPA, variance/covariance; nếu đến lịch thì forward probe thêm một encoder.
4. Kiểm tra loss, backward, clip gradient norm ở 5; từ chối gradient không hữu hạn.
5. AdamW update online/fusion/predictors; EMA teacher; tăng bộ đếm thành công.
6. Theo lịch checkpoint, đo validation bank rồi lưu `last.pt`.

| Tham số | Phase 1 | Phase 2 |
| --- | --- | --- |
| Optimizer | AdamW, weight decay `1e-4` | AdamW, weight decay `1e-4` |
| LR đỉnh / sàn | `2e-4` / `1e-6` | `2e-4` / `1e-6` |
| Warmup | 500 updates | 250 updates |
| Sau warmup | cosine | cosine |
| Tổng updates | 10.000 | 5.000 |
| Batch vật lý | 8 | 4 |
| Accumulation | 1 | 2 |
| Precision | FP32 | FP32 |
| Teacher EMA | 0,99 → 0,999 | không dùng |
| Sensitivity | off 500 updates, ramp 1.000 tới `1e-4` | không dùng |

## 6. Kiểm tra latent và chuyển phase

Bank validation chọn tối đa 64 mẫu, phân bổ giữa trajectory và trải theo thời
gian, giữ cố định sample ID và corruption realization. Nếu dataset nhỏ, dùng
toàn bộ mẫu; cần ít nhất hai mẫu để tính std/rank. Forward theo minibatch nhưng
**nối feature cả bank trước khi tính thống kê**. Effective rank của pooled
feature bị chặn bởi `min(D,N−1)`; bank 64, D128 không thể đạt rank 128.

Đo raw RMS, same-position std, effective rank; thêm std/rank sau LayerNorm trên
10 map: noisy/clean `FI,FU,ZI,ZU` và teacher `TI,TU`. So với giá trị lúc khởi tạo:
std/rank dưới 10% hoặc raw RMS ngoài `[0,1×;10×]` sẽ cảnh báo. Ba kiểm tra liên
tiếp cảnh báo làm gate FAIL và dừng train. Checkpoint WARN cũng chưa được chuyển
phase. Lịch FD hiện phụ thuộc update count, chưa tự hoãn khi gate WARN.

Phase 2 chỉ nhận checkpoint có đúng version/phase, chưa từng dùng reconstruction,
zero decoder calls, manifest/config/normalizer/backbone khớp, đủ toàn bộ ngân
sách phase 1 và gate PASS. Gate PASS kiểm tra suy giảm tương đối, **không chứng
minh latent đã đủ thông tin khôi phục**; cần xem kết quả phase 2/held-out.

## 7. Phase 2: decoder khôi phục từ latent

```mermaid
flowchart LR
    X[Input noisy] --> B[Backbone và normalizer frozen, eval, no_grad]
    B --> ZI[ZI: 128 x 16 x 16]
    B --> ZU[ZU: 128 x 8]
    ZI --> DI[Decoder ảnh mới]
    ZU --> DU[Decoder IMU mới]
    DI --> CI[Hệ số tuyệt đối: 48 x 128 x 128]
    DU --> CU[Hệ số tuyệt đối: 12 x 64]
    CI --> II[Inverse transform ảnh]
    CU --> IU[Inverse Haar]
    II --> RI[Ảnh raw 3 x 256 x 256]
    IU --> RU[IMU normalized 6 x 128]
    RU --> PH[Denormalize về m/s2 và rad/s]
    RI --> L[Loss reconstruction với target clean]
    RU --> L
    L --> O[AdamW chỉ cập nhật decoder]
```

| Tầng decoder | Ảnh | IMU |
| --- | --- | --- |
| Input latent | `128×16×16` | `128×8` |
| Resize + Stage 128→96 | `96×32×32` | `96×16` |
| Resize + Stage 96→64 | `64×64×64` | `64×32` |
| Resize + Stage 64→32 | `32×128×128` | `32×64` |
| Conv head kernel 3 | `48×128×128` | `12×64` |
| Synthesis | `3×256×256` | `6×128` |

Resize bilinear cho ảnh, linear cho IMU, `align_corners=False`. Head tuyến tính
dự đoán hệ số tuyệt đối, không cộng input coefficient. Decoder chỉ nhận ZI/ZU,
không nhận FI/FU, ảnh/IMU raw hay skip tầng sớm. Layout dùng cho inverse transform
là metadata shape, không chứa tín hiệu nhiễu để bypass latent.

```text
L_phase2 = L1(image_raw_restored, image_clean)
           + 0.5 × [SmoothL1(accel_norm_restored, accel_norm_clean)
                    + SmoothL1(gyro_norm_restored, gyro_norm_clean)]
```

Không clamp ảnh trước loss train. Clamp chỉ khi tính image metrics/hiển thị/xuất
PNG. IMU loss dùng normalized để cân bằng scale; chỉ số cuối dùng đơn vị vật lý.
Mỗi update tích lũy hai microbatch B4, mỗi loss chia 2 trước backward. Hash
backbone/normalizer được kiểm tra khi lưu checkpoint để phát hiện thay đổi.

## 8. Checkpoint, đánh giá và cách đọc kết quả

`last.pt` chứa trọng số, optimizer, RNG, số update/microbatch, config và provenance.
Resume giữ tiến độ sampler và corruption epoch; log vượt checkpoint do gián đoạn
được cắt về update đã lưu. Phase 2 giữ best validation score qua resume, parent
so bằng backbone hash nên có thể đổi đường dẫn mount checkpoint. Khi chuyển
session Kaggle, mang theo cả thư mục phase2, gồm best checkpoint và logs.

Validation phase 2 mặc định 32 mẫu cố định trải giữa trajectory/time, không phải
toàn split. Chọn best theo `image_mae_clamped + 0.5×(accel_smooth+gyro_smooth)` sau
gộp overlap IMU. Điểm này khác loss train raw ảnh; cần đọc riêng train và valid.
`evaluate` cuối dùng toàn test khi không truyền `--max-batches`.

```mermaid
flowchart TD
    CK[Best theo validation] --> TE[Test held-out, corruption cố định]
    TE --> IM[Metric ảnh từng frame rồi trung bình]
    TE --> U[IMU window dự đoán]
    U --> M[Gộp cùng trajectory bằng trọng số tam giác dương]
    M --> MASK[Coverage mask: mỗi timestamp được tính một lần]
    MASK --> MET[RMSE, MAE, bias 6 trục và sai số đạo hàm theo dt]
    IM --> OUT[JSON, CSV, biểu đồ baseline và restored]
    MET --> OUT
    TE --> PAN[Panel clean / noisy / restored / error và trace IMU]
```

Metric ảnh: MAE thấp hơn tốt hơn; PSNR/SSIM cao hơn tốt hơn. SSIM dùng cửa sổ
Gaussian tối đa 11, padding zero, data range 1; PSNR cap120 dB khi MSE≤`1e-12`.
Luôn so với `baseline_*` của chính input noisy; loss giảm không đủ để kết luận
khử nhiễu tốt. Có thể restored tệ hơn input, đặc biệt trên input vốn đã sạch.

IMU gộp prediction/clean/noisy cùng trọng số. Derivative metric chỉ tính hai hàng
kề nhau đều covered, chia dt thực; không nối qua khoảng trống. `.npz` lưu đầy đủ
timestamps, clean/noisy/restored `[N,6]` và `coverage[N]`; plot lấy tối đa4.000
điểm cho dễ xem, metrics vẫn dùng toàn bộ hàng covered.

Outputs gồm curves của hai phase, bảng metrics 10 scenario, panel ảnh, trace IMU
6 trục. Xem [CODE_REVIEW_KAGGLE.md](CODE_REVIEW_KAGGLE.md) để biết phạm vi kiểm
chứng và các giới hạn: chưa có kết quả train chính trên Kaggle/camera thật.
