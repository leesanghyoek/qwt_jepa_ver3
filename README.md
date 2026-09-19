# QWT–JEPA v3 cho ảnh thiếu nhiễu sáng và IMU

Source: [leesanghyoek/qwt_jepa_ver3](https://github.com/leesanghyoek/qwt_jepa_ver3).
Trên Kaggle, bắt đầu từ [Cell 1: clone GitHub](KAGGLE_TRAIN_CELLS.md#cell-1--clone-source-từ-github-và-ghi-lại-commit),
sau đó chạy lần lượt các cell train và đánh giá.

Repository này triển khai pipeline hai giai đoạn theo
`QWT_JEPA_JACOBIAN_MIGRATION_GUIDE.md` và
`QWT_JEPA_V3_DETAILED_ARCHITECTURE_DIAGRAMS.md`:

## Tài liệu chạy Kaggle và kiến trúc thực tế

- [Các cell Kaggle, train, resume, đồ thị và kết quả](KAGGLE_TRAIN_CELLS.md).
- [Sơ đồ kiến trúc chi tiết và quy trình train](KIEN_TRUC_VA_QUY_TRINH_TRAIN.md).
- [Kết quả kiểm tra code và giới hạn còn lại](CODE_REVIEW_KAGGLE.md).

Backend ảnh hiện dùng bốn cây db4 dịch pha nguyên và pack 48 kênh. Round-trip đã
được kiểm thử; **chưa xác minh tương đương QWT reference/Hilbert pair**. Tên API
`qwt_dualtree_db4` được giữ để tương thích, không phải bằng chứng QWT chuẩn.

## Luồng tổng quát

```mermaid
flowchart LR
    N[Ảnh và IMU nhiễu] --> B[QWT/Haar + online encoders + fusion]
    B --> Z[Dense latent ZI/ZU]
    C[Ảnh và IMU sạch] --> T[EMA teacher encoders]
    Z --> P[Latent predictors]
    P --> L[JEPA + chống collapse + encoder sensitivity]
    T --> L
    Z --> A[Decoder neo, hệ số tuyệt đối, bị vứt sau phase 1]
    C --> A
    A --> L
    Z2[ZI/ZU từ backbone đã đóng băng] --> D[Decoder phase 2 mới]
    D --> DZ[Δ hệ số]
    IN[Hệ số wavelet của chính input nhiễu] --> SUM[Cộng]
    DZ --> SUM
    SUM --> W[Inverse wavelet]
    W --> R[Ảnh và IMU phục hồi]
```

Phase 1 **có** một decoder phụ làm *neo*: nó chấm điểm latent trên hệ số wavelet
sạch với trọng số `0.45`, rồi bị vứt bỏ khi phase 1 kết thúc. Không có neo này
JEPA vẫn đạt loss thấp trên một latent đã ném đi tín hiệu — đó đúng là kết quả
của lần train đầu tiên. Vẫn không có đường pixel-space trong phase 1
(`reconstruction_loss_weight: 0.0`).

Phase 2 tải checkpoint phase 1 hợp lệ, đóng băng encoder/fusion/normalizer, khởi
tạo **decoder hoàn toàn mới** và chỉ tối ưu hai decoder đó.

## Kiến trúc chi tiết

Mọi shape và số tham số dưới đây được **in ra từ chính model** dựng bằng
`configs/pipeline_v3.yaml`, không phải tính tay. Batch `B` được lược khỏi bảng.

### Sơ đồ kiến trúc

Ba sơ đồ thay vì một, vì GitHub co sơ đồ mermaid cho vừa bề ngang trang: một sơ
đồ to sẽ bị thu nhỏ đến mức không đọc nổi chữ. Tách ra thì mỗi sơ đồ hiện ở cỡ
thật.

Đọc màu: **xanh tím** = wavelet, không tham số và khả nghịch · **xanh dương** =
backbone, học ở phase 1 rồi đóng băng · **tím** = latent · **vàng** = chỉ sống
trong phase 1 rồi bị vứt · **xanh lá** = decoder phase 2 · **hồng** = số hạng loss.

#### 1. Backbone dùng chung cho cả hai phase

```mermaid
flowchart TB
    XN["<b>Ảnh nhiễu</b> · 3 × 256 × 256"]
    UN["<b>IMU nhiễu</b> · 6 × 128"]
    TM["Timestamp cam + IMU<br/>metadata 3 chiều"]
    QW["<b>QWT dual-tree db4</b> · 0 tham số<br/>3 màu × 4 băng × 4 thành phần<br/>Ci = 48 × 128 × 128"]
    HA["<b>Haar 1-D trực chuẩn</b> · 0 tham số<br/>6 kênh × 2 băng<br/>Cu = 12 × 64"]
    EI["<b>Encoder ảnh</b> · 4 stage · 0,75 M<br/>32×128×128 → 64×64×64 → 96×32×32<br/><b>FI = 128 × 16 × 16</b>"]
    EU["<b>Encoder IMU</b> · 4 stage · 0,25 M<br/>32×64 → 64×32 → 96×16<br/><b>FU = 128 × 8</b>"]
    FS["<b>Gated fusion</b> · 0,33 M<br/>128 + 128 + 3 = 259 → MLP 256 → 128<br/>cổng sigmoid bias −2 ⇒ gần identity lúc đầu"]
    ZI["<b>ZI = 128 × 16 × 16</b><br/>32.768 số · nén 6,0×<br/>1 ô latent = khối 16 × 16 pixel"]
    ZU["<b>ZU = 128 × 8</b><br/>1.024 số · giãn 0,75×<br/>IMU không hề bị nén"]
    XN --> QW --> EI --> FS
    UN --> HA --> EU --> FS
    TM --> FS
    FS --> ZI & ZU
    classDef tf fill:#e8eaf6,stroke:#5c6bc0,color:#1a1a1a
    classDef bb fill:#e3f2fd,stroke:#1e88e5,color:#1a1a1a
    classDef lat fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#1a1a1a
    class QW,HA tf
    class EI,EU,FS bb
    class ZI,ZU lat
```

#### 2. Phase 1 — những gì gắn vào latent để ép nó học

```mermaid
flowchart TB
    Z["<b>ZI · ZU</b><br/>latent từ nhánh NHIỄU"]
    F["<b>FI · FU</b><br/>dense feature TRƯỚC fusion"]
    CLEAN["Ảnh + IMU <b>SẠCH</b><br/>chỉ tồn tại lúc train"]
    TE["<b>Teacher EMA</b><br/>bản sao 2 encoder · KHÔNG gradient<br/>m: 0,99 → 0,999"]
    PR["<b>Predictor</b> mỗi modality · 66 K<br/>LN → 128→256 → GELU → 256→128<br/>256 token ảnh · 8 token IMU"]
    AN["<b>Decoder neo</b> · hệ số TUYỆT ĐỐI<br/>cùng kiến trúc decoder phase 2<br/>bị vứt khi phase 1 kết thúc"]
    JE(["<b>JEPA loss</b> · trọng số 1,0<br/>online nhiễu ≈ teacher sạch"])
    VC(["<b>Variance 1,0 + Covariance 0,01</b><br/>8 raw map: FI FU ZI ZU + bản sạch"])
    JA(["<b>Encoder sensitivity</b> Jacobian<br/>Hutchinson + Rademacher<br/>bật sau update 500 · ramp 1000"])
    ANL(["<b>Anchor loss · 0,45</b><br/>+ băng chi tiết 0,5"])
    CLEAN --> TE --> JE
    Z --> PR --> JE
    Z --> AN --> ANL
    CLEAN -. "hệ số wavelet SẠCH = đích" .-> ANL
    Z --> VC
    F --> VC
    F -. "đo TRƯỚC fusion" .-> JA
    classDef lat fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#1a1a1a
    classDef p1 fill:#fff8e1,stroke:#f9a825,color:#1a1a1a
    classDef loss fill:#fce4ec,stroke:#d81b60,color:#1a1a1a
    class Z,F lat
    class CLEAN,TE,PR,AN p1
    class JE,VC,JA,ANL loss
```

Mũi tên nét đứt từ `FI · FU` cho thấy khối Jacobian đo **trước** fusion
(`target: online_dense_before_fusion`), không đo trên latent.

#### 3. Phase 2 — khôi phục, backbone đóng băng

```mermaid
flowchart TB
    ZI["<b>ZI</b> · 128 × 16 × 16<br/>backbone ĐÓNG BĂNG"]
    ZU["<b>ZU</b> · 128 × 8<br/>backbone ĐÓNG BĂNG"]
    DI["<b>Decoder ảnh</b> · 1,53 M · sub-pixel conv<br/>128×16×16 → 128×32×32 → 96×32×32<br/>→ 96×64×64 → 64×64×64<br/>→ 64×128×128 → 32×128×128"]
    DU["<b>Decoder IMU</b> · 0,33 M · sub-pixel conv<br/>128×8 → 128×16 → 96×16<br/>→ 96×32 → 64×32 → 64×64 → 32×64"]
    HI["head conv 3×3 · <b>zero-init</b><br/>Δi = 48 × 128 × 128"]
    HU["head conv 3×3 · <b>zero-init</b><br/>Δu = 12 × 64"]
    CI["<b>Ci</b> · hệ số của chính ảnh mờ<br/>48 × 128 × 128"]
    CU["<b>Cu</b> · hệ số của chính IMU nhiễu<br/>12 × 64"]
    PI(("＋"))
    PU(("＋"))
    SI["QWT synthesis"]
    SU["Haar synthesis"]
    OI["<b>Ảnh phục hồi</b><br/>3 × 256 × 256"]
    OU["<b>IMU phục hồi</b><br/>6 × 128"]
    L1(["<b>Loss phase 2</b><br/>L1 pixel + SmoothL1 accel/gyro β=0,05<br/>+ băng chi tiết LH/HL/HH · 2,0<br/>+ sai phân bậc một IMU · 0,5"])
    SK["<b>3 tầng encoder</b> · skip<br/>96×32×32 · 64×64×64 · 32×128×128<br/>đường nét của ảnh NHIỄU: sắc nhưng chưa đáng tin"]
    GT{{"<b>SkipMerge có cổng</b><br/>cổng = sigmoid(conv(đường latent))<br/>x + cổng × conv(skip)<br/>bias −2 ⇒ ban đầu gần như đóng"}}
    ZI --> DI --> HI --> PI --> SI --> OI --> L1
    ZU --> DU --> HU --> PU --> SU --> OU --> L1
    SK --> GT
    GT --> DI
    GT --> DU
    CI -. "không qua trọng số nào" .-> PI
    CU -.-> PU
    classDef tf fill:#e8eaf6,stroke:#5c6bc0,color:#1a1a1a
    classDef lat fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#1a1a1a
    classDef p2 fill:#e8f5e9,stroke:#43a047,color:#1a1a1a
    classDef loss fill:#fce4ec,stroke:#d81b60,color:#1a1a1a
    class ZI,ZU lat
    class CI,CU,SI,SU tf
    class DI,DU,HI,HU,PI,PU,OI,OU p2
    class SK bb
    class GT p2
    class L1 loss
```

Khối `SkipMerge` là chỗ phân công: **skip cấp độ phân giải, latent quyết định giữ
cái gì**. Cổng được sinh từ đường latent nên nó thay đổi theo từng vị trí và từng
kênh — decoder học cách dùng latent để *lọc* skip, chứ không chỉ pha trộn theo một
tỉ lệ cố định.

Hai mũi tên nét đứt là hai đường **không đi qua trọng số nào**: hệ số của chính
ảnh/IMU nhiễu cộng thẳng vào đầu ra. Đó là sàn identity — xem
[Hai quyết định thiết kế quan trọng](#hai-quyết-định-thiết-kế-quan-trọng).

> **Muốn phóng to?** Copy khối mermaid rồi dán vào <https://mermaid.live> để kéo
> thả và zoom thoải mái. Trên GitHub, bấm vào sơ đồ cũng mở được chế độ xem lớn.

### Một sample gồm gì

| | Shape | Số phần tử |
|---|---|---|
| Ảnh RGB | `[3, 256, 256]` | 196.608 |
| IMU (ax ay az gx gy gz) | `[6, 128]` | 768 |
| Timestamp camera | `[1]` | |
| Timestamp IMU | `[128]` | |

Window IMU dài 1,27 s và **phải bao quanh** thời điểm chụp ảnh; `build_time_metadata`
từ chối sample vi phạm thay vì âm thầm căn lệch.

### Tầng 1 — biến đổi wavelet (không tham số)

Hai biến đổi này **khả nghịch và không có tham số học được**; chúng chỉ đổi hệ
toạ độ.

**Ảnh — QWT dual-tree db4, 1 mức.** Bốn cây db4 dịch pha nguyên chạy song song.
Gói kênh theo thứ tự `[màu, băng, thành phần]`:

```
3 màu (R,G,B) × 4 băng (approx, detail_y, detail_x, detail_xy) × 4 thành phần (real, i, j, k) = 48 kênh
[3, 256, 256]  ->  [48, 128, 128]
```

Băng `approx` (LL) giữ độ sáng và bố cục; ba băng `detail_*` giữ **đường nét**.
Đây chính là nhóm băng mà loss chi tiết ở phase 2 nhắm vào.

**IMU — Haar 1-D trực chuẩn, 1 mức.** Nửa đầu là approx, nửa sau là detail:

```
6 kênh × 2 băng = 12 kênh
[6, 128]  ->  [12, 64]
```

### Tầng 2 — encoder (CNN 4 stage)

Mỗi `Stage` = `ConvBlock` (conv 3×3 → GroupNorm 8 nhóm → SiLU) + `ResBlock`
(hai conv 3×3 + GroupNorm, cộng tắt `act(x + norm2(conv2(y)))`). Stage 0 giữ
nguyên kích thước, ba stage sau `stride=2`.

| Stage | Encoder ảnh | Encoder IMU |
|---|---|---|
| vào | `[48, 128, 128]` | `[12, 64]` |
| 0 | `[32, 128, 128]` | `[32, 64]` |
| 1 — stride 2 | `[64, 64, 64]` | `[64, 32]` |
| 2 — stride 2 | `[96, 32, 32]` | `[96, 16]` |
| 3 — stride 2 | **`FI = [128, 16, 16]`** | **`FU = [128, 8]`** |

`DenseCoefficientEncoder` chỉ trả tầng trung gian khi người gọi **yêu cầu tường
minh** (`return_stages=True`), nên decoder không thể nhặt được skip do vô ý —
`phase2.encoder_skips` là nơi duy nhất quyết định.

### Tầng 3 — fusion có cổng

`SharedGatedFusion` giữ nguyên lưới không gian/thời gian, chỉ trộn thêm ngữ cảnh
toàn cục:

1. Tóm tắt ảnh = trung bình không gian của `FI` → LayerNorm → `[128]`.
2. Tóm tắt IMU = `AdaptiveAvgPool1d(4)` trên `FU` → phẳng `[512]` → Linear → LayerNorm → `[128]`.
3. Metadata thời gian `[3]`: lệch tâm chuẩn hoá, `log(span)`, `log(dt/0.01)`.
4. Nối `[128+128+3 = 259]` → MLP `259 → 256 → 128` → LayerNorm = vector chia sẻ.
5. Vector đó được phát lại lên từng vị trí, nối với feature gốc, qua hai conv 1×1,
   rồi **cộng có cổng**:

```
ZI = FI + sigmoid(gate_i) * delta_i(FI, shared)
ZU = FU + sigmoid(gate_u) * delta_u(FU, shared)
```

Cổng khởi tạo `weight = 0`, `bias = -2.0`, nên `sigmoid(-2) ≈ 0,12`: lúc bắt đầu
fusion gần như là identity và mỗi modality tự học trước, tránh việc một nhánh
nhiễu kéo sập nhánh kia ngay từ update đầu.

### Latent — chỗ quyết định trần chất lượng

| | Shape | Số phần tử | So với input | |
|---|---|---|---|---|
| `ZI` | `[128, 16, 16]` | 32.768 | 196.608 | **nén 6,0×** |
| `ZU` | `[128, 8]` | 1.024 | 768 | **giãn 0,75×** |

Hai dòng này giải thích phần lớn kết quả đo được:

- `ZI` là `16×16`, tức **mỗi ô latent phải mô tả một khối 16×16 pixel**. Đây là
  trần cứng của chi tiết ảnh, và không decoder nào vượt qua được nó. Muốn nét hơn
  thì phải sửa chỗ này (`encoders.py`, đổi `stride=2` của stage cuối thành `1` để
  có latent `32×32`), không phải sửa decoder.
- `ZU` **không hề nén** — nó còn nhiều số hơn chính tín hiệu IMU. Đó là lý do
  metric IMU luôn tốt hơn metric ảnh: bài toán IMU không bị bóp cổ chai.

### Phase 1 — những khối chỉ tồn tại lúc train

**Teacher EMA** (`EMATeachers`): bản `deepcopy` của hai encoder online, **không có
gradient**, cập nhật bằng `θ_t ← m·θ_t + (1−m)·θ_o` với `m` đi từ 0,99 lên 0,999.
Teacher ăn dữ liệu **sạch**, online ăn dữ liệu **nhiễu**.

**Predictor** (`LatentPredictor`, mỗi modality một cái): MLP theo từng token,
`LayerNorm → Linear(128→256) → GELU → Linear(256→128)`. Token ảnh là `16×16 = 256`
vị trí, token IMU là `8` vị trí. Nó dự đoán latent của teacher từ latent online.

**Decoder neo**: cùng kiến trúc decoder phase 2 nhưng **hệ số tuyệt đối**, chấm
điểm trên hệ số wavelet sạch, trọng số 0,45. Bị vứt sau phase 1.

**Encoder sensitivity (khối Jacobian)**: ước lượng Hutchinson bằng sai phân hữu
hạn với probe Rademacher, đo trên dense feature **trước fusion**, bật sau update
500 và ramp trong 1000 update.

### Phase 2 — decoder

Backbone (transform + 2 encoder + fusion) **đóng băng ở chế độ eval**. Chỉ hai
decoder được cập nhật. Nâng kích thước bằng **sub-pixel conv** (`Upsample`: conv
mở rộng kênh ×4 cho ảnh / ×2 cho IMU rồi `pixel_shuffle`), **không** dùng nội suy
bilinear — bilinear là bộ lọc thông thấp nên không sinh được tần số cao.

| Bước | Decoder ảnh | Decoder IMU |
|---|---|---|
| vào | `ZI [128, 16, 16]` | `ZU [128, 8]` |
| `shuffle2` | `[128, 32, 32]` | `[128, 16]` |
| `up2` | `[96, 32, 32]` | `[96, 16]` |
| `shuffle1` | `[96, 64, 64]` | `[96, 32]` |
| `up1` | `[64, 64, 64]` | `[64, 32]` |
| `shuffle0` | `[64, 128, 128]` | `[64, 64]` |
| `up0` | `[32, 128, 128]` | `[32, 64]` |
| `head` conv 3×3 | `Δ [48, 128, 128]` | `Δ [12, 64]` |
| cộng hệ số input | `C_in + Δ` | `C_in + Δ` |
| synthesis | `[3, 256, 256]` | `[6, 128]` |

`head` được **zero-init** ở chế độ residual, nên trước khi học gì đầu ra bằng
đúng đầu vào. Nếu lưới không chia hết cho 8, `resize` bilinear xử lý phần lẻ
**trước** `head` — đường thoát hiểm, không phải đường nâng ảnh.

### Số tham số

| Khối | Tham số | Train ở phase |
|---|---|---|
| `image_encoder` | 753.024 | 1 |
| `imu_encoder` | 248.832 | 1 |
| `fusion` | 331.264 | 1 |
| **backbone (tổng)** | **1.333.120** | 1, đóng băng ở phase 2 |
| predictor ảnh | 66.176 | 1, rồi vứt |
| predictor IMU | 66.176 | 1, rồi vứt |
| decoder ảnh | 1.527.600 | neo ở 1 (vứt), lại từ đầu ở 2 |
| decoder IMU | 328.524 | neo ở 1 (vứt), lại từ đầu ở 2 |
| **decoder (tổng)** | **1.856.124** | 2 |

Teacher EMA là bản sao của hai encoder (1.001.856 tham số) nhưng **không nhận
gradient**, nên không tính vào đây.

Thứ **duy nhất** đi từ phase 1 sang phase 2 là 1.333.120 tham số backbone. Tất cả
predictor và decoder neo đều bị bỏ; phase 2 dựng decoder hoàn toàn mới với seed
riêng (`decoder_initialization_seed`).

## Hai quyết định thiết kế quan trọng

**Decoder phase 2 dự đoán hiệu chỉnh, không dự đoán thay thế.** Với
`input_coefficient_residual: true`, đầu ra là `C_out = C_in + Δ(Z)` và head được
khởi tạo bằng 0, nên tại update 0 model trả lại **đúng** input. Đó là một sàn mà
model không thể tụt xuống dưới, và `Δ` chính là phần đóng góp đo được của latent:
ép `Δ = 0` là quay về baseline. Kiểm chứng end-to-end ở lần validate đầu tiên:

| | head tuyệt đối | **residual** | baseline (không làm gì) |
|---|---|---|---|
| PSNR | 6.26 | **11.43** | 11.40 |
| SSIM | 0.049 | **0.289** | 0.289 |
| accel | 1.631 | **0.612** | 0.613 |

**Decoder neo của phase 1 thì ngược lại: cố ý giữ hệ số tuyệt đối.** Cho nó dùng
residual sẽ để nó thoả mãn neo bằng `Δ ≈ 0` mà không ép được gì vào latent — phá
đúng mục đích của neo.

## Thành phần chính

- `qjepa/models/backbone.py`: QWT ảnh, Haar IMU, hai CNN encoder và gated fusion;
  trả `FI/FU/ZI/ZU`.
- `qjepa/models/pipeline.py`: hai wrapper phase riêng. `LatentPretrainingModel`
  chỉ nhận decoder neo khi `phase1.decoder_enabled` bật, và decoder đó không đi
  sang phase 2; `RestorationSystem` giữ backbone ở eval/frozen.
- `qjepa/models/decoders.py`: nhận `ZI/ZU`, và khi `encoder_skips` bật thì nhận
  thêm ba tầng trung gian của encoder qua `SkipMerge` **có cổng** — cổng sinh từ
  đường latent nên latent quyết định cho bao nhiêu skip đi qua ở từng vị trí.
  Khởi tạo sao cho đóng góp skip ban đầu bằng 0, nên sàn identity không bị phá. Upsample bằng sub-pixel conv (pixel
  shuffle) thay cho nội suy bilinear, vì bilinear là bộ lọc thông thấp nên không
  sinh được tần số cao. Ở chế độ residual, head zero-init và cộng vào hệ số input.
- `qjepa/corruptions/image.py`: blur quang học, blur chuyển động, giảm độ phân
  giải, exposure thấp, gamma, white balance, vignette, shot/read/row noise, hot
  pixel, lượng tử và JPEG. Mỗi frame bốc tham số riêng nên độ sáng và độ nhoè
  thay đổi giữa các frame, không phải một hệ số cố định cho cả segment.
- `qjepa/corruptions/imu.py`: bandwidth blur, scale/cross-axis error, white noise
  có gain thay đổi theo thời gian, rung băng hẹp 8–45 Hz, spike, dropout và lượng
  tử. Bias instability bị **gate** sau `wander_probability: 0.25`: phần lớn window
  dao động *quanh* tín hiệu sạch, chỉ thỉnh thoảng mới lệch đi.
- `qjepa/training/phase1.py`: noisy-to-clean latent prediction, teacher EMA,
  variance/covariance trên tám raw maps và finite-difference Jacobian trước fusion.
- `qjepa/training/phase2.py`: L1 pixel + SmoothL1 accel/gyro, cộng hai số hạng
  tần số cao. **Băng chi tiết** (LH/HL/HH của ảnh và nửa detail của Haar IMU,
  trọng số `2.0`) vì L1 pixel tối ưu về trung vị có điều kiện, mà với bài toán
  bất định như khử mờ thì trung vị đó *chính là ảnh mờ*. **Sai phân bậc một của
  IMU** (`imu_variation_weight: 0.5`) vì mọi số hạng khác chấm điểm từng mẫu độc
  lập, nên tín hiệu giật từng mẫu không bị phạt. `smooth_l1_beta: 0.05` giữ sai
  số IMU (`|x| ≈ 0,12`) trong vùng tuyến tính; ở `1.0` gradient yếu gấp 8 lần.
  Optimizer chỉ chứa decoder.
- `qjepa/execution.py`: chọn thiết bị và bọc forward bằng `DataParallel` khi có
  hai GPU; chỉ dict tensor đi qua ranh giới gather nên loss vẫn thấy cả batch.
- `configs/pipeline_v3.yaml`: recipe chính RGB 256×256, IMU 128×6.
- `configs/kaggle_tartanair_v2.yaml`: cùng recipe đó, chỉ đổi đường dẫn/thiết bị/
  worker cho notebook Kaggle + dataset TartanAir V2.

## Chuẩn bị môi trường

```bash
python3 -m pip install -e '.[test]'
```

Có thể chạy trực tiếp từ root repository mà chưa cài package:

```bash
python3 -m qjepa --help
```

## Thiết bị chạy: một hoặc hai GPU

Pipeline chạy trong **một process** trên CPU, một GPU, hoặc hai GPU bằng
`torch.nn.DataParallel`. `runtime.gpu_count` nhận `auto` (mặc định, lấy tối đa hai
GPU đang thấy), `1` hoặc `2`; `train-phase1`, `train-phase2` và `evaluate` có cờ
`--gpus` để ghi đè cho một lệnh:

```bash
python3 -m qjepa train-phase1 --config configs/pipeline_v3.yaml \
  --manifest manifests/tartanair --output outputs/experiment_01 --gpus 2
```

`--gpus 2` **báo lỗi** nếu chỉ thấy một GPU, thay vì âm thầm chạy chậm trên một
GPU. Backend này không dùng `torchrun`/DDP: nếu `WORLD_SIZE>1` CLI sẽ từ chối.

Điểm quan trọng về ngữ nghĩa: `batch_size` trong config luôn là **batch toàn
cục**. Với B8 trên hai GPU mỗi GPU chạy 4 mẫu, nhưng chỉ *feature dense* được
gather về GPU chính rồi mới tính loss — variance/covariance và JEPA vẫn thấy đủ
8 mẫu. Thống kê chống collapse tính riêng từng GPU rồi lấy trung bình sẽ **sai**
(hai nửa batch hằng số cho variance loss khác hẳn cả batch), nên forward song
song chỉ trả dict tensor, không trả loss theo từng thiết bị.

Vì vậy số GPU là chi tiết thực thi, không phải siêu tham số train: nó không nằm
trong configuration hash, không đổi kết quả mong đợi, và có thể resume một run
1 GPU bằng 2 GPU hoặc ngược lại. Mỗi lệnh in một dòng `Execution: {...}` và ghi
cùng thông tin đó vào `metadata.execution` của checkpoint để truy vết sau này.

## Cấu trúc TartanAir được hỗ trợ

Mỗi trajectory cần có:

```text
<root>/<environment>/<Data_easy|Data_hard>/<Pxxx>/
├── image_lcam_front/*.png
└── imu/
    ├── acc.npy hoặc acc.txt
    ├── gyro.npy hoặc gyro.txt
    ├── imu_time.npy hoặc imu_time.txt
    └── cam_time.npy hoặc cam_time.txt
```

Manifest chia theo `(environment, trajectory_id)` trước khi tạo window. Hai bản
`Data_easy/Data_hard` của cùng chuyển động luôn nằm cùng split. Mỗi sample gồm
một frame và 128 hàng IMU liên tục; không padding và không nối qua trajectory.

`--data-root` phải là thư mục mà **con trực tiếp của nó là các environment**. Trỏ
cao hơn một bậc sẽ nhặt thêm bản sao khác của dataset; nếu bản sao đó nằm dưới
thư mục tên `train/valid/test` thì build-manifest dừng và nêu tên thư mục vi phạm.

Với dataset TartanAir V2 trên Kaggle, xem [kaggle_dataset.md](kaggle_dataset.md):
đối chiếu từng điểm với loader, cách split 83 motion key, và vì sao khoảng 13
frame mỗi trajectory bị loại (window 1,27 s phải bao quanh thời điểm chụp).

Tạo manifest và thống kê normalization từ các timestamp train sạch, mỗi timeline
trùng hoàn toàn chỉ được tính một lần:

```bash
python3 -m qjepa build-manifest \
  --config configs/pipeline_v3.yaml \
  --data-root /duong/dan/TartanAir \
  --output manifests/tartanair
```

Sau đó điền `data.manifest_dir` trong YAML hoặc truyền `--manifest` cho từng lệnh.

## Kiểm tra corruption camera tối

```bash
python3 -m qjepa preview-corruption \
  --config configs/pipeline_v3.yaml \
  --image anh_sach.png \
  --output outputs/corruption_panel.png
```

Panel gồm `clean | corrupted | absolute error`; file JSON cùng tên lưu toàn bộ
tham số đã bốc. Corruption dùng seed theo sample/trajectory nên có thể tái lập.
Thông số mặc định nhấn mạnh điều kiện tối (`exposure_gain=0.21..0.63`,
`tone_gamma=0.46..0.79`) và blur mạnh. Nên đo ảnh camera thật rồi chỉnh các khoảng trong YAML; nếu corruption mô
phỏng tối hơn hoặc khác noise profile thực tế quá nhiều, model sẽ học sai domain.

## Phase 1: chỉ học latent

```bash
python3 -m qjepa train-phase1 \
  --config configs/pipeline_v3.yaml \
  --manifest manifests/tartanair \
  --output outputs/experiment_01
```

Checkpoint `phase1/last.pt` bắt buộc có:

```text
pipeline_version = 3
phase = latent_pretrain
trained_with_reconstruction = <true khi bat neo, false khi tat>
phase1_decoder_forward_calls = <so lan decoder neo chay>
```

Hai trường cuối không còn bị ép về `false/0`, nhưng **phải nhất quán với nhau**:
`trained_with_reconstruction` đúng khi và chỉ khi decoder đã chạy ít nhất một
lần. Metadata không được phép nói dối về việc phase 1 đã dùng decoder hay chưa.

Batch phase 1 phải là B≥8 thật. Gradient accumulation không được dùng để giả lập
batch statistics cho variance/covariance. Trước khi phase 2 được phép chạy,
checkpoint còn phải đạt gate về same-position std, effective rank và raw feature
scale trên validation bank; ba lần cảnh báo liên tiếp sẽ dừng run để chẩn đoán.

Để so sánh đóng góp của Jacobian công bằng, chạy treatment bằng
`configs/pipeline_v3.yaml` và control bằng `configs/phase1_control.yaml`. Hai config
dùng cùng initialization seed, corruption seed và sampler seed; điểm khác duy nhất
là `encoder_sensitivity.enabled`.

## Phase 2: khôi phục từ latent đã học

```bash
python3 -m qjepa train-phase2 \
  --config configs/pipeline_v3.yaml \
  --manifest manifests/tartanair \
  --backbone-checkpoint outputs/experiment_01/phase1/last.pt \
  --output outputs/experiment_01
```

Lệnh từ chối checkpoint phase 1 sai provenance hoặc manifest hash không khớp.
Hai output là `phase2/last.pt` và `phase2/best_joint_validation.pt`.

Mỗi checkpoint in một dòng so sánh trực tiếp với baseline "không làm gì":

```text
validation update=250 | PSNR 11.43 vs 11.40 | SSIM 0.289 vs 0.289 | accel 0.612 vs 0.613 | VUOT baseline
```

Vì head zero-init, dòng validate **đầu tiên** đã phải là `VUOT baseline`. Nếu nó
báo `chua vuot` ngay lần đầu thì residual chưa thực sự bật — kiểm
`phase2.input_coefficient_residual` và `phase2.output_coefficients` trong config
đang dùng. `validate_config` từ chối config mà hai trường này mâu thuẫn nhau, nên
file không thể mô tả sai việc decoder đang làm gì.

## Đánh giá và inference

Đánh giá riêng clean/noisy, thiếu sáng, blur, sensor noise và các nhóm lỗi IMU:

```bash
python3 -m qjepa evaluate \
  --checkpoint outputs/experiment_01/phase2/best_joint_validation.pt \
  --manifest manifests/tartanair \
  --split test \
  --protocol \
  --output outputs/experiment_01/test_results
```

Mỗi phase tự xuất `training_curves.png`, `history.csv`, `training_summary.json`
khi hoàn tất. Lệnh `evaluate` xuất `metrics.json/csv`, `comparison.png`, panel
clean/noisy/restored/error, đồ thị IMU sáu trục và `.npz` sau gộp overlap. Có thể
vẽ lại từ log bằng `python3 -m qjepa plot-training --run-dir outputs/experiment_01`.

Khôi phục một ảnh và một window IMU. CSV nhận 6 cột IMU hoặc 7 cột gồm timestamp:

```bash
python3 -m qjepa infer \
  --checkpoint outputs/experiment_01/phase2/best_joint_validation.pt \
  --image frame.png \
  --imu imu_window.csv \
  --output outputs/inference_001
```

Output gồm `image_restored.png`, `imu_restored.csv` trong đơn vị m/s² và rad/s,
và `metadata.json`.

## Kiểm thử

```bash
python3 -m qjepa smoke --config configs/smoke.yaml
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Biến môi trường ở lệnh pytest chỉ tránh plugin ROS được cài toàn hệ thống tự nạp;
test của project không phụ thuộc ROS.

## Kết quả đã đo được

Đo trên TartanAir V2, split test, so với baseline "đưa thẳng input nhiễu ra":

| Cấu hình | Probe tuyến tính (IMU / ảnh) | PSNR | Baseline |
|---|---|---|---|
| Không neo phase 1 | 17% / 25% | thua baseline ở cả bốn metric | — |
| Neo `0.30`, head tuyệt đối | 43% / 49% | 17.87 | 16.56 |
| Neo `0.45`, head residual | *chưa train lại* | — | — |

**Probe tuyến tính** là hồi quy ridge từ latent đã đóng băng về mục tiêu sạch. Nó
là *cận dưới* của lượng thông tin rút được từ latent, và là chỉ số cho biết neo
có tác dụng hay không, độc lập với chất lượng decoder.

Giới hạn còn lại, đo bằng protocol 10 kịch bản: đầu ra của cấu hình "neo 0.30,
head tuyệt đối" **gần như độc lập với đầu vào** — input trải 107.4 dB PSNR thì
output chỉ nhúc nhích 1.94 dB, và một input gần sạch bị phá (`blur_only` vào
37.5 dB, ra 15.7 dB). Đó chính là lý do có residual. Nút thắt **không** nằm ở
decoder: decoder đã rút được 62–74% trong khi probe tuyến tính chỉ rút 43%, nên
thêm ResNet hay pointwise vào decoder không giải quyết được gì — giới hạn là số
chiều của `ZI`, mỗi ô latent phủ một khối 16×16 pixel.

`encoder_skips` giờ đã cài và **bật mặc định** — nhưng nó mua độ nét bằng cách
lấy chi tiết từ ảnh đầu vào, nên hãy đọc con số kèm theo `decoder_input:
latent_plus_encoder_skips` chứ đừng gọi đó là khôi phục thuần từ latent.

Chưa làm: latent đa tỉ lệ, và adversarial loss.

## Giới hạn dữ liệu

JEPA teacher phase 1 và reconstruction loss phase 2 đều cần reference sạch trong
train. Ảnh từ camera kém chỉ dùng làm input deployment; nếu tập huấn luyện chỉ có
ảnh tối/nhiễu mà không có clean reference tương ứng, pipeline supervised này chưa
đủ thông tin để học target sạch. TartanAir clean có thể làm reference ban đầu,
nhưng cần fine-tune hoặc hiệu chỉnh corruption bằng dữ liệu camera thật để giảm
domain gap.
