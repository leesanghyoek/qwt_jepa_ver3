# QWT–JEPA v3 cho ảnh thiếu nhiễu sáng và IMU

Source: [leesanghyoek/qwt_jepa_ver3](https://github.com/leesanghyoek/qwt_jepa_ver3).
Trên Kaggle, bắt đầu từ [Cell 1: clone GitHub](KAGGLE_TRAIN_CELLS.md#cell-1--clone-source-từ-github-và-ghi-lại-commit),
sau đó chạy lần lượt các cell train và đánh giá.

Lần thử Kaggle hiện tại dùng `configs/kaggle_tartanair_v2.yaml`: phase 1
**8.000 update, neo 0,45, detail 2,0**; phase 2 **5.000 update, encoder skip tắt**,
head đọc hệ số input, beta 0,05 / detail 2,0 / variation 0,5 / energy 1,0.
Dùng OUT mới `outputs/p1_detail2_trial` để so với run phase 1 detail 0,5 trước đó.
Notebook cũ cần thay toàn bộ Cell 4 theo tài liệu; các override cũ không tự mất
khi clone source mới. Các sơ đồ bên dưới mô tả cả đường skip tùy chọn; đường đó
không hoạt động trong recipe Kaggle này. Lưu archive ở Cell 14 sau phase 1 và
sau phase 2 để giữ checkpoint qua phiên.

Repository này triển khai pipeline hai giai đoạn theo
`QWT_JEPA_JACOBIAN_MIGRATION_GUIDE.md` và
`QWT_JEPA_V3_DETAILED_ARCHITECTURE_DIAGRAMS.md`:

## Tài liệu chạy Kaggle và kiến trúc thực tế

- [Các cell Kaggle, train, resume, đồ thị và kết quả](KAGGLE_TRAIN_CELLS.md).
- [Sơ đồ kiến trúc chi tiết và quy trình train](KIEN_TRUC_VA_QUY_TRINH_TRAIN.md).
- [Kết quả kiểm tra code và giới hạn còn lại](CODE_REVIEW_KAGGLE.md).
- [**Kiến trúc: trước và sau**](KIEN_TRUC_TRUOC_VA_SAU.md) — so sánh từng thay đổi kèm phép đo.

Backend ảnh mặc định là `qwt_dualtree_hilbert`: **cặp Hilbert thật**, thiết kế
bằng liệt kê phân tích phổ (`tools/design_hilbert_pair.py`), 14 tap, đo được
**năng lượng tần số âm 0,0677** — thấp hơn 2,68 lần so với trần **0,1814** mà
backend cũ `qwt_dualtree_db4` đứng yên ở đó. Trần này là cấu trúc, không phải
vấn đề chọn wavelet: hai cây dùng *cùng* một filter lệch số nguyên luôn cho
`W_B(ω) = W_A(ω)·e^{-jωd}`, nên dư lượng chỉ phụ thuộc phép dịch — đo được
**đúng 0,1804 cho db2…db20, mọi symlet và mọi coiflet**. Backend cũ vẫn chọn
được để chạy ablation. Xem `tests/test_qwt_analyticity.py`.

## Thay đổi lần này — và những gì nó làm mất hiệu lực

Mỗi thay đổi đều kèm phép đo chứ không phải lời khẳng định.

| | Trước | Sau | Đo bằng |
|---|---|---|---|
| **QWT** | 4 cây db4 lệch 1 mẫu. Không phải cặp Hilbert; năng lượng tần số âm **0,1814** | Cặp Hilbert thiết kế riêng, 14 tap, **0,0677** (tốt hơn 2,68×) | `tests/test_qwt_analyticity.py` |
| **Blur ảnh** | bốc ngẫu nhiên, độc lập với IMU | **giữ nguyên** (quyết định 23/09), nhưng đường nối IMU đã dựng xong và bật được bằng `motion_from_imu: true` | `tests/test_imu_motion_blur.py` |
| **Jacobian** | 1 hướng Rademacher, phạt đẳng hướng, trọng số `1e-4` (trơ) | `log(g_nhiễu / g_tín hiệu)`, không thứ nguyên, trọng số `0,05` | `tests/test_sensitivity_ratio.py` |
| **Loss chi tiết ảnh** | chấm trên 48 kênh hệ số decoder xuất ra — QWT dư 4 lần nên decoder hạ được loss bằng năng lượng ảnh **không hiện ra** | chấm trên **ảnh khôi phục** (`image_detail_source: restored_image`); modulus `\|q\|` đã thử ở p5 và bỏ vì sinh **sọc** | `tests/test_image_detail_source.py` |
| **Decoder ảnh** | từ latent 16×16 dựng lên 48 kênh hệ số QWT rồi synthesis; ba biến thể đều dừng ở cùng một mức chi tiết | **ResNet trên pixel**: ảnh mờ ở độ phân giải đầy đủ (skip) + latent JEPA → phần residual cộng vào ảnh mờ (`image_decoder: resnet_pixel`); tái tạo đúng chỗ nhiều đường nét hơn (0,309 → 0,359) | `tests/test_resnet_decoder.py` |

**Phải train lại phase 1.** Biểu diễn đầu vào đã đổi (bộ lọc wavelet khác) và
phân phối blur đã đổi, nên checkpoint phase 1 cũ không còn so sánh được. Đây là
đứt gãy thật, không phải đổi tên: `model.image_transform` là một phần của
configuration hash.

**Chạy ba lệnh này trước khi train:**

```bash
# 1. Xác nhận hình học gyro→camera trên chính dataset của bạn
python3 tools/imu_blur_axis_check.py --data-root <root> --trajectories 12

# 2. Xem lại thiết kế bộ lọc (và đổi bậc nếu muốn)
python3 tools/design_hilbert_pair.py --orders 4 6 7 8

# 3. Toàn bộ test
python3 -m pytest tests/ -q
```

**Ablation có sẵn, không cần sửa code:** đặt `model.image_transform:
qwt_dualtree_db4` để quay về transform cũ, và `corruption.image.motion_from_imu:
true` để nối blur với IMU. Mặc định hiện tại là transform Hilbert + blur độc lập
với IMU.

## Vì sao ảnh vẫn mờ, và vì sao p5 ra sọc

**1. Loss chấm sai chỗ.** QWT giữ 4 cây và synthesis lấy trung bình 4 cây, nên 48
kênh hệ số decoder xuất ra dư 4 lần: phần nằm trong null space của synthesis không
hiện lên ảnh. Trước đây mọi số hạng chi tiết ảnh chấm trên chính 48 kênh đó. Đo trên
một frame thật, giữ ảnh y nguyên (lệch tối đa 1e-7) mà vẫn hạ được modulus **63%**,
L1 hệ số **41%**, energy gap **68%**. Trong lúc train, p5 cất **34%** năng lượng chi
tiết vào phần vô hình đó. `phase2.image_detail_source: restored_image` phân tích lại
chính ảnh khôi phục rồi mới chấm, nên chỉ cái người xem thấy mới được tính. Gradient
đi vào phần vô hình giảm từ 50–67% xuống ~1e-7.

**2. Modulus và energy không xét pha.** Hai số hạng này chỉ hỏi "đủ năng lượng chi
tiết chưa", không hỏi "đặt đúng chỗ chưa". Một hệ số chi tiết lệch **đều** tổng hợp
ra đúng sọc chu kỳ 2 px: LH → sọc ngang, HL → sọc dọc, HH → ô bàn cờ. Decoder đọc
latent 16×16 tạo ra những trường trơn như vậy rất dễ, nên đó là "chi tiết" rẻ nhất nó
mua được. Về lý thuyết, modulus tối ưu ở đúng độ mạnh cạnh (s = 1,0 so với 0,5 của L1
hệ số khi cạnh lệch ±0,75 px), nhưng decoder tìm ra một nghiệm rẻ hơn thế.

A/B cục bộ, cùng phase 1, 600 update phase 2, 32 frame valid (kịch bản full), đo
trên ảnh khôi phục (`validation_image_*_power`, 1,00 = như ảnh clean):

| nhánh | PSNR | cạnh thật (chu kỳ 4–16 px) | sọc (chu kỳ 2 px) |
|---|---|---|---|
| input | 11,13 | 0,116 | 0,17 |
| p5: modulus + energy, chấm trên hệ số decoder | 16,46 | 0,140 | 0,34 (34% vô hình) |
| modulus + energy, chấm trên ảnh | 16,38 | 0,138 | **3,21** |
| modulus một mình, chấm trên ảnh | 16,41 | 0,140 | 0,54 |
| **L1 hệ số + energy, chấm trên ảnh** (recipe hiện tại) | 16,40 | 0,139 | **0,14** |

Bịt lỗ hổng mà vẫn giữ modulus + energy thì sọc tăng 9 lần. Chỉ L1 hệ số mới giữ
được pha, và với nó thì energy không mua được sọc. **Năng lượng cạnh thật ở cả năm
nhánh đều bằng nhau (0,14× ảnh clean)**: với decoder hệ số, không số hạng mean/moment
nào ở đây làm ảnh nét thật. Thứ tiếp theo đo được là **kiến trúc decoder** — xem
[Decoder ảnh ResNet](#decoder-ảnh-resnet-p7).

Chỉ đổi phase 2, nên `configuration_hash` phase 1 không đổi và checkpoint phase 1
dùng lại được (`REUSE_PHASE1_FROM` trong Cell 4 của notebook). Báo cáo train
(`tools/training_report.py`) giờ in năng lượng cạnh/sọc và tự cảnh báo khi sọc > 1,5×.

## Decoder ảnh ResNet (p7)

Ý tưởng: QWT + Jacobian đưa ảnh về miền đường nét, JEPA học ảnh mờ và ảnh nét tương
ứng với nhau ra sao (student đọc ảnh mờ, teacher EMA đọc ảnh sạch), rồi một **ResNet
có skip từ ảnh mờ** kết hợp ảnh mờ với đặc trưng JEPA để dựng lại ảnh nét. Decoder cũ
chỉ nhìn ảnh qua latent 16×16 (mỗi ô là một khối 16×16 pixel) và 48 kênh hệ số; ResNet
đọc **thẳng ảnh mờ ở 256×256**, còn latent cho biết cảnh sạch nên trông thế nào.

A/B cục bộ: cùng phase 1, cùng loss (L1 pixel + L1 hệ số chi tiết chấm trên ảnh ·
2,0 + energy · 1,0), 600 update, cùng 32 frame valid (kịch bản full):

| decoder ảnh | PSNR | SSIM | đường nét **đúng chỗ** (chu kỳ 4–16 px) | sai số dải đó | sọc 2 px |
|---|---|---|---|---|---|
| input (không làm gì) | 11,13 | 0,501 | 0,285 | 0,546 | 0,17 |
| hệ số QWT (cũ) | **16,40** | 0,558 | 0,309 | 0,520 | 0,14 |
| **ResNet + latent JEPA** | 16,04 | **0,572** | **0,359** | **0,465** | 0,17 |
| ResNet, bỏ latent | 14,91 | 0,577 | 0,402 | 0,429 | 0,17 |

"Đúng chỗ" là phần nội dung đường nét của ảnh sạch được tái tạo **đúng pha**
(`Re Σ F_out·F̄_clean / Σ |F_clean|²` trên dải 4–16 px, 1 = hoàn hảo). Chỉ đo năng
lượng thì không đủ: nhiễu và sọc cũng làm năng lượng tăng. Ở đây năng lượng tăng
**và** sai số dải giảm, nên phần tăng thêm là đường nét thật.

Đọc bảng cho đúng:

- ResNet hơn decoder hệ số dù **ít tham số hơn** (0,80 M so với 1,58 M). Ba decoder hệ
  số trước đó dừng ở cùng một sàn vì cả ba đều chỉ chạm tới ảnh qua hệ số dựng từ
  latent 16×16; giới hạn nằm ở kiểu decoder.
- PSNR của ResNet thấp hơn 0,36 dB nhưng SSIM cao hơn: nó giữ cấu trúc tốt hơn, còn
  sai số độ sáng/màu còn hơi lớn ở ngân sách 600 update.
- **Bỏ latent** cho nhiều đường nét hơn nhưng PSNR tụt 1,1 dB. Phase 1 ở đây chỉ train
  **40 update**, latent gần như chưa học gì, nên phép thử này **chưa trả lời được**
  JEPA có giúp phần đường nét hay không. Run Kaggle với phase 1 đầy đủ 5000 update mới
  trả lời được; `delta_report.py --ablate-latent` đo trên checkpoint thật.
- Mỗi nhánh chạy một lần, 600 update, 32 frame: chiều hướng rõ, chênh lệch nhỏ giữa
  hai nhánh ResNet có thể là nhiễu.

QWT và Jacobian vẫn ở đúng vai trò: QWT Hilbert là miền đầu vào của encoder và là
miền chấm loss chi tiết (trên ảnh khôi phục); Jacobian (tỉ số độ nhạy) ép encoder ở
phase 1 phản ứng với đường nét chứ không với nhiễu. Chỉ đổi phase 2, nên phase 1
dùng lại được.

## Luồng tổng quát

```mermaid
flowchart LR
    CL["Ảnh + IMU<br/><b>SẠCH</b>"]
    GY["gyro sạch"]
    KER["tích phân trên<br/>phơi sáng<br/>⇒ <b>kernel blur</b>"]
    NZ["Ảnh + IMU<br/><b>NHIỄU</b>"]
    BB["<b>①</b> QWT Hilbert + Haar<br/>2 encoder + fusion"]
    Z["<b>②</b> ZI · ZU"]
    L1["<b>Phase 1</b><br/>JEPA · VICReg<br/>Jacobian tỉ số · neo"]
    D["<b>③ Phase 2</b><br/>ResNet trên pixel<br/>ảnh mờ + ZI"]
    R["Ảnh + IMU<br/>phục hồi"]
    CL --> GY --> KER --> NZ --> BB --> Z
    CL --> NZ
    Z --> L1
    Z --> D --> R
    CL -. "teacher EMA · đích của neo" .-> L1
    classDef d fill:#e8f5e9,stroke:#43a047,color:#1a1a1a
    classDef l fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#1a1a1a
    class GY,KER d
    class Z l
```

Khối ① là thay đổi cấu trúc lớn nhất. Trước đây blur được **bốc ngẫu nhiên** độc
lập với IMU, nên cửa sổ IMU không mang một bit nào về cách ảnh bị làm mờ — có thể
xoá hẳn nhánh IMU mà metric ảnh gần như không đổi. Giờ blur là chuyển động thật
tích phân trên thời gian phơi sáng, còn nhánh IMU chỉ nhận **bản nhiễu** của
chính chuyển động đó. Khoảng cách giữa hai thứ là bài toán.

Phase 1 **có** một decoder phụ làm *neo*: nó chấm điểm latent trên hệ số wavelet
sạch với trọng số `0.45`, rồi bị vứt bỏ khi phase 1 kết thúc. Không có neo này
JEPA vẫn đạt loss thấp trên một latent đã ném đi tín hiệu — đó đúng là kết quả
của lần train đầu tiên. Vẫn không có đường pixel-space trong phase 1
(`reconstruction_loss_weight: 0.0`).

Phase 2 tải checkpoint phase 1 hợp lệ, đóng băng encoder/fusion/normalizer, khởi
tạo **decoder hoàn toàn mới** và chỉ tối ưu hai decoder đó: ResNet trên pixel cho
ảnh, decoder hệ số Haar cho IMU.

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
    QW["<b>QWT dual-tree Hilbert</b> · 0 tham số<br/>3 màu × 4 băng × 4 thành phần<br/>Ci = 48 × 128 × 128"]
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
    PN["probe <b>NHIỄU</b><br/>clean + ε·(noisy−clean)"]
    PS["probe <b>TÍN HIỆU</b><br/>clean + ε·(clean−lowpass)"]
    FC["<b>FI_clean · FU_clean</b><br/>điểm gốc — đã tính sẵn"]
    JA(["<b>Jacobian tỉ số</b> · 0,05<br/>log(g_nhiễu / g_tín hiệu)<br/>chặn dưới · bật sau update 500"])
    ANL(["<b>Anchor loss · 0,45</b><br/>+ băng chi tiết 2,0"])
    CLEAN --> TE --> JE
    Z --> PR --> JE
    Z --> AN --> ANL
    CLEAN -. "hệ số wavelet SẠCH = đích" .-> ANL
    Z --> VC
    F --> VC
    CLEAN --> PN & PS
    PN --> JA
    PS --> JA
    FC -. "đo TRƯỚC fusion" .-> JA
    classDef lat fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#1a1a1a
    classDef p1 fill:#fff8e1,stroke:#f9a825,color:#1a1a1a
    classDef loss fill:#fce4ec,stroke:#d81b60,color:#1a1a1a
    class Z,F,FC lat
    class CLEAN,TE,PR,AN,PN,PS p1
    class JE,VC,JA,ANL loss
```

Khối Jacobian đo **trước** fusion (`target: online_dense_before_fusion`), không
đo trên latent, và đo quanh điểm gốc **sạch** — mà feature của nó (`FI_clean`)
phase 1 đã tính sẵn cho variance/covariance, nên chi phí thêm là **hai** forward
encoder chứ không phải ba.

Hai probe thay vì một là cả điểm mấu chốt: bản cũ đo **một** hướng Rademacher rồi
tối thiểu hoá gain đó, tức phạt co đẳng hướng — bảo encoder bớt nhạy với *mọi
thứ*, kéo thẳng về collapse. Tỉ số hai gain thì **không thứ nguyên**: collapse
đưa cả hai về 0 và tỉ số đứng yên, nên nó nâng trọng số lên được thật.

#### 3. Phase 2 — khôi phục, backbone đóng băng

```mermaid
flowchart TB
    ZI["<b>ZI</b> · 128 × 16 × 16<br/>backbone ĐÓNG BĂNG"]
    ZU["<b>ZU</b> · 128 × 8<br/>backbone ĐÓNG BĂNG"]
    CI["<b>Ci</b> · hệ số QWT của ảnh mờ<br/>48 × 128 × 128"]
    SB["QWT synthesis<br/>tái tạo hoàn hảo"]
    IB["<b>Ảnh mờ</b> · 3 × 256 × 256"]
    RH["head conv 3×3 · 32 × 256 × 256"]
    RD["conv 4×4 stride 2 · 64 × 128 × 128"]
    RL["conv 1×1 + upsample<br/>→ 64 × 128 × 128"]
    RF["nối + conv 3×3 · 64 × 128 × 128"]
    RT["<b>8 khối residual</b><br/>conv–ReLU–conv + identity<br/>64 × 128 × 128"]
    RU["conv + pixel shuffle ×2<br/>32 × 256 × 256"]
    RS["nối với head · conv 3×3<br/>conv 3×3 <b>zero-init</b><br/>Δ = 3 × 256 × 256"]
    PI(("＋"))
    OI["<b>Ảnh phục hồi</b><br/>3 × 256 × 256"]
    AI["QWT analysis<br/>loss chi tiết chấm trên ảnh"]
    DU["<b>Decoder IMU</b> · 0,36 M · sub-pixel conv<br/>128×8 → 128×16 → 96×16<br/>→ 96×32 → 64×32 → 64×64 → 32×64"]
    HU["head conv 3×3 · <b>zero-init</b><br/>đọc cả x và Cu<br/>Δu = 12 × 64"]
    CU["<b>Cu</b> · hệ số của chính IMU nhiễu<br/>12 × 64"]
    PU(("＋"))
    SU["Haar synthesis"]
    OU["<b>IMU phục hồi</b><br/>6 × 128"]
    L1(["<b>Loss phase 2</b><br/>L1 pixel + SmoothL1 accel/gyro β=0,05<br/>+ L1 hệ số chi tiết LH/HL/HH trên ảnh · 2,0<br/>+ sai phân bậc một IMU<br/>+ khớp năng lượng đường nét · 1,0"])
    SK["<b>3 tầng encoder IMU</b> · skip"]
    GT{{"<b>SkipMerge có cổng</b><br/>cổng = sigmoid(conv(đường latent))<br/>x + cổng × conv(skip)"}}
    CI --> SB --> IB --> RH --> RD --> RF
    ZI --> RL --> RF --> RT --> RU --> RS --> PI --> OI --> AI --> L1
    RH -. "skip U-Net" .-> RS
    IB -. "không qua trọng số nào" .-> PI
    ZU --> DU --> HU --> PU --> SU --> OU --> L1
    SK --> GT --> DU
    CU -.-> PU
    CU --> HU
    classDef tf fill:#e8eaf6,stroke:#5c6bc0,color:#1a1a1a
    classDef lat fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#1a1a1a
    classDef p2 fill:#e8f5e9,stroke:#43a047,color:#1a1a1a
    classDef loss fill:#fce4ec,stroke:#d81b60,color:#1a1a1a
    class ZI,ZU lat
    class CI,CU,SB,SU,AI,IB tf
    class RH,RD,RL,RF,RT,RU,RS,PI,OI,DU,HU,PU,OU p2
    class GT p2
    class L1 loss
```

Nhánh ảnh là ResNet trên pixel: **ảnh mờ cho vị trí đường nét, latent cho biết cảnh
sạch nên trông thế nào**. Latent được upsample từ 16×16 lên 128×128 rồi trộn vào thân
ResNet; tần số cao đi qua đường pixel và skip U-Net ở 256×256, nên upsample latent
bằng bilinear không làm mất chi tiết. Nhánh IMU giữ decoder hệ số, với `SkipMerge`
có cổng: skip cấp độ phân giải, latent quyết định giữ cái gì.

Hai mũi tên nét đứt tới dấu cộng là hai đường **không đi qua trọng số nào**: ảnh mờ
và hệ số của IMU nhiễu cộng thẳng vào đầu ra. Đó là sàn identity — xem
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

**Ảnh — QWT dual-tree Hilbert, 1 mức.** Bốn cây chạy song song; dọc mỗi trục tín
hiệu đi qua cây `A` hoặc cây `B`, và cây B là **bản đảo thời gian** của cây A —
chính phép đảo đó tạo độ trễ nửa mẫu mà một phép dịch số nguyên không thể tạo ra.
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

- `ZI` là `16×16`, tức **mỗi ô latent phải mô tả một khối 16×16 pixel**. Đó là trần
  của những gì latent tự mang được. Decoder hệ số cũ chỉ chạm tới ảnh qua latent nên
  dừng ở trần này; ResNet trên pixel đọc thẳng ảnh mờ ở 256×256 nên vượt được
  (đường nét đúng chỗ 0,309 → 0,359). Cách khác là latent `32×32` (`encoders.py`,
  đổi `stride=2` của stage cuối thành `1`), nhưng cách đó phải train lại phase 1.
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

**Encoder sensitivity (khối Jacobian)** — *đã viết lại*. Bản cũ đo **một** hướng
Rademacher ngẫu nhiên rồi tối thiểu hoá gain đó: một phạt co **đẳng hướng**, tức
bảo encoder bớt nhạy với *mọi thứ*, kéo thẳng về collapse. Nó chỉ sống được ở
trọng số `1e-4`, nơi nó không tác động gì đo được.

Bản mới đo **hai** gain quanh cùng một điểm sạch, theo hai hướng có nghĩa:

- `g_noise` — hướng mà corruption **thực sự** đã đẩy mẫu này đi, `noisy − clean`.
- `g_signal` — phần tần số cao của mẫu sạch, tức thứ decoder phải dựng lại.

Loss là `log(g_noise) − log(g_signal)`, **không thứ nguyên**: collapse đưa cả hai
về 0 và tỉ số đứng yên, nên khác bản cũ, số hạng này không thưởng cho collapse và
nâng trọng số lên được thật (`weight_max: 0.05`). Có `floor_log_ratio` chặn dưới
để nó ngừng đẩy khi encoder đã đủ điếc — không có chặn thì mục tiêu vô hạn dưới.

Mẫu bị corruptor bỏ qua (`clean_probability`) không có hướng nhiễu để đo, nên bị
loại khỏi trung bình; `sensitivity_valid_fraction` báo tỉ lệ còn lại. Điểm gốc là
**đầu vào sạch**, mà feature của nó phase 1 đã tính sẵn cho variance/covariance,
nên chi phí thêm là hai forward encoder chứ không phải ba.

**Đo 30 update đầu trên dữ liệu thật (24.314 sample TartanAir):**

| nhánh | `g_noise` | `g_signal` | tỉ số | đọc là |
|---|---|---|---|---|
| ảnh | 1,1 – 25 | 21 – 101 | **0,05 – 0,26** | đã nhạy với tín hiệu hơn nhiễu 4–20 lần |
| IMU | 0,9 – 12,7 | 0,9 – 4,2 | **1,0 – 3,3** | **nhạy với nhiễu ngang hoặc hơn tín hiệu** |

Đây là kết quả đáng chú ý nhất của lần đo: **khối này tồn tại chủ yếu vì nhánh
IMU**, không phải nhánh ảnh. Encoder IMU đang để nhiễu lấn át tín hiệu, đúng với
bất đối xứng đã biết của bài toán — ảnh cần *thêm* tần số cao, IMU cần *bớt*. Một
con số như vậy không thể đọc ra từ khối Jacobian cũ, vì nó chỉ trả về một gain
đơn không có gì để so sánh.

### Phase 2 — decoder

Backbone (transform + 2 encoder + fusion) **đóng băng ở chế độ eval**. Chỉ hai
decoder được cập nhật.

**Decoder ảnh — ResNet trên pixel** (`image_decoder: resnet_pixel`, 799.811 tham số):

| Bước | Shape |
|---|---|
| vào | ảnh mờ `[3, 256, 256]` (synthesis của `C_in`, tái tạo hoàn hảo) + `ZI [128, 16, 16]` |
| `head` conv 3×3 + ReLU | `[32, 256, 256]` |
| `down` conv 4×4 stride 2 + ReLU | `[64, 128, 128]` |
| `latent` conv 1×1 + upsample | `[64, 128, 128]` |
| `fuse` nối + conv 3×3 | `[64, 128, 128]` |
| `trunk` 8 khối residual (conv–ReLU–conv + identity) | `[64, 128, 128]` |
| `up` conv 3×3 → pixel shuffle ×2 + ReLU | `[32, 256, 256]` |
| `tail` nối với `head` (skip U-Net), conv–ReLU–conv (**zero-init**) | `Δ [3, 256, 256]` |
| cộng ảnh mờ | `ảnh mờ + Δ` |
| QWT analysis (chỉ để chấm loss chi tiết) | `[48, 128, 128]` |

Conv cuối zero-init nên ở update 0 đầu ra **đúng bằng** ảnh mờ — cùng sàn identity
với decoder hệ số (`tests/test_resnet_decoder.py`).

**Decoder IMU** (và decoder ảnh cũ `qwt_coefficients`, vẫn chọn được): Nâng kích thước bằng **sub-pixel conv** (`Upsample`: conv
mở rộng kênh ×4 cho ảnh / ×2 cho IMU rồi `pixel_shuffle`), **không** dùng nội suy
bilinear — bilinear là bộ lọc thông thấp nên không sinh được tần số cao.

| Bước | Decoder ảnh | Decoder IMU |
|---|---|---|
| vào | `ZI [128, 16, 16]` | `ZU [128, 8]` |
| `shuffle2` | `[128, 32, 32]` | `[128, 16]` |
| `up2` | `[96, 32, 32]` | `[96, 16]` |
| **`merge2`** skip có cổng | `+ [96, 32, 32]` từ encoder | `+ [96, 16]` |
| `shuffle1` | `[96, 64, 64]` | `[96, 32]` |
| `up1` | `[64, 64, 64]` | `[64, 32]` |
| **`merge1`** skip có cổng | `+ [64, 64, 64]` từ encoder | `+ [64, 32]` |
| `shuffle0` | `[64, 128, 128]` | `[64, 64]` |
| `up0` | `[32, 128, 128]` | `[32, 64]` |
| **`merge0`** skip có cổng | `+ [32, 128, 128]` từ encoder | `+ [32, 64]` |
| `head` conv 3×3 — đọc **cả `C_in`** | `Δ [48, 128, 128]` | `Δ [12, 64]` |
| cộng hệ số input | `C_in + Δ` | `C_in + Δ` |
| synthesis | `[3, 256, 256]` | `[6, 128]` |

Head nhận `concat(x, C_in)`, không chỉ `x`. Nếu chỉ đọc latent thì `Δ = f(Z)` và
decoder **không biểu diễn được phép khử nhiễu** — muốn trừ bớt nhiễu thì phải đọc
được nó, mà latent được huấn luyện để đoán latent của tín hiệu *sạch*, tức để vứt
bỏ hiện thực của nhiễu. Đo được: head cũ giảm được **0%** sai số trên bài co giãn
wavelet, head mới giảm **100%**.

Ba khối `merge*` dùng `x + sigmoid(conv_gate(x)) × conv_skip(skip)`: cổng sinh từ
**đường latent** nên nó đổi theo từng vị trí và từng kênh — latent quyết định cho
bao nhiêu skip đi qua. `conv_skip` zero-init nên đóng góp ban đầu bằng đúng 0.

`head` được **zero-init** ở chế độ residual, nên trước khi học gì đầu ra bằng
đúng đầu vào — và sàn identity đó **sống sót qua cả ba điểm nối** (đo được: lệch
tối đa 3e-7, tức chỉ là làm tròn float32). Nếu lưới không chia hết cho 8, `resize` bilinear xử lý phần lẻ
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
| decoder neo (ảnh + IMU) | 1.856.124 | 1, rồi vứt — **không skip, không residual** |
| decoder ảnh phase 2 — ResNet pixel | 799.811 | 2 |
| decoder IMU phase 2 | 358.012 | 2 |
| **decoder phase 2 (tổng)** | **1.157.823** | 2 |
| *(decoder ảnh cũ `qwt_coefficients`, nếu chọn)* | *1.577.392* | 2 |

Decoder neo giữ kiến trúc hệ số, không skip, không residual, vì skip và residual
đều là đường vòng quanh latent — mà việc của neo là ép thông tin VÀO latent.

Teacher EMA là bản sao của hai encoder (1.001.856 tham số) nhưng **không nhận
gradient**, nên không tính vào đây.

Thứ **duy nhất** đi từ phase 1 sang phase 2 là 1.333.120 tham số backbone. Tất cả
predictor và decoder neo đều bị bỏ; phase 2 dựng decoder hoàn toàn mới với seed
riêng (`decoder_initialization_seed`).

## Hai quyết định thiết kế quan trọng

**Decoder phase 2 dự đoán hiệu chỉnh, không dự đoán thay thế.** ResNet ảnh cộng
`Δ` vào chính ảnh mờ, conv cuối zero-init. Với
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
- `qjepa/models/decoders.py`: `PixelResNetDecoder` cho ảnh (ảnh mờ + `ZI` → residual
  trên pixel, khi `image_decoder: resnet_pixel`); decoder hệ số nhận `ZI/ZU`, và khi
  `encoder_skips` bật thì nhận
  thêm ba tầng trung gian của encoder qua `SkipMerge` **có cổng** — cổng sinh từ
  đường latent nên latent quyết định cho bao nhiêu skip đi qua ở từng vị trí.
  Khởi tạo sao cho đóng góp skip ban đầu bằng 0, nên sàn identity không bị phá. Upsample bằng sub-pixel conv (pixel
  shuffle) thay cho nội suy bilinear, vì bilinear là bộ lọc thông thấp nên không
  sinh được tần số cao. Ở chế độ residual, head zero-init và cộng vào hệ số input.
- `qjepa/corruptions/motion.py`: blur chuyển động tích phân từ gyro sạch. **Tắt
  mặc định** (`motion_from_imu: false`): dự án này coi ảnh mờ là do **camera** và
  nhiễu IMU là do **môi trường**, hai nguyên nhân độc lập. Đánh đổi phải ghi rõ —
  khi độc lập, cửa sổ IMU không mang **một bit nào** về cách ảnh bị làm mờ, nên
  nhánh IMU không đóng góp được gì cho việc khôi phục *ảnh*; nó vẫn học khôi phục
  chính nó. Bật `motion_from_imu: true` để nối lại. Hình học được **đo**
  chứ không giả định (`tools/imu_blur_axis_check.py`): hệ `lcam_front` trùng hệ
  body của IMU, tương quan ≥ 0,9998 và slope 1,00 trên 12 trajectory. Do đó
  `gyro_z` (yaw) → dịch ngang, `gyro_y` (pitch) → dịch dọc, `gyro_x` (roll) là
  xoay trong mặt phẳng — **không** gộp vào kernel vì một kernel tích chập không
  biểu diễn được nó, và được báo riêng ở `roll_radians`.

  Phân phối thực tế, đo trên 400 sample train ngẫu nhiên: trung vị **1,08 px**,
  p90 **3,11 px**, tối đa **9,62 px**; `Data_easy` trung bình 1,06 px,
  `Data_hard` 2,35 px. Bốc ngẫu nhiên kiểu cũ cho trung bình ~1,8 px, nên độ nặng
  gần tương đương — cái đổi là blur **tương quan** với IMU chứ không phải độc
  lập. `angular_gain` chỉnh độ nặng mà **không** phá tương quan đó.
- `qjepa/corruptions/image.py`: blur quang học, giảm độ phân giải, exposure thấp,
  gamma, white balance, vignette, shot/read/row noise, hot pixel, lượng tử và
  JPEG. Mỗi frame bốc tham số riêng nên độ sáng và độ nhoè thay đổi giữa các
  frame. `exposure_tracks_darkness` nối thời gian phơi sáng với độ tối: frame tối
  hơn nghĩa là màn trập mở lâu hơn, nên thiếu sáng và nhoè mạnh đến **cùng lúc**
  thay vì được bốc độc lập.
- `qjepa/corruptions/imu.py`: bandwidth blur, scale/cross-axis error, white noise
  có gain thay đổi theo thời gian, rung băng hẹp 8–45 Hz, spike, dropout và lượng
  tử. Bias instability bị **gate** sau `wander_probability: 0.25`: phần lớn window
  dao động *quanh* tín hiệu sạch, chỉ thỉnh thoảng mới lệch đi.
- `qjepa/training/phase1.py`: noisy-to-clean latent prediction, teacher EMA,
  variance/covariance trên tám raw maps, và Jacobian **bất đẳng hướng** trước
  fusion — tỉ số độ nhạy nhiễu / độ nhạy tín hiệu, xem mục trên.
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
- `configs/kaggle_tartanair_v2.yaml`: thử phase 1 detail 2,0 trong 8.000 update;
  phase 2 tắt skip, 5.000 update, workers 0 và pin memory tắt trên Kaggle.

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
