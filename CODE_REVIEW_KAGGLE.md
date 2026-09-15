# Kiểm tra code phục vụ train trên Kaggle

Ngày: 15/09/2026. Phạm vi sửa: `qwt_jepa_version3`.

## Kết luận

Đã kiểm tra luồng data → latent JEPA → decoder → checkpoint/resume → evaluation
và bổ sung xuất báo cáo. Pipeline giữ hai giai đoạn riêng: phase 1 không decoder,
phase 2 chỉ train decoder từ latent của backbone frozen. Thực thi hỗ trợ một hoặc
hai GPU trong một process; batch trong config vẫn là batch toàn cục.

**41 tests đã qua trên CPU**, gồm luồng CLI hai phase và mô phỏng ngắt session
sau checkpoint rồi resume. Đã sinh và mở kiểm tra PNG, cùng CSV/JSON/NPZ.
14 Python cell trong [hướng dẫn Kaggle](KAGGLE_TRAIN_CELLS.md) đã kiểm tra cú pháp.
Một test nữa cần hai GPU CUDA thật nên **báo skip** trong môi trường này.
Đây chưa phải chạy notebook trong môi trường Kaggle hoặc train model chính.

## Những lỗi/phần thiếu đã sửa

| Vấn đề | Sửa đổi | Kiểm chứng |
| --- | --- | --- |
| Timestamp Unix bị cast float32, mất khoảng cách IMU 100 Hz | Giữ float64 qua collate và inference; tính metadata từ chênh lệch double; kiểm tra finite, tăng nghiêm ngặt và camera nằm trong window | Regression Unix timestamp ~1,75 tỷ giây |
| Resume GPU có thể đưa RNG tensor về CUDA trong khi CPU RNG yêu cầu tensor CPU | Chuyển RNG state về `.cpu()` trước restore | Kiểm tra code; chưa chạy CUDA trong môi trường này |
| Resume phase2 quên best score và phụ thuộc đường dẫn parent | Lưu/khôi phục best score; đối chiếu backbone hash; yêu cầu/copy best checkpoint cũ khi chuyển thư mục | Test ngắt sau update1, resume update2 với validation xấu hơn; best file giữ nguyên |
| Có thể chuyển phase từ checkpoint latent trung gian | Yêu cầu hoàn tất ngân sách phase1, gate PASS, hash config/manifest/backbone và normalization khớp | End-to-end checkpoint loading và kiểm tra điều kiện trong source |
| Resume phase1 chưa kiểm tra dữ liệu/normalization hiện tại | Kiểm tra manifest và normalizer trước load state | Kiểm tra luồng resume |
| Hash manifest bỏ qua nhiều trường timestamp/split và normalization | Hash toàn bộ PairedSample, split và normalization; schema2 | Regression thay timestamp hoặc mean phải bị từ chối |
| Split rõ ràng chỉ có ở một phần trajectory bị bỏ qua | Từ chối layout vừa có split vừa thiếu split | Regression partial split hints |
| Rank validation tính riêng từng minibatch rồi lấy trung bình | Bank cố định trải giữa trajectory/time; nối feature cả bank rồi tính | Helper bank test, luồng CLI và code review |
| Feature hằng số có effective rank trả về1 | Trả về0 khi tổng singular values bằng0 | Regression constant features |
| Thiếu diagnostics sau LayerNorm | Ghi std/rank raw và normalized, RMS raw; vẽ riêng `latent_diagnostics.png` | JSONL và PNG thực từ smoke |
| Full evaluation đôi khi giữ input clean do xác suất5% | Tắt `clean_probability` ở valid/test; giữ scenario clean riêng | Luồng dataset/evaluation |
| Không biết model cải thiện so với input bao nhiêu | Thêm baseline noisy cho image MAE/PSNR/SSIM, IMU RMSE/MAE/bias/variation | Test evaluation overlap và CLI reports |
| Không có ảnh/đồ thị sau train | Tự vẽ sau mỗi phase; thêm `plot-training`; evaluate xuất panel ảnh, trace IMU, metrics và arrays | PNG mở được, CSV/JSON/NPZ đủ nội dung |
| Resume hoặc chạy mới có thể trộn log | Từ chối chạy mới vào thư mục đã có run; resume bỏ các record mới hơn checkpoint | Test overwrite guard; kiểm tra source |
| Config sai `run_kind` hoặc hạ statistics floor có thể lách B8 | Chỉ nhận main/smoke; main physical batch≥8; kiểm tra số update/batch/interval | Config tests và validation code |
| Gradient norm tràn vô hạn có thể qua bước clipping | Kiểm tra norm hữu hạn và các gradient trước optimizer step | Luồng optimizer/forward-backward tests |
| CSV inference không đọc lại header do chính nó xuất | Hỗ trợ header6/7 cột; giữ timestamp float64 | CSV round-trip regression |
| Loader evaluation chưa đối chiếu decoder hash | Kiểm tra decoder state so với metadata | Strict checkpoint round-trip |
| Chỉ chạy được một GPU trong khi Kaggle cấp T4 x2 | Thêm `qjepa/execution.py`: `runtime.gpu_count` (`auto`/1/2) và cờ `--gpus` cho train-phase1/phase2/evaluate; bọc forward bằng `DataParallel` | Test chọn device, từ chối request không hợp lệ, và test train hai GPU thật (skip khi thiếu phần cứng) |
| Chia batch theo GPU có thể làm hỏng thống kê chống collapse | Chỉ gather dict tensor feature; variance/covariance và JEPA tính sau khi gộp đủ batch, không lấy trung bình loss theo từng GPU | Test hai nửa batch hằng số cho variance khác hẳn cả batch; test wrapper scatter/gather giữ nguyên loss và gradient ở B8 |
| `DataParallel` có thể làm key checkpoint thành `module.*` hoặc để teacher nhận gradient | Optimizer và checkpoint dùng module gốc chưa bọc; EMA teacher update ngoài forward song song | Test không có key `module.` trong payload; test hai GPU kiểm tra teacher không có gradient |
| Số GPU có thể bị coi là thay đổi semantics và chặn resume | Loại topology khỏi configuration hash; ghi riêng vào `metadata.execution` | Test hash phase1/phase2 không đổi khi `gpu_count` đổi |
| Resume trên máy ít GPU hơn có thể lỗi khi khôi phục RNG CUDA | Bỏ qua state của device không tồn tại | Test restore chỉ gọi set_rng_state cho device đang có |
| Test multi-GPU dùng `request` làm tham số `parametrize` khiến **toàn bộ suite không collect được** | Đổi tên tham số thành `requested` (`request` là fixture dành riêng của pytest) | Suite chạy lại đầy đủ: 41 passed, 1 skipped |
| Cell 3 chọn **mount** làm `DATA_ROOT`; với dataset thật mount chứa cả `tartanair-v2` lẫn `tartanair-v2-jepa/train` nên run luôn hỏng ở cell 5 | Dò theo `<root>/<env>/<difficulty>/<Pxxx>` rồi đếm trajectory từng root và chọn root nhiều nhất; kiểm tra split hint đồng nhất ngay tại cell 3 | Chạy logic cell 3 trên bản sao có đúng cái bẫy: chọn đúng `tartanair-v2` |
| Lỗi "Mixed explicit and missing split directories" không nói root nào sai | Nêu số lượng và ví dụ đường dẫn của cả hai nhóm, kèm cách sửa | Test nêu tên thư mục vi phạm |
| Chưa có config Kaggle; người dùng phải tự sửa `data.root`/`device`/`num_workers` | Thêm `configs/kaggle_tartanair_v2.yaml` kế thừa `pipeline_v3.yaml`, chỉ đổi đường dẫn/thiết bị/worker | Train hai phase qua đúng config này cho loss trùng khít recipe gốc |
| Lệnh build-manifest trong `kaggle_dataset.md` dùng flag không tồn tại (`--root/--out/--window`) | Viết lại tài liệu dataset kèm lệnh đúng (`--config/--data-root/--output`) | Đối chiếu với `build_parser()` |

Hợp đồng checkpoint nay bao gồm `run_kind`, version recipe validation và số batch
validation phase2. Manifest dùng schema2. **Manifest/checkpoint cũ có thể bị từ
chối có chủ đích**; không sửa hash để vượt kiểm tra. Rebuild manifest và bắt đầu
run mới khi thay semantics. Hash manifest xác nhận metadata/path/pairing và
normalization, không băm toàn bộ byte ảnh/IMU; vẫn cần giữ dataset nguồn bất biến.

## Bằng chứng chạy

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/qjepa-matplotlib \
  python3 -m pytest -q --basetemp=outputs/kaggle_verification_current
# 41 passed, 1 skipped
# SKIPPED [1] tests/test_multi_gpu.py: Requires two real CUDA GPUs
```

Test tích hợp dùng dataset tổng hợp trên đĩa, đủ train/valid/test, timestamp Unix,
ảnh32×32 và IMU32 mẫu. Chạy phase1 một update; phase2 hai update với một lần ngắt
được mô phỏng. Validation score1 rồi2 trong test resume được **chủ động đặt để
kiểm tra chọn best**, không phải số đo chất lượng model. Evaluation ảnh/IMU cuối
vẫn tính từ output model thực trên dữ liệu tổng hợp.

Trong test đầy đủ,4 ảnh test tạo4 window32 mẫu, gộp còn80 hàng IMU duy nhất;
không đếm128 hàng có overlap như các mẫu độc lập.

Đã chạy thêm `evaluate --protocol --max-batches 2 --panels 1` trên CPU:
**10/10 scenario hoàn tất**, mỗi scenario đánh giá 2 ảnh tổng hợp và xuất báo
cáo. [Biểu đồ protocol smoke](outputs/kaggle_protocol_verification/comparison.png)
chỉ kiểm tra chức năng báo cáo trên subset, không phải kết quả khử nhiễu chính.

Ví dụ artifacts local, mang nhãn SMOKE:

- [Curves phase1](outputs/kaggle_verification_current/test_two_phase_cli_generates_k0/run/phase1/training_curves.png).
- [Latent diagnostics](outputs/kaggle_verification_current/test_two_phase_cli_generates_k0/run/phase1/latent_diagnostics.png).
- [Curves phase2](outputs/kaggle_verification_current/test_two_phase_cli_generates_k0/run/phase2/training_curves.png).
- [Panel ảnh](outputs/kaggle_verification_current/test_two_phase_cli_generates_k0/run/test_results/requested/images/frame_000000.png).
- [Metrics smoke](outputs/kaggle_verification_current/test_two_phase_cli_generates_k0/run/test_results/metrics.json).

Artifacts nằm trong `outputs/` đã ignore; chúng là bằng chứng chạy local, không
phải dữ liệu/pretrained weights bắt buộc để chạy trên Kaggle.
Các liên kết artifacts ở trên chỉ có trong workspace đã chạy test, không được
đưa lên GitHub. Chạy lại lệnh kiểm thử để tạo chúng trong checkout của bạn.

## Phần còn giới hạn hoặc chưa kiểm chứng

1. **Dataset Kaggle đã đối chiếu, nhưng chỉ gián tiếp:** `kaggle_dataset.md` nay
   ghi đủ layout, tần số và quy mô (14 env,166 trajectory,256×256,100/10 Hz).
   Đã dựng **bản sao đúng cấu trúc đó ở local** và chạy trọn build-manifest →
   phase1 → phase2 → evaluate, cộng test regression khoá lại hình học window.
   Vẫn **chưa chạy trên dataset thật**: bản audit của bạn mới đọc 3/166
   trajectory, nên nếu có trajectory lệch `len(ảnh) != len(cam_time)` thì chỉ
   phát hiện khi build-manifest quét toàn bộ (lệnh sẽ dừng và nêu tên).
2. **QWT reference chưa đạt audit:** transform hiện là bốn cây db4 dịch nguyên,
   periodic boundary, pack48 kênh. Round-trip và gradient đã qua, nhưng chưa
   kiểm chứng Hilbert pairs/QWT reference. Đã sửa nhãn G1 trong audit cũ thành
   PARTIAL. Không có learned Hamilton convolution trong source hiện tại.
3. **Chưa chạy GPU/Kaggle và main training:** môi trường này báo
   `torch.cuda.is_available() == False`. Chưa đo VRAM B8@256, tốc độ,10.000/5.000
   update hoặc khả năng khử nhiễu camera thật. Cell7 dành để đo recipe thực.
   Riêng đường chạy hai GPU: logic chọn device, ranh giới gather, quyền sở hữu
   optimizer/checkpoint và hash đã được test trên CPU bằng wrapper mô phỏng
   scatter/gather, **nhưng chưa chạy trên hai GPU CUDA thật**. Test
   `test_actual_two_gpu_training_and_single_gpu_checkpoint_loading` sẽ tự chạy
   khi notebook có T4 x2; chưa có bằng chứng về tốc độ hay VRAM thực tế khi chia
   hai GPU, và `DataParallel` vốn kém hiệu quả hơn DDP.
4. **Latent gate chỉ tương đối:** bank nhỏ bị giới hạn rank; PASS không bảo đảm
   thông tin có ích. FD hiện theo lịch update, chưa tự hoãn khi gate WARN. Chưa
   chạy control/treatment Jacobian để kết luận đóng góp.
5. **Sampler chưa bảo đảm độc lập chuyển động/time:** easy/hard cùng motion có
   thể được coi là hai trajectory key trong batch; không áp đặt khoảng cách
   thời gian tối thiểu. Split vẫn giữ chúng cùng một tập để tránh leakage.
6. **Validation và RAM:** validation phase2 mặc định bank32 mẫu, khác toàn split;
   final test mới đánh giá toàn bộ khi không giới hạn batch. Evaluator hiện giữ
   arrays gộp theo trajectory trong RAM; dữ liệu cực lớn cần streaming từng
   trajectory để giảm peak RAM. Chưa benchmark trên kích thước dataset Kaggle.
7. **Reproducibility:** đã kiểm tra sampler resume và checkpoint/score khi resume
   CPU; chưa chứng minh bitwise reproducibility giữa GPU khác nhau hoặc khi đổi
   phiên bản torch/CUDA. Hiện chưa ghi git/source hash tự động vào checkpoint;
   các cell Kaggle ghi commit vào `source_revision.txt`, kiểm tra commit trước
   khi resume và mang theo source/config trong archive của run.

## Tài liệu bàn giao

- [KAGGLE_TRAIN_CELLS.md](KAGGLE_TRAIN_CELLS.md):14 cell từ source/dataset tới
  train, resume, baseline metrics, panel và archive.
- [KIEN_TRUC_VA_QUY_TRINH_TRAIN.md](KIEN_TRUC_VA_QUY_TRINH_TRAIN.md): sơ đồ dữ liệu,
  encoder/fusion/teacher/predictor/decoder, shapes, loss, gradient và lịch train.
