# Train QWT–JEPA v3 trên Kaggle: các cell và kết quả minh họa

Chạy lần lượt các block **Python** dưới đây, mỗi block là một code cell.
Model học latent JEPA ở phase 1; hoàn tất và đạt latent gate mới chạy decoder
ở phase 2. Cả hai phase tự xuất đồ thị khi kết thúc. Các cell cuối tạo bảng
metric, so sánh baseline, ảnh khôi phục, trace IMU và file tải về.

Source: [leesanghyoek/qwt_jepa_ver3](https://github.com/leesanghyoek/qwt_jepa_ver3).
Cell 1 clone repository trực tiếp; không cần upload source thành Kaggle Dataset.

## Trước khi chạy

- Bật **Internet** để clone GitHub và cài dependency còn thiếu.
- Source được clone từ `https://github.com/leesanghyoek/qwt_jepa_ver3.git`, nhánh
  `main`; `qjepa/`, `configs/`, `tests/` nằm ngay tại root repository.
- Gắn dataset camera + IMU sạch, có timestamp đồng bộ, đã giải nén.
- Bật GPU trong notebook. Code chạy **một process**, FP32, dùng **một hoặc hai
  GPU** qua `DataParallel`; `runtime.gpu_count: auto` tự lấy tối đa hai GPU đang
  thấy, đúng với accelerator **T4 x2** của Kaggle. Chưa hỗ trợ DDP/`torchrun`/AMP.
  `batch_size` trong config luôn là **batch toàn cục**: B8 trên hai GPU nghĩa là
  mỗi GPU xử lý 4 mẫu rồi gather lại, loss variance/covariance vẫn tính trên đủ
  8 mẫu — chia GPU không làm đổi ngữ nghĩa thống kê batch.
  Cell đo thử sẽ kiểm tra B8 thực tế; không suy đoán trước VRAM/thời gian cần.
- Dataset chỉ đọc từ `/kaggle/input`; source làm việc, manifest, checkpoint và
  kết quả đặt trong `/kaggle/working/qwt_jepa_version3`.

**Dataset đã đối chiếu:** dataset TartanAir V2 của bạn (14 environment, 166
trajectory, ảnh 256×256, IMU 100 Hz, camera 10 Hz) **khớp với loader, không cần
sửa code**. Chi tiết đối chiếu từng điểm ở [kaggle_dataset.md](kaggle_dataset.md).
Dùng config sẵn [`configs/kaggle_tartanair_v2.yaml`](configs/kaggle_tartanair_v2.yaml).
Cell 3 vẫn tự dò `DATA_ROOT` vì đường dẫn mount của Kaggle có thể khác với đường
dẫn lúc bạn audit; không hardcode.

Layout loader hiện hỗ trợ, các tầng wrapper bên ngoài có thể có thêm:

```text
DATA_ROOT/
  [train|valid|val|test/]          # tùy chọn; nếu dùng thì mọi trajectory phải có split
    environment/
      Data_easy hoặc Data_hard/
        Pxxx/
          image_lcam_front/
            000000_lcam_front.png
            ...
          imu/
            acc.npy              # [N,3], m/s²; hoặc acc.txt
            gyro.npy             # [N,3], rad/s; hoặc gyro.txt
            imu_time.npy         # [N], giây; hoặc imu_time.txt
            cam_time.npy         # [M], giây; hoặc cam_time.txt
```

`M` phải bằng số ảnh theo thứ tự tên file; `acc` và `gyro` dùng cùng timeline.
Nếu không có split sẵn, code chia theo nhóm `(environment, trajectory_id)` trước
khi tạo window; nên có ít nhất 4 nhóm chuyển động để train có ≥2 nhóm và có
valid/test. Dataset của bạn không có thư mục split nên rơi vào nhánh này: 83
motion key được chia ≈67 train /8 valid /8 test, và `Data_easy`/`Data_hard` của
cùng chuyển động luôn ở cùng split.

**Cảnh báo về `DATA_ROOT`:** trỏ đúng thư mục mà con trực tiếp của nó là các
environment. Trong dataset của bạn, `tartanair-v2-jepa/train/` nằm cạnh
`tartanair-v2`; nếu chọn thư mục cha thì loader nhặt cả hai, một nửa trajectory
có split hint `train` còn một nửa không, và build-manifest dừng với
`Mixed explicit and missing split directories`. Không dùng camera có đơn vị độ/giây hoặc timestamp nano giây mà chưa
chuyển đổi về hợp đồng SI trên. Dataset chỉ chứa ZIP/ảnh, thiếu IMU/timestamp sẽ
không chạy được loader này; cần giải nén/bổ sung adapter dựa trên layout thực.

**Trạng thái backend ảnh:** code chạy được với backend shifted-db4 hiện tại,
chưa xác minh tương đương QWT reference. Xem
[báo cáo kiểm tra](CODE_REVIEW_KAGGLE.md) trước khi dùng kết quả để kết luận về QWT.

## Cell 1 — Clone source từ GitHub và ghi lại commit

Lần đầu để `SOURCE_REF='main'`. Khi resume ở session mới, thay bằng commit SHA
được lưu trong `source_revision.txt` của run trước. Cell không tự pull/update
source đang dùng giữa chừng. Khi muốn đổi phiên bản, dùng session mới.

```python
from pathlib import Path
import os
import shutil
import subprocess
import sys

INPUT = Path('/kaggle/input')
PROJECT = Path('/kaggle/working/qwt_jepa_version3')
REPO_URL = 'https://github.com/leesanghyoek/qwt_jepa_ver3.git'
SOURCE_REF = 'main'  # Khi resume: thay bằng SHA trong source_revision.txt của run trước

if not PROJECT.exists():
    subprocess.run(['git', 'clone', '--branch', 'main', REPO_URL, str(PROJECT)], check=True)
    subprocess.run(['git', '-C', str(PROJECT), 'checkout', '--detach', SOURCE_REF], check=True)

assert (PROJECT / '.git').is_dir(), 'PROJECT đã tồn tại nhưng không phải Git checkout; dùng session mới.'
def git_output(*args):
    return subprocess.check_output(['git', '-C', str(PROJECT), *args], text=True).strip()

assert git_output('remote', 'get-url', 'origin').removesuffix('.git') == REPO_URL.removesuffix('.git')
SOURCE_COMMIT = git_output('rev-parse', 'HEAD')
assert SOURCE_COMMIT == git_output('rev-parse', SOURCE_REF + '^{commit}'), (
    'Checkout hiện tại khác SOURCE_REF; dùng đúng SHA của run hoặc bắt đầu session mới.'
)
assert (PROJECT / 'qjepa/evaluation/reporting.py').is_file(), 'Checkout thiếu module reporting.'
(PROJECT / 'source_revision.txt').write_text(SOURCE_COMMIT + '\n', encoding='utf-8')
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ['MPLCONFIGDIR'] = str(PROJECT / 'outputs/matplotlib_cache')
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
print('Project:', PROJECT)
print('GitHub:', REPO_URL)
print('Source commit:', SOURCE_COMMIT)
```

Nếu clone báo lỗi DNS/network, kiểm tra Internet của notebook. Cell trên dành
cho repository public. Nếu repository private, cần quyền truy cập và cấu hình
xác thực qua Kaggle Secrets trước khi clone; không ghi token vào code cell.

## Cell 2 — Dependency, GPU và kiểm thử code

Chỉ cài các package còn thiếu; giữ torch/CUDA do notebook cung cấp. Nếu package
thiếu và Internet tắt, bật Internet hoặc gắn wheel phù hợp rồi cài offline.

```python
import importlib.util
import subprocess

packages = {
    'numpy': 'numpy>=1.24', 'PIL': 'Pillow>=10', 'yaml': 'PyYAML>=6',
    'scipy': 'scipy>=1.11', 'matplotlib': 'matplotlib>=3.7', 'pytest': 'pytest>=8',
}
missing = [requirement for module, requirement in packages.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)

import torch
import qjepa
assert Path(qjepa.__file__).resolve().is_relative_to(PROJECT)
assert torch.cuda.is_available(), 'Bật GPU rồi khởi động lại session notebook.'
print('Python:', sys.version)
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f'GPU {index}:', properties.name,
          '| VRAM GiB:', round(properties.total_memory / 2**30, 2))
print('Số GPU sẽ dùng (gpu_count=auto):', min(2, torch.cuda.device_count()))
subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=PROJECT, check=True)
```

Trên máy chỉ có một GPU, test train hai GPU thật sẽ báo `skipped` kèm lý do
`Requires two real CUDA GPUs`; đó là bỏ qua có chủ đích, không phải lỗi. Với
accelerator **T4 x2** test đó chạy thật: nó train phase 1 trên hai GPU, train
phase 2, lưu checkpoint rồi nạp lại và so khớp output. Nếu nó **fail** trên T4 x2,
dừng lại và chẩn đoán trước khi train chính; đừng đặt `gpu_count: 1` để né.

## Cell 3 — Tự dò và audit cấu trúc dataset

Cell tìm **thư mục root đúng** chứ không phải mount. Root đúng là thư mục có con
trực tiếp là các environment (`<root>/AmericanDiner/Data_easy/P000`). Chọn nhầm
mount cha sẽ nhặt luôn `tartanair-v2-jepa/train/` và hỏng split ở cell 5.

```python
from collections import Counter
from qjepa.data.tartanair import discover_trajectories, audit_trajectory

DATA_ROOT = None  # Điền Path('/kaggle/input/<slug>/tartanair-v2') để bỏ qua tự dò

def candidate_roots(base):
    """Đếm trajectory theo <root>/<env>/<difficulty>/<Pxxx>."""
    found = Counter()
    for timeline in base.rglob('imu_time.*'):
        if timeline.suffix not in {'.npy', '.txt'} or timeline.parent.name != 'imu':
            continue
        trajectory = timeline.parent.parent
        if not (trajectory / 'image_lcam_front').is_dir() or len(trajectory.parents) < 3:
            continue
        found[trajectory.parents[2]] += 1
    return found

if DATA_ROOT is None:
    counts = Counter()
    for mount in sorted(INPUT.iterdir()):
        if mount.is_dir():
            counts.update(candidate_roots(mount))
    assert counts, 'Không thấy trajectory nào; kiểm tra dataset đã gắn và đã giải nén.'
    for root, total in counts.most_common():
        print(f'{total:5d} trajectory  {root}')
    DATA_ROOT = counts.most_common(1)[0][0]
    if len(counts) > 1:
        print('CHON:', DATA_ROOT, '| bo qua cac root con lai o tren.')
        print('Neu chon sai, dien tay DATA_ROOT roi chay lai cell nay.')

DATA_ROOT = Path(DATA_ROOT)
trajectories = discover_trajectories(DATA_ROOT)
keys = [trajectory.key for trajectory in trajectories]
assert len(keys) == len(set(keys)), 'Trajectory key trùng nhau; DATA_ROOT đang gộp hai bản sao.'
hints = {trajectory.split_hint for trajectory in trajectories}
assert len(hints) == 1, f'Split hint không đồng nhất {hints}; DATA_ROOT đang trỏ quá cao.'
print('DATA_ROOT :', DATA_ROOT)
print('Trajectory:', len(trajectories))
print('Split hint:', hints.pop(), '(None = manifest tự chia theo motion key)')
print('Motion key:', len({trajectory.motion_key for trajectory in trajectories}))
print('Difficulty:', dict(Counter(trajectory.difficulty for trajectory in trajectories)))
for trajectory in trajectories[:3]:
    print(audit_trajectory(trajectory, window=128))
# build-manifest ở cell 5 sẽ đọc và kiểm tra toàn bộ trajectory.
```

Với dataset của bạn, cell in `166 trajectory`, `83 motion key`, `Data_easy 83 /
Data_hard 83`, split hint `None`, và mỗi audit cho `imu_rate_hz≈100`,
`camera_rate_hz≈10`, `images_match_camera_timestamps=True`, `enough_imu=True`.

## Cell 4 — Khóa cấu hình train hai giai đoạn

Main mặc định: 10.000 update phase 1, sau đó 5.000 update phase 2. Không dùng
`configs/smoke.yaml` để công bố kết quả model chính. Smoke chỉ kiểm tra wiring.
Giữ nguyên config từ đầu đến cuối run để hash checkpoint khớp khi resume.

```python
import json
import yaml
from qjepa.config import load_config, serializable_config, validate_config

MANIFEST = PROJECT / 'manifests/kaggle'
OUT = PROJECT / 'outputs/kaggle_main'
CONFIG = PROJECT / 'configs/kaggle.yaml'
# kaggle_tartanair_v2.yaml đã chứa recipe chính + thiết lập Kaggle; chỉ còn ghi
# đè các đường dẫn thực của session này (mount có thể đổi tên giữa các session).
config = serializable_config(load_config(PROJECT / 'configs/kaggle_tartanair_v2.yaml'))
config['data'].update(root=str(DATA_ROOT), manifest_dir=str(MANIFEST))
config['runtime'].update(output_dir=str(OUT))
validate_config(config)
CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
print(CONFIG.read_text())
print('Source commit:', SOURCE_COMMIT)
```

Có thể giảm `num_workers` về 0 khi gặp lỗi worker/RAM (notebook GPU của Kaggle
chỉ có 4 vCPU, config đặt sẵn 2). Không giảm batch phase 1 dưới 8 để né OOM; loss
thống kê batch sẽ đổi ý nghĩa — dùng `--gpus 2` thay vì hạ batch. Cell 7 đo chi
phí recipe thực trên GPU đang có. Mọi thay đổi kiến trúc/corruption/lịch train
cần run mới.

Config ghi ra `configs/kaggle.yaml` nằm trong `/kaggle/working` nên còn lại sau
khi session dừng; cell 8 dùng đúng file đó để resume.

## Cell 5 — Tạo manifest, kiểm tra split và normalization

```python
def run_cli(*arguments):
    # -u để log train hiện liên tục; không dùng shell và không bỏ qua mã lỗi.
    command = [sys.executable, '-u', '-m', 'qjepa', *map(str, arguments)]
    print('Running:', ' '.join(command))
    subprocess.run(command, cwd=PROJECT, check=True)

run_cli('build-manifest', '--config', CONFIG, '--data-root', DATA_ROOT, '--output', MANIFEST)
from qjepa.data import read_manifest
manifest = read_manifest(MANIFEST)
meta = manifest['meta']
print(json.dumps(meta, indent=2))

counts = meta['samples_per_split']
assert counts['train'] >= 8 and counts['valid'] >= 2 and counts['test'] > 0, counts
train_keys = {sample.trajectory_key for sample in manifest['samples']['train']}
assert len(train_keys) >= config['data']['minimum_trajectories_per_batch'], 'Train thiếu trajectory đa dạng.'
motion_groups = [{(row.environment, row.trajectory_id) for row in manifest['samples'][split]}
                 for split in ('train', 'valid', 'test')]
assert all(motion_groups[i].isdisjoint(motion_groups[j]) for i in range(3) for j in range(i + 1, 3))
if counts['valid'] < 64:
    print('Validation có <64 mẫu: bank dùng toàn bộ, rank bị giới hạn bởi N-1.')

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(list(counts), list(counts.values()))
axes[0].set(title='Paired samples per split', ylabel='Samples')
axes[1].bar(meta['normalization']['channels'], meta['normalization']['std'])
axes[1].set(title='Clean train IMU std (mixed SI units)')
fig.tight_layout()
plt.show()
```

**Kỳ vọng với dataset của bạn:** `rejected_centre ≈ 13` mỗi trajectory là bình
thường, không phải lỗi. Window 128 mẫu ở 100 Hz dài 1,27 s và phải bao quanh thời
điểm chụp, nên ~6–7 ảnh đầu và ~6–7 ảnh cuối mỗi trajectory không đủ IMU hai bên.
Con số 13 **không đổi theo độ dài trajectory**: với 284 ảnh là mất ≈4,6 %, với
515 ảnh chỉ ≈2,5 %. Tổng 166 trajectory sẽ mất khoảng 2.000 sample.

Chỉ cần lo khi `rejected_nonuniform` hoặc `rejected_outside` lớn, hoặc
`rejected_centre` vượt xa 13/trajectory — khi đó timestamp/đơn vị mới thực sự có
vấn đề. Không nới ngưỡng pairing chỉ để đủ số lượng mẫu.

## Cell 6 — Xem ảnh tối/mờ/nhiễu và IMU bị làm xấu

```python
import numpy as np
from qjepa.cli import _dataset

preview_dataset = _dataset(config, manifest, 'valid', fixed_realization=True)
sample = preview_dataset[len(preview_dataset) // 2]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name in zip(axes[:2], ('image_clean', 'image_noisy')):
    ax.imshow(sample[name].permute(1, 2, 0).numpy())
    ax.set_title(name)
    ax.axis('off')
axes[2].imshow((sample['image_clean'] - sample['image_noisy']).abs().mean(0),
               cmap='magma', vmin=0, vmax=1)
axes[2].set_title('Mean absolute RGB error [0,1]')
axes[2].axis('off')
plt.show()

t = sample['imu_times'].numpy()
t = t - t[0]
fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
for channel, ax in enumerate(axes.T.flat):
    ax.plot(t, sample['imu_clean_phys'][:, channel], label='clean')
    ax.plot(t, sample['imu_noisy_phys'][:, channel], alpha=0.7, label='corrupted')
    ax.set(title=('ax', 'ay', 'az', 'gx', 'gy', 'gz')[channel],
           ylabel='m/s²' if channel < 3 else 'rad/s', xlabel='Time (s)')
    ax.legend()
fig.tight_layout()
plt.show()
print(json.dumps(sample['corruption'], indent=2))
```

Đây là preview corruption, chưa phải output model. Chỉnh corruption ở cell 4
theo dữ liệu camera thực trước khi bắt đầu run, rồi chạy lại từ cell 4.

## Cell 7 — Đo một update phase 1 trên GPU với shape thật

Cell này tạo model tạm và bỏ sau khi đo; checkpoint phase 1 chính vẫn khởi tạo
mới bằng seed cấu hình. Đo ở thời điểm FD bật để tính thêm một encoder forward.
Kết quả là chi phí một update, chưa gồm validation, checkpoint và tải dữ liệu
của toàn run. OOM ở đây nghĩa là GPU chưa chạy được recipe đang chọn.

```python
import gc
import time
from qjepa.config import build_phase1_model, build_normalizer, seed_everything
from qjepa.cli import _loader
from qjepa.training.phase1 import Phase1Trainer

def measure_main_update():
    seed_everything(config['phase1']['initialization_seed'])
    ds = _dataset(config, manifest, 'train', fixed_realization=False)
    batch = next(iter(_loader(config, ds, config['phase1']['batch_size'], train=False)))
    model = build_phase1_model(config, build_normalizer(meta))
    trainer = Phase1Trainer(model, config, torch.device('cuda'), meta['manifest_hash'])
    trainer.successful_updates = config['encoder_sensitivity']['start_after_updates']
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics = trainer.step(batch)
    torch.cuda.synchronize()
    assert not metrics['skipped'], metrics
    print('Seconds / measured update:', time.perf_counter() - started)
    print('Peak allocated GiB:', torch.cuda.max_memory_allocated() / 2**30)
    print('Phase1 parameters:', sum(p.numel() for p in model.parameters()))
    print(metrics)

try:
    measure_main_update()
finally:
    gc.collect()
    torch.cuda.empty_cache()
```

## Cell 8 — Khôi phục một run từ session trước, nếu có

Để tiếp tục qua session Kaggle, lưu/tải về outputs ở cell 14 và gắn lại như một
input của session mới. `RESUME_RUN` trỏ tới thư mục chứa **phase1 và phase2**,
không trỏ riêng một file `.pt`. Với cùng session, để `None`; cell train tự tìm
`last.pt`. Source/config và mount dataset phải phù hợp với run cũ.

```python
RESUME_RUN = None  # Path('/kaggle/input/<previous-output>/outputs/kaggle_main')
if RESUME_RUN is not None:
    assert not OUT.exists(), 'Đã có run tại OUT; không ghi đè. Dùng bản đang có hoặc session mới.'
    previous_revision = Path(RESUME_RUN) / 'source_revision.txt'
    assert previous_revision.is_file(), 'Cần source_revision.txt của run cũ để xác định đúng code.'
    assert previous_revision.read_text().strip() == SOURCE_COMMIT, 'Chạy lại cell 1 với SHA của run cũ.'
    shutil.copytree(Path(RESUME_RUN), OUT)
OUT.mkdir(parents=True, exist_ok=True)
revision_file = OUT / 'source_revision.txt'
if revision_file.exists():
    assert revision_file.read_text().strip() == SOURCE_COMMIT, 'OUT thuộc commit khác; dùng đúng source.'
revision_file.write_text(SOURCE_COMMIT + '\n', encoding='utf-8')
for phase in ('phase1', 'phase2'):
    print(phase, 'resume from:', OUT / phase / 'last.pt',
          'exists:', (OUT / phase / 'last.pt').exists())
```

Log curves đầy đủ cần cả `train.jsonl`; tiếp tục phase 2 cần mang theo
`best_joint_validation.pt` để giữ checkpoint tốt nhất trước đó. Đổi đường dẫn
parent `.pt` được hỗ trợ bằng hash backbone; không tự thay đổi nội dung manifest
hoặc normalization. Với checkpoint tạo từ source cũ, kiểm tra hash có thể từ
chối vì hợp đồng validation đã thay đổi; khi đó bắt đầu một run v3 mới.

## Cell 9 — Train phase 1: chỉ học latent

```python
arguments = ['train-phase1', '--config', CONFIG, '--manifest', MANIFEST, '--output', OUT]
PHASE1 = OUT / 'phase1/last.pt'
if PHASE1.exists():
    arguments += ['--resume', PHASE1]
started = time.perf_counter()
run_cli(*arguments)
print('Phase 1 invocation elapsed minutes:', (time.perf_counter() - started) / 60)
```

CLI lưu checkpoint mỗi250 update. Nếu session ngắt, resume từ update đã lưu;
các update sau đó có thể phải chạy lại. Đơn vị lịch train là optimizer update,
không phải epoch. Không giảm tổng update của config khi resume để ép chuyển phase.

Lệnh tự chọn số GPU theo `runtime.gpu_count` (mặc định `auto` → tối đa hai GPU).
Thêm `'--gpus', '1'` vào `arguments` để ép một GPU khi cần cô lập lỗi, hoặc
`'--gpus', '2'` để bắt buộc hai GPU và **báo lỗi thay vì âm thầm chạy một GPU**
nếu notebook chỉ cấp một. Dòng `Execution: {...}` đầu log ghi backend, `device_ids`
và xác nhận loss tính trên batch gather toàn cục; giá trị này cũng được lưu vào
`metadata.execution` của checkpoint. Đổi số GPU **không** đổi `batch_size` toàn
cục, không đổi configuration hash và resume qua lại giữa 1 và 2 GPU đều hợp lệ.

## Cell 10 — Xem latent curves và kiểm tra gate

```python
from IPython.display import display, Image as DisplayImage
from qjepa.training.checkpoints import load_checkpoint, require_phase1_checkpoint

run_cli('plot-training', '--run-dir', OUT / 'phase1')
display(DisplayImage(filename=str(OUT / 'phase1/training_curves.png')))
display(DisplayImage(filename=str(OUT / 'phase1/latent_diagnostics.png')))
payload = load_checkpoint(PHASE1, 'cpu')  # chỉ load checkpoint của chính run này
require_phase1_checkpoint(payload)
print(json.dumps(payload['metadata'], indent=2))
print('Latent gate reasons:', payload.get('latent_metrics', {}).get('gate_reasons'))
assert payload['successful_updates'] == config['phase1']['max_successful_updates']
assert payload['metadata']['latent_gate_status'] == 'PASS', 'Chẩn đoán collapse/scale trước phase 2.'
del payload
gc.collect()
```

Đồ thị gồm total/JEPA/variance/covariance, LR, gradient norm, FD/EMA, std/rank/RMS.
Loss JEPA giảm đồng thời rank/std sụt mạnh có thể là collapse. Nếu gate WARN/FAIL,
không sửa metadata checkpoint hoặc thêm reconstruction vào phase 1 để vượt gate.
Kiểm tra diversity dataset/batch, corruption, LR và so control không FD bằng
`configs/phase1_control.yaml` trong **run khác**. Gate PASS là kiểm tra tương đối,
không phải chứng minh latent đã có chất lượng tốt.

## Cell 11 — Train phase 2: chỉ cập nhật decoder

```python
arguments = ['train-phase2', '--config', CONFIG, '--manifest', MANIFEST,
             '--backbone-checkpoint', PHASE1, '--output', OUT]
PHASE2 = OUT / 'phase2/last.pt'
if PHASE2.exists():
    arguments += ['--resume', PHASE2]
started = time.perf_counter()
run_cli(*arguments)
print('Phase 2 invocation elapsed minutes:', (time.perf_counter() - started) / 60)
run_cli('plot-training', '--run-dir', OUT / 'phase2')
display(DisplayImage(filename=str(OUT / 'phase2/training_curves.png')))
BEST = OUT / 'phase2/best_joint_validation.pt'
assert BEST.is_file()
print((OUT / 'phase2/training_summary.json').read_text())
```

Backbone/normalizer frozen, mới khởi tạo decoder, batch4×accumulation2. Checkpoint
best chọn bằng joint validation score; mặc định validation là bank32 mẫu cố
định trải giữa trajectory/time. Có thể tăng `runtime.validation_batches` **trước
khi bắt đầu run** để đánh giá nhiều hơn; giữ ổn định khi so best và resume.

Phase 2 cũng nhận `--gpus` như phase 1. Lưu ý phần chia GPU áp dụng cho từng
**microbatch**, không phải cả update: batch4×accumulation2 nghĩa là mỗi
microbatch 4 mẫu được chia 2 mẫu mỗi GPU. Hai GPU giúp phần forward/backward của
decoder, không thay đổi số update hay nghĩa của gradient accumulation.

## Cell 12 — Đánh giá toàn bộ test và xuất kết quả

Chỉ dùng test sau khi đã chọn checkpoint bằng validation. Protocol chạy10 nhóm,
chi phí khoảng10 lượt inference toàn test; không giới hạn batch trong kết quả
chính. Nếu cần thử báo cáo nhanh, dùng `--max-batches 2` ở output khác và ghi rõ
đó là subset, không gọi là kết quả toàn test.

```python
EVAL = OUT / 'test_results'
if not (EVAL / 'metrics.json').exists():
    run_cli('evaluate', '--checkpoint', BEST, '--manifest', MANIFEST,
            '--split', 'test', '--protocol', '--device', 'cuda',
            '--output', EVAL, '--panels', '6')
else:
    print('Đang đọc kết quả đã có. Muốn đánh giá run/checkpoint khác, chọn EVAL mới.')
results = json.loads((EVAL / 'metrics.json').read_text())
display(DisplayImage(filename=str(EVAL / 'comparison.png')))

columns = ['image_psnr_db', 'image_ssim', 'image_mae', 'accel_rmse', 'gyro_rmse']
for scenario, values in results.items():
    print('\nSCENARIO:', scenario,
          '| images:', values['image_count'], '| unique IMU rows:', values['imu_covered_unique_rows'])
    for key in columns:
        print(f"  {key}: input={values['baseline_' + key]:.6f} -> restored={values[key]:.6f}")
```

Scenario: clean/clean, noisy-image/clean-IMU, clean-image/noisy-IMU, noisy/noisy,
low-light-only, blur-only, sensor-noise-only, IMU-white-only, IMU-bias-only,
IMU-bandwidth-only. Full-eval corruption tắt xác suất giữ clean; các xác suất
blur/noise thành phần vẫn như config. Clean PSNR được cap120 dB khi MSE≤`1e-12`.

## Cell 13 — Xem ảnh phục hồi và trace IMU sáu trục

```python
SCENARIO = 'noisy_noisy'  # đổi thành low_light_only, blur_only, ...
for path in sorted((EVAL / SCENARIO / 'images').glob('*.png'))[:3]:
    display(DisplayImage(filename=str(path)))
for path in sorted((EVAL / SCENARIO / 'imu').glob('*.png'))[:2]:
    display(DisplayImage(filename=str(path)))

imu_index = json.loads((EVAL / SCENARIO / 'imu_index.json').read_text())
assert imu_index, 'Không có IMU covered.'
entry = imu_index[0]
with np.load(EVAL / SCENARIO / entry['file']) as data:
    covered = data['coverage']
    print('Trajectory:', str(data['trajectory']))
    print('Covered rows:', int(covered.sum()))
    print('Restored physical shape:', data['restored'][covered].shape)
    print('RMSE per axis:', np.sqrt(((data['restored'][covered] - data['clean'][covered])**2).mean(0)))
```

Panel ảnh: clean → input tối/mờ/nhiễu → restored → sai số RGB trung bình với
thang màu cố định `[0,1]`. IMU plot gồm clean/noisy/restored cùng timeline; ảnh
được chọn đều theo thứ tự sample để tái lập, không chọn riêng kết quả đẹp.
IMU windows được gộp trọng số tam giác; metrics chỉ đếm mỗi hàng covered một lần.
Plot lấy tối đa4.000 điểm/trajectory; `.npz` và metrics giữ toàn bộ dữ liệu covered.

## Cell 14 — Xuất archive để tải về và tiếp tục session sau

Archive chứa source/config, manifest, logs, checkpoint và báo cáo; không sao chép
dataset ảnh/IMU vào archive. File có thể lớn do optimizer/checkpoints và các
chuỗi IMU trong10 scenario; xem dung lượng trước khi tải.

```python
import zipfile
from IPython.display import FileLink

EXPORT = PROJECT / 'exports/qwt_jepa_v3_run.zip'
EXPORT.parent.mkdir(parents=True, exist_ok=True)
include = ['qjepa', 'configs', 'tests', 'pyproject.toml', 'README.md', 'source_revision.txt',
           'QWT_JEPA_JACOBIAN_MIGRATION_GUIDE.md', 'QWT_JEPA_V3_DETAILED_ARCHITECTURE_DIAGRAMS.md',
           'kaggle_dataset.md',
           'KAGGLE_TRAIN_CELLS.md', 'KIEN_TRUC_VA_QUY_TRINH_TRAIN.md', 'CODE_REVIEW_KAGGLE.md',
           'manifests/kaggle', 'outputs/kaggle_main']
with zipfile.ZipFile(EXPORT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for name in include:
        path = PROJECT / name
        paths = path.rglob('*') if path.is_dir() else [path]
        for file in paths:
            if file.is_file() and '__pycache__' not in file.parts and '.pytest_cache' not in file.parts:
                archive.write(file, file.relative_to(PROJECT))
print('Archive GiB:', EXPORT.stat().st_size / 2**30)
display(FileLink(str(EXPORT.relative_to(PROJECT))))
```

Ngoài download, lưu notebook version/output để giữ artifacts của session theo
giao diện Kaggle đang dùng. Khi session sắp hết giờ mà train chưa xong, có thể
dừng sau checkpoint và chạy cell14; các phase sau chưa có sẽ được bỏ qua khi zip.

## Các file cần kiểm tra sau train

```text
outputs/kaggle_main/
├── source_revision.txt
├── phase1/
│   ├── last.pt
│   ├── resolved_config.yaml
│   ├── validation_bank.json
│   ├── train.jsonl / history.csv
│   └── training_curves.png / latent_diagnostics.png / training_summary.json
├── phase2/
│   ├── last.pt / best_joint_validation.pt
│   ├── resolved_config.yaml / validation_bank.json
│   ├── train.jsonl / history.csv
│   └── training_curves.png / training_summary.json
└── test_results/
    ├── metrics.json / metrics.csv / comparison.png
    ├── evaluation_config.json
    └── <scenario>/
        ├── per_frame.csv
        ├── images/frame_*.png
        ├── imu_index.json
        └── imu/<trajectory_hash>.npz và .png
```

## Diễn giải kết quả và lỗi thường gặp

| Quan sát | Cần kiểm tra |
| --- | --- |
| PSNR/SSIM restored cao hơn input, MAE thấp hơn | Cải thiện ảnh trên scenario đó; vẫn xem chi tiết panel |
| Accel/gyro RMSE restored thấp hơn input | Cải thiện IMU; xem thêm bias từng trục và derivative error |
| Train loss giảm nhưng validation xấu | Có thể overfit hoặc lệch corruption/domain; xem best checkpoint |
| JEPA nhỏ nhưng std/rank gần 0 | Collapse; không chuyển phase và không sửa gate để bỏ qua |
| Không tìm thấy trajectory | Kiểm tra layout `imu/` và `image_lcam_front/`, dataset đã giải nén, root đủ cao |
| Thiếu valid/test hoặc quá ít trajectory | Bổ sung/chọn lại dataset; không chia ngẫu nhiên các window chồng lấn |
| Timestamp không tăng hoặc dt≈0 | Dùng giây float64, kiểm tra đồng bộ; không ép timestamp Unix sang float32 |
| Checkpoint/config hash khác | Dùng đúng source/config/data của run; thay đổi semantics cần run mới |
| OOM ở B8 phase1 | GPU chưa đủ cho cấu hình đó; accumulation không thay thế thống kê B8. Thử `--gpus 2` nếu notebook có T4 x2: batch toàn cục giữ nguyên nhưng mỗi GPU chỉ giữ một nửa activation |
| `Requested 2 GPU(s) ... only 1 visible` | Notebook chỉ cấp một GPU; đổi accelerator sang T4 x2 hoặc bỏ `--gpus 2` để chạy `auto` |
| Hai GPU nhưng `nvidia-smi` chỉ thấy một GPU có tải | Bình thường với `DataParallel`: GPU chính giữ thêm phần gather/loss/optimizer nên luôn nặng hơn |
| Muốn dùng `torchrun`/DDP | Backend hiện là `DataParallel` một process; CLI từ chối khi `WORLD_SIZE>1`. Chạy `python -m qjepa` một lần, không bọc `torchrun` |
| Không có PNG vì session dừng giữa train | Chạy `plot-training` trên log hiện có; khôi phục từ `last.pt` |

Chưa có kết quả train chính trên dataset của bạn tại thời điểm viết tài liệu.
Các chỉ số hiển thị sau khi chạy là số đo của run thực; không có bảng kết quả
chất lượng được điền sẵn. Smoke local chỉ xác nhận code và file báo cáo chạy được.
