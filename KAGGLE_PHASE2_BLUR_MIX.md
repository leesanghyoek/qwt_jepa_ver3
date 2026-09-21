# Thử nghiệm phase 2 để sửa trường hợp blur

Run hiện tại vẫn là mốc đối chứng: full/full cải thiện nhưng 232/256 ảnh có blur thật bị tệ hơn ở PSNR, MAE, lỗi cạnh và cả ba băng LH/HL/HH. Không xóa hoặc ghi đè `p1_detail2_trial`.

Chạy hai cell dưới đây trong Kaggle sau khi đã mount dataset và có `phase1/last.pt` cùng manifest của run cũ. Nếu phiên Kaggle mới không còn checkpoint, khôi phục archive/checkpoint cũ trước; không cần train lại phase 1. Cell này tải source mới vào thư mục riêng, không đổi source đang dùng của notebook cũ.

```python
from pathlib import Path
import os, subprocess, sys, yaml

OLD = Path('/kaggle/working/qwt_jepa_version3')
SOURCE = Path('/kaggle/working/qwt_jepa_blur_mix')
SOURCE_COMMIT = '11c2c574789addf1c8126d49a0cce2400fa573c3'
P1 = OLD / 'outputs/p1_detail2_trial/phase1/last.pt'
MANIFEST = OLD / 'manifests/kaggle'
OUT = OLD / 'outputs/p2_blur_mix_trial'
assert P1.is_file(), f'Khôi phục checkpoint phase 1 cũ: {P1}'
assert (MANIFEST / 'meta.json').is_file(), f'Khôi phục manifest cũ: {MANIFEST}'
if not SOURCE.exists():
    subprocess.run(['git', 'clone', 'https://github.com/leesanghyoek/qwt_jepa_ver3.git', str(SOURCE)], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'fetch', '--no-tags', 'origin', 'main'], check=True)
subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)
CONFIG = SOURCE / 'configs/kaggle_phase2_blur_mix.yaml'
assert CONFIG.is_file(), f'Source chưa có thử nghiệm mới: {CONFIG}'
saved_config = OLD / 'outputs/p1_detail2_trial/phase1/resolved_config.yaml'
assert saved_config.is_file(), f'Cần config gốc của phase 1: {saved_config}'
config = yaml.safe_load(saved_config.read_text())
experiment = yaml.safe_load(CONFIG.read_text())
config['phase2'].update(experiment['phase2'])
config['runtime']['output_dir'] = str(OUT)
config['data']['manifest_dir'] = str(MANIFEST)
OUT.mkdir(parents=True, exist_ok=True)
RUN_CONFIG = OUT / 'blur_mix_config.yaml'
if RUN_CONFIG.exists():
    assert yaml.safe_load(RUN_CONFIG.read_text()) == config, 'Config run cũ khác; chọn OUT mới.'
else:
    RUN_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
print('Source:', subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip())
```

```python
env = os.environ.copy()
env['PYTHONPATH'] = str(SOURCE) + os.pathsep + env.get('PYTHONPATH', '')
last = OUT / 'phase2/last.pt'
command = [sys.executable, '-u', '-m', 'qjepa', 'train-phase2',
           '--config', str(RUN_CONFIG), '--manifest', str(MANIFEST),
           '--output', str(OUT), '--backbone-checkpoint', str(P1)]
if last.is_file():
    command += ['--resume', str(last)]
print('Running:', ' '.join(command))
subprocess.run(command, cwd=SOURCE, env=env, check=True)
```

Mỗi lần validation sẽ in `blur actual`, MAE và `edge` trên cùng một bank cố định, chỉ lấy frame thực sự có blur. `worst ratio < 1` nghĩa là cả MAE lẫn lỗi cạnh đều tốt hơn input trên bank đó. Mô hình lưu riêng `best_joint_validation.pt` và `best_blur_validation.pt`; chúng có thể là hai checkpoint khác nhau. Đừng kết luận từ bank 128 ảnh dùng để chọn checkpoint; dùng 256 ảnh rải đều làm kiểm tra sau train:

```python
import json

for name in ('best_joint_validation', 'best_blur_validation'):
    checkpoint = OUT / 'phase2' / f'{name}.pt'
    if not checkpoint.is_file():
        continue
    report = OUT / 'diagnostics' / f'{name}_blur_spread_256.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-u', str(SOURCE / 'tools/image_blur_audit.py'),
                    '--checkpoint', str(checkpoint), '--manifest', str(MANIFEST),
                    '--samples', '256', '--device', 'cuda', '--output', str(report)],
                   cwd=SOURCE, env=env, check=True)
    active = json.loads(report.read_text())['scenarios']['blur_only']['blur_active_only']
    print(name, 'blur thật:', active['samples'],
          '| MAE:', active['image_mae_input'], '→', active['image_mae_restored'],
          '| lỗi cạnh:', active['strong_edge_gradient_mae_input'], '→',
          active['strong_edge_gradient_mae_restored'])
```

Nếu có checkpoint vượt input trên ảnh blur, đánh giá full/full toàn bộ validation để so sánh trực tiếp với run cũ (PSNR 20.59 dB, SSIM 0.677, accel RMSE 0.807, gyro RMSE 0.077):

```python
CANDIDATE = OUT / 'phase2/best_blur_validation.pt'
EVAL = OUT / 'eval_best_blur_full_valid'
assert CANDIDATE.is_file()
if not (EVAL / 'metrics.json').is_file():
    subprocess.run([sys.executable, '-u', '-m', 'qjepa', 'evaluate',
                    '--checkpoint', str(CANDIDATE), '--manifest', str(MANIFEST),
                    '--split', 'valid', '--image-mode', 'full', '--imu-mode', 'full',
                    '--device', 'cuda', '--output', str(EVAL), '--panels', '2'],
                   cwd=SOURCE, env=env, check=True)
print(json.dumps(json.loads((EVAL / 'metrics.json').read_text())['requested'], indent=2))
```

Chỉ nhận mô hình mới nếu trên 256 ảnh blur thật cả MAE, lỗi cạnh, LH/HL đều giảm so với input **và** full/full cùng IMU không tụt đáng kể so với checkpoint cũ. Nếu không đạt, giữ checkpoint cũ và gửi lại hai JSON audit cùng `eval_best_blur_full_valid/metrics.json`; chưa nên thay kiến trúc hay train lại phase 1 chỉ từ thử nghiệm này.

Trước khi kết thúc phiên Kaggle, lưu run mới để không mất checkpoint:

```python
import zipfile
from IPython.display import FileLink, display

archive = Path('/kaggle/working/p2_blur_mix_trial.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            bundle.write(path, path.relative_to(OUT.parent))
print('Đã lưu:', archive)
display(FileLink(str(archive)))
```
