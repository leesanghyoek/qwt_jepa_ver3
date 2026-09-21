# Phase 2 fine-tune có bảo vệ full/full và IMU

Thí nghiệm trộn blur từ đầu đã giảm lỗi blur nhưng làm full/full và IMU gần về baseline. `best_blur_validation.pt` của run đó được chọn ở update 250, quá sớm cho tác vụ chung. Run này bắt đầu từ decoder **cũ** tại `p1_detail2_trial/phase2/best_joint_validation.pt`, giữ nguyên phase 1/backbone, dùng 75% full/full và 20% blur_only, learning rate thấp hơn. Nó chạy trong output riêng.

`best_guarded_validation.pt` chỉ được lưu khi trên **cùng bank cố định** ảnh full/full giảm PSNR không quá 0.2 dB, SSIM không quá 0.01, accel/gyro RMSE tăng không quá 3%, đồng thời MAE và lỗi cạnh blur đều tốt hơn decoder cũ. Đây là điều kiện chọn ứng viên, không thay cho audit 256 ảnh và full validation cuối.

Chạy trong Kaggle sau khi đã có phase 1 checkpoint, checkpoint phase 2 cũ và manifest. Nếu phiên đã mất, khôi phục archive run `p1_detail2_trial`; không chạy lại phase 1.

```python
from pathlib import Path
import json, os, subprocess, sys, yaml

OLD_PROJECT = Path('/kaggle/working/qwt_jepa_version3')
GUARD_SOURCE = Path('/kaggle/working/qwt_jepa_guarded')
GUARD_OUT = OLD_PROJECT / 'outputs/p2_guarded_finetune'
P1 = OLD_PROJECT / 'outputs/p1_detail2_trial/phase1/last.pt'
OLD_BEST = OLD_PROJECT / 'outputs/p1_detail2_trial/phase2/best_joint_validation.pt'
MANIFEST = OLD_PROJECT / 'manifests/kaggle'
SOURCE_COMMIT = 'ad5aa11d69b75b17866e66e401d88fa2ca60c087'
for path in (P1, OLD_BEST, MANIFEST / 'meta.json'):
    assert path.is_file(), f'Cần khôi phục file cũ: {path}'
if not GUARD_SOURCE.exists():
    subprocess.run(['git', 'clone', 'https://github.com/leesanghyoek/qwt_jepa_ver3.git', str(GUARD_SOURCE)], check=True)
subprocess.run(['git', '-C', str(GUARD_SOURCE), 'fetch', '--no-tags', 'origin', 'main'], check=True)
subprocess.run(['git', '-C', str(GUARD_SOURCE), 'checkout', '--detach', SOURCE_COMMIT], check=True)

saved = OLD_PROJECT / 'outputs/p1_detail2_trial/phase1/resolved_config.yaml'
assert saved.is_file(), f'Cần config phase 1 cũ: {saved}'
config = yaml.safe_load(saved.read_text())
recipe = yaml.safe_load((GUARD_SOURCE / 'configs/kaggle_phase2_guarded_finetune.yaml').read_text())
config['phase2'].update(recipe['phase2'])
config['runtime'].update(recipe['runtime'])
config['runtime']['output_dir'] = str(GUARD_OUT)
config['data']['manifest_dir'] = str(MANIFEST)
GUARD_OUT.mkdir(parents=True, exist_ok=True)
RUN_CONFIG = GUARD_OUT / 'config.yaml'
if RUN_CONFIG.exists():
    assert yaml.safe_load(RUN_CONFIG.read_text()) == config, 'Run cũ dùng config khác; chọn output mới.'
else:
    RUN_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False))
guard_env = os.environ.copy()
guard_env['PYTHONPATH'] = str(GUARD_SOURCE) + os.pathsep + guard_env.get('PYTHONPATH', '')
```

```python
LAST = GUARD_OUT / 'phase2/last.pt'
command = [sys.executable, '-u', '-m', 'qjepa', 'train-phase2',
           '--config', str(RUN_CONFIG), '--manifest', str(MANIFEST),
           '--output', str(GUARD_OUT), '--backbone-checkpoint', str(P1)]
if LAST.is_file():
    command += ['--resume', str(LAST)]
else:
    command += ['--decoder-init-checkpoint', str(OLD_BEST)]
subprocess.run(command, cwd=GUARD_SOURCE, env=guard_env, check=True)
```

Nếu không có `best_guarded_validation.pt`, không checkpoint nào đồng thời giữ được full/full, IMU và cải thiện blur trên bank này. Giữ model cũ và gửi `phase2/train.jsonl` cùng `guard_reference.json`; không dùng `best_blur_validation.pt` hoặc `last.pt` chỉ vì chúng tồn tại.

Nếu có `best_guarded_validation.pt`, kiểm tra trên 256 ảnh valid rải đều và full/full toàn bộ validation. Chỉ thay checkpoint cũ nếu trên ảnh blur thật MAE, lỗi cạnh và LH/HL đều thấp hơn **input** và full/full cùng IMU không giảm đáng kể so với mốc cũ (20.59 dB, SSIM 0.677, accel 0.807, gyro 0.077):

```python
candidate = GUARD_OUT / 'phase2/best_guarded_validation.pt'
if candidate.is_file():
    audit_json = GUARD_OUT / 'diagnostics/guarded_blur_256.json'
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-u', str(GUARD_SOURCE / 'tools/image_blur_audit.py'),
                    '--checkpoint', str(candidate), '--manifest', str(MANIFEST),
                    '--samples', '256', '--device', 'cuda', '--output', str(audit_json)],
                   cwd=GUARD_SOURCE, env=guard_env, check=True)
    full_eval = GUARD_OUT / 'eval_guarded_full_valid'
    if not (full_eval / 'metrics.json').is_file():
        subprocess.run([sys.executable, '-u', '-m', 'qjepa', 'evaluate',
                        '--checkpoint', str(candidate), '--manifest', str(MANIFEST),
                        '--split', 'valid', '--image-mode', 'full', '--imu-mode', 'full',
                        '--device', 'cuda', '--output', str(full_eval), '--panels', '2'],
                       cwd=GUARD_SOURCE, env=guard_env, check=True)
    print(json.dumps(json.loads((full_eval / 'metrics.json').read_text())['requested'], indent=2))
```

Lưu `GUARD_OUT` thành Kaggle Output hoặc archive trước khi đóng phiên.
