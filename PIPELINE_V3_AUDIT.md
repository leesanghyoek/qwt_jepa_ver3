# Audit triển khai QWT–JEPA v3

Ngày audit: 15/09/2026.

Trạng thái ban đầu của `qwt_jepa_version3` chỉ có hai file đặc tả Markdown, không
có source, config, test hoặc checkpoint để migrate. Vì vậy source v3 được dựng mới
trong chính thư mục này. Không checkpoint cũ nào được gắn nhãn lại hoặc dùng làm
parent cho main pipeline.

| Hạng mục | File/hàm thực | Hiện trạng | Bằng chứng |
| --- | --- | --- | --- |
| Latent forward | `qjepa/models/backbone.py::MultimodalBackbone.encode_online` | Trả FI/FU trước fusion và ZI/ZU sau fusion; không decoder | Shape tests + smoke |
| Phase 1 API | `qjepa/models/pipeline.py::LatentPretrainingModel` | Không có decoder module | `test_phase1_has_dense_shapes_teacher_stop_gradient_and_no_decoder` |
| Phase 1 optimizer/loss | `qjepa/training/phase1.py` | Encoder, fusion, predictors; JEPA + variance/covariance + sensitivity | Optimizer ownership test |
| Teacher EMA | `qjepa/models/teachers.py` | Deepcopy random online encoders, stop-gradient, update sau optimizer | Unit/smoke path |
| Transform ảnh | `qjepa/transforms/qwt.py` | Bốn cây shifted-db4, periodic, pack 48; QWT reference chưa xác minh | Round-trip + synthesis gradient chỉ chứng minh khả nghịch |
| Haar IMU | `qjepa/transforms/haar.py` | `[B,6,128] -> [B,12,64]` | Round-trip test |
| Decoder input | `qjepa/models/decoders.py::LatentDecoders.forward` | Chỉ hai đối số ZI/ZU; absolute coefficients | Signature/permutation tests |
| Phase 2 freeze | `qjepa/models/pipeline.py::RestorationSystem` và `training/phase2.py` | Backbone/normalizer eval, requires_grad false, optimizer decoder-only | Hash before/after test |
| Low-light image data | `qjepa/corruptions/image.py` | Blur → exposure/gamma → sensor noise → codec | Determinism/darkening/group tests |
| IMU corruption | `qjepa/corruptions/imu.py` | Sinh toàn trajectory trước khi cắt window | Exact overlap test |
| Split/normalization | `qjepa/data/manifest.py` | Split trajectory trước window; unique clean train rows | Synthetic manifest test |
| Layout dataset Kaggle | `qjepa/data/tartanair.py::discover_trajectories` | TartanAir V2 `<env>/<difficulty>/<Pxxx>`, IMU .npy 100 Hz, camera 10 Hz | Bản sao đúng cấu trúc + regression hình học window (13 frame rìa/trajectory) |
| Checkpoint history | `qjepa/training/checkpoints.py` | Strict pipeline/phase/reconstruction fields và hashes | Provenance test |
| Thực thi 1–2 GPU | `qjepa/execution.py::parallel_forward` | Một process, `DataParallel` khi có hai GPU; chỉ dict tensor qua ranh giới gather | Device selection, gather scatter/gather test, hash topology |
| Loss trên batch toàn cục | `qjepa/training/phase1.py::Phase1Trainer.step` | Variance/covariance/JEPA tính sau gather đủ B mẫu | B8 loss/gradient khớp qua wrapper hai chunk |

## Gate

| Gate | Trạng thái | Phạm vi bằng chứng |
| --- | --- | --- |
| G0 — tách phase | PASS | API, optimizer ownership, checkpoint metadata, one-step smoke |
| G1 — transform/shape/data | PARTIAL | Shape/Haar/round-trip và manifest qua test; chưa có đối chiếu QWT reference |
| G2 — variance/covariance | PASS (code/toy) | Constant, position-only, B−1 covariance, gradient tests |
| G3 — encoder sensitivity | PASS (code/toy) | Phase-1 gradient smoke, actual post-clamp energy path |
| G4 — latent-only decoder | PASS (code/toy) | Signature, permutation, output shapes, frozen hash |
| G5 — resume/reproducibility | PASS (code path) | SHA-256 streams, sampler `(seed,epoch)`, khôi phục batch offset, strict checkpoint loaders |
| G6 — pilot TartanAir/GPU | NOT_RUN | Chưa có dataset và ngân sách GPU trong thư mục này |
| G7 — thực thi hai GPU | PASS (code/toy) | Chọn device, gather, hash, quyền sở hữu optimizer/checkpoint qua test CPU; train hai GPU CUDA thật vẫn NOT_RUN (test tự skip khi thiếu phần cứng) |

`PASS (code/toy)` chỉ xác nhận wiring và tính số học; không phải bằng chứng latent
đã hữu ích hoặc model đã khử nhiễu tốt trên camera thật.

Bản kiểm tra bổ sung cho Kaggle và bằng chứng mới nằm trong
[CODE_REVIEW_KAGGLE.md](CODE_REVIEW_KAGGLE.md). Hai báo cáo phase cũ ghi nhận smoke
tại thời điểm triển khai ban đầu, không phải kết quả train chính sau các sửa đổi.
