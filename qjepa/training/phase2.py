"""Phase 2 trainer: frozen backbone and fresh latent-only decoders."""

from __future__ import annotations

from typing import Any

import torch

from ..models.pipeline import RestorationSystem
from ..execution import RestorationForward, execution_metadata, parallel_forward
from .checkpoints import configuration_hash, rng_state, state_dict_hash
from .losses import phase2_reconstruction_loss
from .phase1 import _finite_gradients, _to_device
from .schedules import warmup_cosine_lr


class Phase2Trainer:
    def __init__(
        self,
        system: RestorationSystem,
        config: dict[str, Any],
        device: torch.device,
        parent_checkpoint: str,
        manifest_hash: str = "unknown",
    ) -> None:
        self.system = system.to(device)
        self.system.freeze_backbone()
        self.config = config
        self.phase = config["phase2"]
        self.device = device
        self.forward_model, self.device_ids = parallel_forward(
            RestorationForward(self.system), device, config["runtime"].get("gpu_count", "auto")
        )
        self.parent_checkpoint = parent_checkpoint
        self.manifest_hash = manifest_hash
        self.successful_updates = 0
        self.parameters = list(self.system.decoders.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=self.phase["learning_rate"],
            weight_decay=self.phase["weight_decay"],
        )
        self.frozen_backbone_hash = state_dict_hash(self.system.backbone)
        self.frozen_normalizer_hash = state_dict_hash(self.system.normalizer)
        self.decoder_initialization_hash = state_dict_hash(self.system.decoders)

    def _set_lr(self) -> float:
        lr = warmup_cosine_lr(
            self.successful_updates,
            self.phase["max_successful_updates"],
            self.phase["warmup_updates"],
            self.phase["learning_rate"],
            self.phase["minimum_lr"],
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def assert_backbone_frozen(self) -> None:
        if state_dict_hash(self.system.backbone) != self.frozen_backbone_hash:
            raise RuntimeError("Frozen backbone changed during phase 2")
        if state_dict_hash(self.system.normalizer) != self.frozen_normalizer_hash:
            raise RuntimeError("Frozen IMU normalizer changed during phase 2")

    def step(self, raw_batches: dict[str, Any] | list[dict[str, Any]]) -> dict[str, float | bool | str]:
        microbatches = raw_batches if isinstance(raw_batches, list) else [raw_batches]
        expected = int(self.phase["gradient_accumulation"])
        if len(microbatches) != expected:
            raise ValueError(f"Phase 2 expects {expected} microbatches per update, got {len(microbatches)}")
        self.system.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        lr = self._set_lr()
        totals: dict[str, float] = {"loss": 0.0, "image_l1": 0.0, "imu_accel_smooth_l1": 0.0, "imu_gyro_smooth_l1": 0.0}
        for raw_batch in microbatches:
            batch = _to_device(raw_batch, self.device)
            restored = self.forward_model(
                batch["image_noisy"], batch["imu_noisy_phys"], batch["image_time"], batch["imu_times"]
            )
            clean_imu = self.system.normalizer.normalize(batch["imu_clean_phys"])
            loss, parts = phase2_reconstruction_loss(
                restored["image"],
                batch["image_clean"],
                restored["imu_normalized"],
                clean_imu,
                beta=self.phase["smooth_l1_beta"],
            )
            loss = self.phase["reconstruction_loss_weight"] * loss
            (loss / expected).backward()
            totals["loss"] += float(loss.detach()) / expected
            for key, value in parts.items():
                totals[key] += float(value.detach()) / expected
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.phase["gradient_clip_norm"])
        if not torch.isfinite(gradient_norm) or not _finite_gradients(self.parameters):
            self.optimizer.zero_grad(set_to_none=True)
            return {"skipped": True, "reason": "non_finite_gradient", **totals}
        self.optimizer.step()
        self.successful_updates += 1
        return {
            "skipped": False,
            **totals,
            "gradient_norm": float(gradient_norm),
            "learning_rate": lr,
            "successful_updates": self.successful_updates,
        }

    def checkpoint_payload(self, config: dict[str, Any]) -> dict[str, Any]:
        self.assert_backbone_frozen()
        return {
            "metadata": {
                "pipeline_version": 3,
                "phase": "latent_decoder_train",
                "decoder_input": "fused_dense_latent_only",
                "output_coefficients": "absolute_prediction",
                "successful_updates": self.successful_updates,
                "data_microbatches_consumed": self.successful_updates * self.phase["gradient_accumulation"],
                "parent_phase1_checkpoint": self.parent_checkpoint,
                "frozen_backbone_hash": self.frozen_backbone_hash,
                "frozen_normalizer_hash": self.frozen_normalizer_hash,
                "decoder_initialization_hash": self.decoder_initialization_hash,
                "decoder_current_hash": state_dict_hash(self.system.decoders),
                "configuration_hash": configuration_hash(config, "phase2"),
                "manifest_hash": self.manifest_hash,
                "execution": execution_metadata(self.device, self.device_ids),
            },
            "system": self.system.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "successful_updates": self.successful_updates,
            "config": config,
            "rng": rng_state(),
        }
