"""Phase-specific wrappers that make forbidden data paths impossible by API."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.normalize import ImuNormalizer
from .backbone import LatentBatch, MultimodalBackbone
from .decoders import LatentDecoders
from .predictors import LatentPredictor, image_tokens, imu_tokens
from .teachers import EMATeachers


class LatentPretrainingModel(nn.Module):
    """Phase 1 owns backbone, predictors and teachers.

    It may also own an auxiliary decoder. That decoder never ships: phase 2
    builds its own from scratch. Its only job is to keep the latent decodable,
    because every other phase-1 term is self-referential and can be satisfied by
    a representation that has thrown the signal away.
    """

    def __init__(
        self,
        backbone: MultimodalBackbone | None = None,
        normalizer: ImuNormalizer | None = None,
        predictor_hidden: int = 256,
        decoders: LatentDecoders | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone or MultimodalBackbone()
        self.normalizer = normalizer or ImuNormalizer()
        embedding_dim = self.backbone.image_encoder.out_channels
        self.image_predictor = LatentPredictor(embedding_dim, predictor_hidden)
        self.imu_predictor = LatentPredictor(embedding_dim, predictor_hidden)
        self.teachers = EMATeachers(self.backbone)
        self.decoders = decoders
        self.register_buffer("decoder_forward_calls", torch.zeros((), dtype=torch.long))

    @property
    def reconstructs(self) -> bool:
        return self.decoders is not None

    def online_parameters(self):
        parameters = list(self.backbone.parameters())
        parameters += list(self.image_predictor.parameters())
        parameters += list(self.imu_predictor.parameters())
        if self.decoders is not None:
            parameters += list(self.decoders.parameters())
        return parameters

    def reconstruct(self, latent: LatentBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict clean coefficients from the noisy latent, as phase 2 will."""
        if self.decoders is None:
            raise ValueError("Phase-1 reconstruction requires decoder_enabled")
        return self.decoders(latent.ZI, latent.ZU)

    def encode_online(
        self, image: torch.Tensor, imu_phys: torch.Tensor, image_time: torch.Tensor, imu_times: torch.Tensor
    ) -> LatentBatch:
        return self.backbone.encode_online(
            image, self.normalizer.normalize(imu_phys), image_time, imu_times
        )

    def predictions(self, latent: LatentBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image_predictor(image_tokens(latent.ZI)), self.imu_predictor(imu_tokens(latent.ZU))

    @torch.no_grad()
    def targets(self, image_clean: torch.Tensor, imu_clean_phys: torch.Tensor):
        return self.teachers.encode_clean(
            self.backbone, image_clean, self.normalizer.normalize(imu_clean_phys)
        )


@dataclass
class RestoredBatch:
    image: torch.Tensor
    imu_normalized: torch.Tensor
    imu_physical: torch.Tensor
    image_coefficients: torch.Tensor
    imu_coefficients: torch.Tensor


class RestorationSystem(nn.Module):
    """Phase 2/inference: frozen backbone followed by latent-only decoders."""

    def __init__(
        self,
        backbone: MultimodalBackbone,
        normalizer: ImuNormalizer,
        decoders: LatentDecoders | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.normalizer = normalizer
        self.decoders = decoders or LatentDecoders()
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False).eval()
        self.normalizer.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        self.training = mode
        self.backbone.eval()
        self.normalizer.eval()
        self.decoders.train(mode)
        return self

    def encode(
        self, image_noisy: torch.Tensor, imu_noisy_phys: torch.Tensor, image_time: torch.Tensor, imu_times: torch.Tensor
    ) -> LatentBatch:
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone.encode_online(
                image_noisy,
                self.normalizer.normalize(imu_noisy_phys),
                image_time,
                imu_times,
            )

    def decode(self, latent: LatentBatch) -> RestoredBatch:
        image_coeff, imu_coeff = self.decoders(
            latent.ZI, latent.ZU, latent.image_coefficients, latent.imu_coefficients
        )
        image = self.backbone.image_transform.synthesis(image_coeff, latent.image_layout)
        imu_norm = self.backbone.imu_transform.synthesis(imu_coeff, latent.imu_layout)
        return RestoredBatch(
            image=image,
            imu_normalized=imu_norm,
            imu_physical=self.normalizer.denormalize(imu_norm),
            image_coefficients=image_coeff,
            imu_coefficients=imu_coeff,
        )

    def forward(
        self, image_noisy: torch.Tensor, imu_noisy_phys: torch.Tensor, image_time: torch.Tensor, imu_times: torch.Tensor
    ) -> RestoredBatch:
        return self.decode(self.encode(image_noisy, imu_noisy_phys, image_time, imu_times))
