"""Fresh phase-2 decoders driven by ZI and ZU, optionally correcting the input."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Stage, Upsample, initialize_trainable, resize


class SkipMerge(nn.Module):
    """Noi feature encoder vao duong decoder.

    Skip mang duong net cua anh NHIEU: sac net nhung khong dang tin, vi trong do
    co ca canh that lan hat nhieu. Latent moi la thu biet canh nao la that — no
    duoc huan luyen de doan latent cua anh SACH, cong voi neo ep no giu he so sach
    o bang chi tiet. Nen phan cong dung la: skip cap do phan giai, latent quyet
    dinh giu cai gi.

    `gated=False` KHONG lam duoc viec do. Mot conv pointwise tren tensor noi chi
    hoc duoc mot TI LE PHA TRON co dinh theo kenh — sau khi train, kenh nao lay
    bao nhieu skip la co dinh o moi vi tri, moi anh. No khong the nhin latent de
    quyet dinh "cho nay canh that, cho qua; cho kia la hat nhieu, chan lai".

    `gated=True` sinh cong TU DUONG LATENT, nen cong phu thuoc tung vi tri va
    tung kenh. Dung mau va quy uoc cua SharedGatedFusion: bias -2.0 cho
    sigmoid(-2) ~ 0.12, tuc luc bat dau cong gan nhu dong va mo dan theo muc skip
    to ra huu ich.

    Ca hai che do deu bat dau o identity tren duong decoder, nen san identity cua
    residual (head zero-init) con nguyen o update 0.

    Danh doi phai noi ro: net di qua duong nay den tu ANH DAU VAO, khong phai tu
    latent. Do la ly do `encoder_skips` phai bat tuong minh, va la ly do
    delta_report.py co --ablate-latent.
    """

    def __init__(
        self, channels: int, skip_channels: int, *, dim: int,
        gated: bool = True, gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.dim = dim
        self.gated = gated
        if not gated:
            self.project = conv(channels + skip_channels, channels, 1)
            nn.init.zeros_(self.project.bias)
            with torch.no_grad():
                self.project.weight.zero_()
                for index in range(channels):
                    self.project.weight[index, index] = 1.0
            return
        self.gate = conv(channels, channels, 1)
        self.skip_project = conv(skip_channels, channels, 1)
        # Cong khoi tao tu bias thuan tuy: trong so 0 nen luc dau cong khong phu
        # thuoc noi dung, va skip_project zero-init nen dong gop ban dau dung bang 0.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        nn.init.zeros_(self.skip_project.weight)
        nn.init.zeros_(self.skip_project.bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.shape[2:] != x.shape[2:]:
            skip = resize(skip, tuple(x.shape[2:]), dim=self.dim)
        if not self.gated:
            return self.project(torch.cat((x, skip), dim=1))
        return x + torch.sigmoid(self.gate(x)) * self.skip_project(skip)


class LatentCoefficientDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int,
        output_size: tuple[int, ...],
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        *,
        dim: int,
        groups: int = 8,
        residual: bool = False,
        skip_channels: tuple[int, ...] | None = None,
        skip_gating: bool = True,
        sees_input: bool = True,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.dim = dim
        self.residual = residual
        self.sees_input = bool(residual and sees_input)
        self.uses_skips = skip_channels is not None
        self.output_size = output_size
        sizes = []
        for divisor in (4, 2, 1):
            sizes.append(tuple(max(1, (value + divisor - 1) // divisor) for value in output_size))
        self.sizes = tuple(sizes)
        self.shuffle2 = Upsample(c3, dim=dim)
        self.up2 = Stage(c3, c2, dim=dim, groups=groups)
        self.shuffle1 = Upsample(c2, dim=dim)
        self.up1 = Stage(c2, c1, dim=dim, groups=groups)
        self.shuffle0 = Upsample(c1, dim=dim)
        self.up0 = Stage(c1, c0, dim=dim, groups=groups)
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        # Voi sees_input, head doc CA duong latent lan he so dau vao. Khong co no,
        # delta = f(Z) va decoder khong the bieu dien mot phep khu nhieu: muon tru
        # bot phan nhieu thi phai DOC duoc no, ma latent duoc huan luyen de doan
        # latent cua tin hieu SACH, tuc duoc day de vut bo hien thuc cua nhieu.
        # Voi he so wavelet, co bien phep co gian — delta = -alpha * C_in tren nua
        # bang detail — nam trong tam mot conv duy nhat.
        head_in = c0 + output_channels if self.sees_input else c0
        self.head = conv(head_in, output_channels, 3, padding=1)
        initialize_trainable(self)
        # Sau initialize_trainable, vi SkipMerge tu dat trong so identity+zero cua
        # no va khong duoc Kaiming ghi de.
        if skip_channels is not None:
            if len(skip_channels) != 3:
                raise ValueError(f"Expected three skip levels, got {len(skip_channels)}")
            self.merge2 = SkipMerge(c2, skip_channels[0], dim=dim, gated=skip_gating)
            self.merge1 = SkipMerge(c1, skip_channels[1], dim=dim, gated=skip_gating)
            self.merge0 = SkipMerge(c0, skip_channels[2], dim=dim, gated=skip_gating)
        if residual:
            # Bat dau o dung identity: update 0 tra lai chinh he so dau vao, nen
            # model khong the te hon input va moi buoc chi co the di len.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        latent: torch.Tensor,
        base: torch.Tensor | None = None,
        skips: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if self.uses_skips and skips is None:
            raise ValueError("Decoder was built with skips; pass the encoder stages")
        if skips is not None and not self.uses_skips:
            raise ValueError("Decoder was built without skips; refusing to use them")
        x = self.up2(self.shuffle2(latent))
        if skips is not None:
            x = self.merge2(x, skips[0])
        x = self.up1(self.shuffle1(x))
        if skips is not None:
            x = self.merge1(x, skips[1])
        x = self.up0(self.shuffle0(x))
        if skips is not None:
            x = self.merge0(x, skips[2])
        if tuple(x.shape[2:]) != tuple(self.output_size):
            # Luoi khong chia het cho 8; chi con lai phan le sau ba lan nhan doi.
            x = resize(x, self.output_size, dim=self.dim)
        if not self.residual:
            return self.head(x)
        if base is None:
            raise ValueError("Residual decoding needs the input coefficients as base")
        predicted = self.head(torch.cat((x, base), dim=1) if self.sees_input else x)
        return base + predicted


class LatentDecoders(nn.Module):
    """Decode ZI/ZU into wavelet coefficients, optionally with input paths."""

    def __init__(
        self,
        image_coefficient_size: tuple[int, int] = (128, 128),
        imu_coefficient_length: int = 64,
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        groups: int = 8,
        residual: bool = False,
        skip_channels: tuple[int, ...] | None = None,
        skip_gating: bool = True,
        sees_input: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.uses_skips = skip_channels is not None
        self.image = LatentCoefficientDecoder(
            48, image_coefficient_size, channels, dim=2, groups=groups,
            residual=residual, skip_channels=skip_channels, skip_gating=skip_gating,
            sees_input=sees_input,
        )
        self.imu = LatentCoefficientDecoder(
            12, (imu_coefficient_length,), channels, dim=1, groups=groups,
            residual=residual, skip_channels=skip_channels, skip_gating=skip_gating,
            sees_input=sees_input,
        )

    def forward(
        self,
        ZI: torch.Tensor,
        ZU: torch.Tensor,
        image_base: torch.Tensor | None = None,
        imu_base: torch.Tensor | None = None,
        image_skips: tuple[torch.Tensor, ...] | None = None,
        imu_skips: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.image(ZI, image_base, image_skips),
            self.imu(ZU, imu_base, imu_skips),
        )

