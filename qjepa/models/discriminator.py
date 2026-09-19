"""Patch discriminator cho phase 2 — doi tuong duy nhat khong toi uu trung vi.

L1, SmoothL1 va ca so hang bang chi tiet deu la sai so trung binh, nen nghiem toi
uu cua chung la trung vi co dieu kien. Voi bai toan bat dinh nhu khu mo, trung vi
do CHINH LA anh mo: o mot vi tri co the co canh, xoa tan so cao giam sai so chac
chan, con doan them canh thi rui ro. Khong trong so nao thoat khoi dieu do — do
dung la thu delta_report.py do duoc khi bang chi tiet TE HON sau khi sua.

Discriminator khong hoi "trung binh co gan dung khong" ma hoi "mieng nay co giong
mot mieng anh that khong". Mot mieng bi lam phang tra loi SAI cau hoi do, ke ca
khi no gan trung vi. Day la ly do doi tuong doi khang lam duoc viec ma moi trong
so L1 khong lam duoc.

Cham diem theo TUNG O thay vi ca anh: net la tinh chat cuc bo, va mot ban do
logit cho nhieu tin hieu gradient hon mot so duy nhat.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: tuple[int, ...] = (64, 128, 256),
        groups: int = 8,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("Discriminator needs at least one stage")
        layers: list[nn.Module] = []
        previous = in_channels
        for index, width in enumerate(channels):
            if index and width % groups:
                raise ValueError(f"{width} channels are not divisible by {groups} groups")
            layers.append(nn.Conv2d(previous, width, 4, stride=2, padding=1))
            # Stage dau khong chuan hoa: no phai nhin duoc do sang tuyet doi, thu
            # ma GroupNorm se xoa mat.
            if index:
                layers.append(nn.GroupNorm(groups, width))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            previous = width
        layers.append(nn.Conv2d(previous, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge: phat khi that cham duoi +1 hoac gia vuot tren -1.

    Hinge dung thay BCE vi no ngung day mot khi mieden da phan loai dung voi bien
    du rong, nen discriminator kho thang tuyet doi va gradient cho decoder khong
    bi triet tieu — che hong thuong gap nhat cua GAN.
    """
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def adversarial_weight(
    successful_updates: int,
    *,
    start_after: int,
    ramp_updates: int,
    maximum: float,
) -> float:
    """0 cho toi start_after, roi len tuyen tinh — giong lich cua encoder sensitivity.

    Bat doi khang tu update 0 se day decoder di sinh ket cau truoc khi no kip hoc
    dung mau va bo cuc, va phan do rat kho go ra sau.
    """
    if maximum <= 0 or successful_updates < start_after:
        return 0.0
    return maximum * min((successful_updates - start_after + 1) / max(1, ramp_updates), 1.0)
