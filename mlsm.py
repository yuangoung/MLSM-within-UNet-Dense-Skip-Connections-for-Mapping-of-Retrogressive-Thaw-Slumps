# YY editor
# 2025/11/20

"""
Lightweight Multi-Level Self-Modulation (MLSM).
This file provides:
1. MLSM: a plug-and-play multi-level aggregation block.
2. A minimal runnable test in the main function.

The MLSM workflow:
- multi-level feature alignment
- local branch: adaptive max pooling + depth-wise 3x3 convolution
- low-rank branch: soft low-rank descriptor from Frobenius and L1 norms
- self-modulation mask generation
- multi-level concatenation and 1x1 aggregation

Input to MLSM:
    a list/tuple of feature maps from multiple encoder levels

Output from MLSM:
    one aggregated feature map at the target resolution
"""

from __future__ import annotations
from typing import Sequence, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSM(nn.Module):

    def __init__(
        self,
        in_channels_list: Sequence[int],
        out_channels: int,
        down_scale: int = 8,
        low_rank_lambda: float = 1e-2,
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        if len(in_channels_list) < 1:
            raise ValueError("in_channels_list must contain at least one feature map.")

        self.in_channels_list = list(in_channels_list)
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.down_scale = down_scale
        self.low_rank_lambda = low_rank_lambda
        self.align_corners = align_corners

        self.level_projs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            for c in in_channels_list
        ])

        self.dw_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            groups=out_channels, bias=True
        )
        self.mask_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.agg_conv = nn.Conv2d(
            out_channels * self.num_levels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.alpha = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.act = nn.GELU()

    def _align(self, x: torch.Tensor, proj: nn.Module, target_size: Tuple[int, int]) -> torch.Tensor:
        x = proj(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x, size=target_size, mode="bilinear", align_corners=self.align_corners
            )
        return x

    def _local_branch(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled_h = max(h // self.down_scale, 1)
        pooled_w = max(w // self.down_scale, 1)
        x_pool = F.adaptive_max_pool2d(x, (pooled_h, pooled_w))
        return self.dw_conv(x_pool)

    def _low_rank_branch(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        x_flat = x.view(b, c, -1)
        x_fro = torch.norm(x_flat, p=2, dim=-1, keepdim=True)
        x_l1 = torch.norm(x_flat, p=1, dim=-1, keepdim=True)
        r = x_fro + self.low_rank_lambda * x_l1
        return r.unsqueeze(-1)

    def _modulate_one(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x_s = self._local_branch(x)
        r = self._low_rank_branch(x)

        temp = self.alpha * x_s + self.beta * r
        mask = self.act(self.mask_conv(temp))
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=self.align_corners)
        return x * mask

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if len(features) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature maps, but got {len(features)}.")

        if target_size is None:
            target_size = features[0].shape[-2:]

        aligned = [self._align(feat, proj, target_size) for feat, proj in zip(features, self.level_projs)]
        modulated = [self._modulate_one(x) for x in aligned]
        x_cat = torch.cat(modulated, dim=1)
        return self.agg_conv(x_cat)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = [
        torch.randn(2, 32, 128, 128, device=device),
        torch.randn(2, 64, 64, 64, device=device),
        torch.randn(2, 128, 32, 32, device=device),
    ]
    module = MLSM([32, 64, 128], out_channels=32).to(device)
    out = module(feats, target_size=(128, 128))
    print("Output shape:", tuple(out.shape))