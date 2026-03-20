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
from typing import Sequence, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLSM(nn.Module):
    """
    Lightweight Multi-Level Self-Modulation block.

    Args:
        in_channels_list: channel numbers of all input levels.
        out_channels: unified output channel width.
        down_scale: pooling ratio used in the local branch.
        low_rank_lambda: weight of the L1 term in the low-rank descriptor.
        align_corners: flag for bilinear interpolation.
        use_projection: if True, each level is projected to out_channels first.
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        out_channels: int,
        down_scale: int = 8,
        low_rank_lambda: float = 1e-2,
        align_corners: bool = False,
        use_projection: bool = True,
    ) -> None:
        super().__init__()

        if len(in_channels_list) == 0:
            raise ValueError("in_channels_list must not be empty.")

        self.in_channels_list = list(in_channels_list)
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.down_scale = down_scale
        self.low_rank_lambda = low_rank_lambda
        self.align_corners = align_corners

        # Project all levels to a common channel width.
        if use_projection:
            self.level_projs = nn.ModuleList(
                [
                    nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0)
                    for c in in_channels_list
                ]
            )
        else:
            if any(c != out_channels for c in in_channels_list):
                raise ValueError(
                    "When use_projection=False, all input channels must equal out_channels."
                )
            self.level_projs = nn.ModuleList([nn.Identity() for _ in in_channels_list])

        # Local branch operator.
        self.dw_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
        )

        # Modulation mapping f_m.
        self.mask_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Aggregation mapping f_agg.
        self.agg_conv = nn.Conv2d(
            out_channels * self.num_levels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Learnable self-modulation coefficients.
        self.alpha = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        self.act = nn.GELU()

    def _align_feature(
        self,
        x: torch.Tensor,
        proj: nn.Module,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Project one feature map to the common channel width and spatial size.
        """
        x = proj(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return x

    def _local_branch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Local descriptor:
            x_s = DWConv(Pool(x))
        """
        h, w = x.shape[-2:]
        pooled_h = max(h // self.down_scale, 1)
        pooled_w = max(w // self.down_scale, 1)

        x_pool = F.adaptive_max_pool2d(x, (pooled_h, pooled_w))
        x_s = self.dw_conv(x_pool)
        return x_s

    def _low_rank_branch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft low-rank descriptor:
            r = ||x||_F + lambda * ||x||_1

        The descriptor is computed channel-wise after flattening the spatial
        dimensions. The output shape is [B, C, 1, 1].
        """
        b, c, _, _ = x.shape
        x_flat = x.view(b, c, -1)

        x_fro = torch.norm(x_flat, p=2, dim=-1, keepdim=True)
        x_l1 = torch.norm(x_flat, p=1, dim=-1, keepdim=True)

        r = x_fro + self.low_rank_lambda * x_l1
        r = r.unsqueeze(-1)
        return r

    def _modulate_one_level(self, x: torch.Tensor) -> torch.Tensor:
        """
        Self-modulate one aligned level.

        Steps:
            1. build local descriptor
            2. build low-rank descriptor
            3. fuse them with learnable coefficients
            4. generate modulation mask
            5. upsample the mask and apply it to the feature
        """
        h, w = x.shape[-2:]

        x_s = self._local_branch(x)
        r = self._low_rank_branch(x)

        temp = self.alpha * x_s + self.beta * r
        mask = self.act(self.mask_conv(temp))
        mask = F.interpolate(
            mask,
            size=(h, w),
            mode="bilinear",
            align_corners=self.align_corners,
        )

        x_hat = x * mask
        return x_hat

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: multi-level feature maps, each with shape [B, C_l, H_l, W_l]
            target_size: target spatial size (H, W). If None, use features[0] size.

        Returns:
            Aggregated feature map with shape [B, out_channels, H, W].
        """
        if len(features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature levels, but got {len(features)}."
            )

        if target_size is None:
            target_size = features[0].shape[-2:]

        aligned_features: List[torch.Tensor] = []
        for feat, proj in zip(features, self.level_projs):
            aligned = self._align_feature(feat, proj, target_size)
            aligned_features.append(aligned)

        modulated_features = [self._modulate_one_level(x) for x in aligned_features]
        x_cat = torch.cat(modulated_features, dim=1)
        x_mlsm = self.agg_conv(x_cat)

        return x_mlsm


class ConvBNReLU(nn.Module):
    """
    A small helper block used in the plug-in example.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLSMSkipPlugIn(nn.Module):
    """
    A simple wrapper that shows how MLSM can be inserted into a skip path.

    It first aggregates multi-level encoder features with MLSM, then concatenates
    the aggregated context with the decoder feature, and finally refines the fused
    feature with a standard convolution block.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        out_channels: int,
        down_scale: int = 8,
        low_rank_lambda: float = 1e-2,
    ) -> None:
        super().__init__()

        self.mlsm = MLSM(
            in_channels_list=encoder_channels,
            out_channels=out_channels,
            down_scale=down_scale,
            low_rank_lambda=low_rank_lambda,
        )

        self.fuse = ConvBNReLU(out_channels + decoder_channels, out_channels)

    def forward(
        self,
        encoder_features: Sequence[torch.Tensor],
        decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        target_size = decoder_feature.shape[-2:]
        skip_context = self.mlsm(encoder_features, target_size=target_size)
        fused = torch.cat([decoder_feature, skip_context], dim=1)
        return self.fuse(fused)


def _test_mlsm() -> None:
    """
    Minimal executable test.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example multi-level encoder features
    features = [
        torch.randn(2, 32, 192, 192, device=device),
        torch.randn(2, 64, 96, 96, device=device),
        torch.randn(2, 128, 48, 48, device=device),
        torch.randn(2, 256, 24, 24, device=device),
    ]

    mlsm = MLSM(
        in_channels_list=[32, 64, 128, 256],
        out_channels=32,
        down_scale=8,
        low_rank_lambda=1e-2,
    ).to(device)

    out = mlsm(features, target_size=(192, 192))
    print("==== MLSM block test ====")
    for i, feat in enumerate(features):
        print(f"input level {i}: {tuple(feat.shape)}")
    print(f"mlsm output:    {tuple(out.shape)}")

    decoder_feature = torch.randn(2, 32, 192, 192, device=device)
    skip_block = MLSMSkipPlugIn(
        encoder_channels=[32, 64, 128, 256],
        decoder_channels=32,
        out_channels=32,
    ).to(device)

    fused = skip_block(features, decoder_feature)
    print(f"decoder feature: {tuple(decoder_feature.shape)}")
    print(f"fused output:    {tuple(fused.shape)}")


if __name__ == "__main__":
    _test_mlsm()