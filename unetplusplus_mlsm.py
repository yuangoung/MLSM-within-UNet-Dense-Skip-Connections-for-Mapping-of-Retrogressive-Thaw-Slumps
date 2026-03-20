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
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlsm import MLSM


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlusMLSM(nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 4,
        deep_supervision: bool = True,
        nb_filter: Optional[List[int]] = None,
        down_scale: int = 8,
        low_rank_lambda: float = 1e-2,
        align_corners: bool = False,
        backbone: Optional[str] = None,
        output_stride: Optional[int] = None,
        sync_bn: bool = False,
        freeze_bn: bool = False,
    ) -> None:
        super().__init__()

        if nb_filter is None:
            nb_filter = [32, 64, 128, 256, 512]

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.deep_supervision = deep_supervision
        self.nb_filter = nb_filter
        self.align_corners = align_corners

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] * 2 + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 3 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 4 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 5 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.mlsm0_1 = MLSM([nb_filter[0], nb_filter[1]], nb_filter[0], down_scale, low_rank_lambda, align_corners)
        self.mlsm1_1 = MLSM([nb_filter[1], nb_filter[2]], nb_filter[1], down_scale, low_rank_lambda, align_corners)
        self.mlsm2_1 = MLSM([nb_filter[2], nb_filter[3]], nb_filter[2], down_scale, low_rank_lambda, align_corners)
        self.mlsm3_1 = MLSM([nb_filter[3], nb_filter[4]], nb_filter[3], down_scale, low_rank_lambda, align_corners)

        self.mlsm0_2 = MLSM([nb_filter[0], nb_filter[0], nb_filter[1]], nb_filter[0], down_scale, low_rank_lambda, align_corners)
        self.mlsm1_2 = MLSM([nb_filter[1], nb_filter[1], nb_filter[2]], nb_filter[1], down_scale, low_rank_lambda, align_corners)
        self.mlsm2_2 = MLSM([nb_filter[2], nb_filter[2], nb_filter[3]], nb_filter[2], down_scale, low_rank_lambda, align_corners)

        self.mlsm0_3 = MLSM([nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[1]], nb_filter[0], down_scale, low_rank_lambda, align_corners)
        self.mlsm1_3 = MLSM([nb_filter[1], nb_filter[1], nb_filter[1], nb_filter[2]], nb_filter[1], down_scale, low_rank_lambda, align_corners)

        self.mlsm0_4 = MLSM([nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[1]], nb_filter[0], down_scale, low_rank_lambda, align_corners)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.ds_weights = nn.Parameter(torch.ones(4))
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        if freeze_bn:
            self.freeze_bn()

    def _up(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=self.align_corners)

    def _node_forward(self, features: List[torch.Tensor], mlsm_block: MLSM, conv_block: nn.Module) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        mlsm_context = mlsm_block(features, target_size=target_size)
        x = torch.cat(features + [mlsm_context], dim=1)
        return conv_block(x)

    def forward(self, x: torch.Tensor, return_deep_supervision: bool = False):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self._node_forward([x0_0, self._up(x1_0, x0_0)], self.mlsm0_1, self.conv0_1)
        x1_1 = self._node_forward([x1_0, self._up(x2_0, x1_0)], self.mlsm1_1, self.conv1_1)
        x2_1 = self._node_forward([x2_0, self._up(x3_0, x2_0)], self.mlsm2_1, self.conv2_1)
        x3_1 = self._node_forward([x3_0, self._up(x4_0, x3_0)], self.mlsm3_1, self.conv3_1)

        x0_2 = self._node_forward([x0_0, x0_1, self._up(x1_1, x0_0)], self.mlsm0_2, self.conv0_2)
        x1_2 = self._node_forward([x1_0, x1_1, self._up(x2_1, x1_0)], self.mlsm1_2, self.conv1_2)
        x2_2 = self._node_forward([x2_0, x2_1, self._up(x3_1, x2_0)], self.mlsm2_2, self.conv2_2)

        x0_3 = self._node_forward([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], self.mlsm0_3, self.conv0_3)
        x1_3 = self._node_forward([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], self.mlsm1_3, self.conv1_3)

        x0_4 = self._node_forward([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], self.mlsm0_4, self.conv0_4)

        if self.deep_supervision:
            y1 = self.final1(x0_1)
            y2 = self.final2(x0_2)
            y3 = self.final3(x0_3)
            y4 = self.final4(x0_4)

            weights = torch.softmax(self.ds_weights, dim=0)
            y = weights[0] * y1 + weights[1] * y2 + weights[2] * y3 + weights[3] * y4

            if return_deep_supervision:
                return y, [y1, y2, y3, y4], weights
            return y

        y = self.final(x0_4)
        if return_deep_supervision:
            return y, [y], torch.ones(1, device=y.device, dtype=y.dtype)
        return y

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.conv0_0, self.conv1_0, self.conv2_0, self.conv3_0, self.conv4_0]
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    yield p

    def get_10x_lr_params(self):
        modules = [
            self.conv0_1, self.conv1_1, self.conv2_1, self.conv3_1,
            self.conv0_2, self.conv1_2, self.conv2_2,
            self.conv0_3, self.conv1_3,
            self.conv0_4,
            self.mlsm0_1, self.mlsm1_1, self.mlsm2_1, self.mlsm3_1,
            self.mlsm0_2, self.mlsm1_2, self.mlsm2_2,
            self.mlsm0_3, self.mlsm1_3, self.mlsm0_4,
        ]
        if self.deep_supervision:
            modules.extend([self.final1, self.final2, self.final3, self.final4])
        else:
            modules.append(self.final)

        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    yield p


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetPlusPlusMLSM(num_classes=2, input_channels=3, deep_supervision=True).to(device)
    x = torch.randn(1, 3, 384, 384, device=device)
    y, ds_outputs, weights = model(x, return_deep_supervision=True)

    print("Input shape :", tuple(x.shape))
    print("Final shape :", tuple(y.shape))
    for i, item in enumerate(ds_outputs, 1):
        print(f"DS-{i} shape:", tuple(item.shape))