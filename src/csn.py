"""ir-CSN-R50 (interaction-reduced channel-separated network), vendored.

WHY VENDORED AND NOT A DEPENDENCY. The weights are mmaction2's, and mmaction2
pulls mmcv + mmengine + a model registry to build one backbone. pytorchvideo's
`create_csn` is the other published builder, but it is 0.1.5/2022 with a
documented torchvision-0.17+ import break (issue #251). m99 set the precedent on
exactly this problem: copy-and-trim the structure, take no dependency.

THE KEY NAMES ARE THE CONTRACT. Every module here is named so that the
mmaction2 checkpoint loads with `strict=True` and ZERO key remapping --
`conv1.conv` / `conv1.bn`, `layerN.i.conv{1,3}.conv`, `layerN.i.conv2.0.conv`
(the depthwise conv lives inside an nn.Sequential upstream, so the `.0.` is
load-bearing), `layerN.0.downsample.conv`. A remap is a silent-failure surface;
a strict load is a proof. If this file and the checkpoint ever disagree, the
load raises, which is the whole point of writing it this way.

EVERY CONSTANT BELOW WAS READ, NOT ASSUMED. Source: the K400-ft checkpoint's own
embedded `meta['cfg']` (ResNet3dCSN, depth=50, bottleneck_mode='ir',
with_pool2=False, norm_eval=True, bn_frozen=True) plus mmaction2's
`resnet3d_csn.py` / `resnet3d.py` docstrings and constructors, read 2026-08-19:

  * conv1     3x7x7, stride (conv1_stride_t=1, 2, 2), padding (1,3,3), no bias
  * maxpool   (1,3,3), stride (pool1_stride_t=1, 2, 2), padding (0,1,1)
  * pool2     DISABLED -- the config says with_pool2=False
  * stages    blocks (3,4,6,3); spatial_strides (1,2,2,2); temporal_strides
              (1,2,2,2)  => temporal downsample x8 overall
  * style     'pytorch' => conv1 carries stride 1 and CONV2 CARRIES THE STRIDE
  * conv2     3x3x3 DEPTHWISE (groups=planes) -- this is what 'ir' means; the
              'ip' mode's extra 1x1x1 is absent, which the shapes confirm
  * conv3     1x1x1, NO activation before the residual add
  * BN eps 1e-3, NOT torch's 1e-5 default. Silent, and it moves every
              activation in the network.

Derived from mmaction2 (https://github.com/open-mmlab/mmaction2), Apache-2.0.
Copyright (c) OpenMMLab. Structure reimplemented; no mmaction2 code is imported.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# mmaction2's ResNet3dCSN default, and NOT nn.BatchNorm3d's 1e-5. Read off
# resnet3d_csn.py's documented norm_cfg.
BN_EPS = 1e-3


class ConvBN(nn.Module):
    """mmcv's ConvModule, trimmed to what CSN uses.

    The submodule names `conv` and `bn` ARE the checkpoint's key names. ReLU
    carries no parameters, so naming it costs no keys.
    """

    def __init__(self, cin: int, cout: int, kernel, stride=1, padding=0,
                 groups: int = 1, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, kernel, stride=stride, padding=padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(cout, eps=BN_EPS)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return x if self.act is None else self.act(x)


class CSNBottleneck(nn.Module):
    """ir-bottleneck: 1x1x1 -> 3x3x3 DEPTHWISE -> 1x1x1, stride on conv2."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, spatial_stride: int = 1,
                 temporal_stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = ConvBN(inplanes, planes, 1)
        # nn.Sequential is not decoration: upstream builds conv2 as a list so
        # that 'ip' mode can prepend a 1x1x1. We are 'ir' and the list has one
        # entry -- but the key is `conv2.0.*` either way, so the wrapper stays.
        self.conv2 = nn.Sequential(
            ConvBN(planes, planes, 3,
                   stride=(temporal_stride, spatial_stride, spatial_stride),
                   padding=1, groups=planes)          # groups=planes => depthwise
        )
        self.conv3 = ConvBN(planes, planes * self.expansion, 1, act=False)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.relu(out + identity)


class IPCSNBottleneck(CSNBottleneck):
    """ip-bottleneck (QUEUE §113, D48 (2)): 1x1x1 -> 1x1x1 (BN, NO activation) -> 3x3x3 DEPTHWISE -> 1x1x1.

    READ, NOT ASSUMED, from two sources on 2026-09-08: (a) mmaction2 v0.24.1 `resnet3d_csn.py`
    (the version that produced the converted VMZ checkpoint) prepends, in 'ip' mode,
    `ConvModule(planes, planes, 1, stride=1, bias=False, norm_cfg=..., act_cfg=None)` to the
    depthwise ConvModule -- so the extra 1x1x1 carries a BatchNorm and NO ReLU; (b) the fetched
    file's own keys: `layerN.i.conv2.0.{conv,bn}.*` (the 1x1x1) and `layerN.i.conv2.1.{conv,bn}.*`
    (the depthwise 3x3x3), the stride living on conv2.1. The ir-bottleneck's `conv2.0.*` becomes
    `conv2.1.*` here, which is exactly why the nn.Sequential wrapper above was kept load-bearing.
    """

    def __init__(self, inplanes: int, planes: int, spatial_stride: int = 1,
                 temporal_stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__(inplanes, planes, spatial_stride, temporal_stride, downsample)
        self.conv2 = nn.Sequential(
            ConvBN(planes, planes, 1, act=False),   # the 'ip' 1x1x1: BN, act_cfg=None upstream
            ConvBN(planes, planes, 3,
                   stride=(temporal_stride, spatial_stride, spatial_stride),
                   padding=1, groups=planes)          # groups=planes => depthwise
        )


class IRCSNResNet50(nn.Module):
    """The backbone alone. Returns the (B, 2048, T', H', W') feature map."""

    STAGE_BLOCKS = (3, 4, 6, 3)
    SPATIAL_STRIDES = (1, 2, 2, 2)
    TEMPORAL_STRIDES = (1, 2, 2, 2)
    BLOCK = CSNBottleneck   # §113: the ip variant swaps the block class and nothing else

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.conv1 = ConvBN(in_channels, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.maxpool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        inplanes = 64
        for i, (n_blocks, ss, ts) in enumerate(
                zip(self.STAGE_BLOCKS, self.SPATIAL_STRIDES, self.TEMPORAL_STRIDES)):
            planes = 64 * 2 ** i
            out = planes * self.BLOCK.expansion
            # Present on EVERY stage's first block -- layer1 too, where the
            # stride is 1 but 64 -> 256 still needs the projection.
            downsample = ConvBN(inplanes, out, 1, stride=(ts, ss, ss), act=False)
            blocks = [self.BLOCK(inplanes, planes, ss, ts, downsample)]
            inplanes = out
            blocks += [self.BLOCK(inplanes, planes) for _ in range(n_blocks - 1)]
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        # pool2 is deliberately absent: the config says with_pool2=False.
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class IRCSNResNet152(IRCSNResNet50):
    """ir-CSN-152 (QUEUE §49, D19 a): the same stem, bottleneck and key layout as the R50 above with the
    (3, 8, 36, 3) stage depths of mmaction2's ResNet3dCSN depth=152, so the IG-65M→K400 bnfrozen checkpoint
    loads strict=True through the same strip_prefix path. 29.4 M backbone params (~118 MB fp32, ~59 MB fp16)."""

    STAGE_BLOCKS = (3, 8, 36, 3)


class IPCSNResNet152(IRCSNResNet152):
    """ip-CSN-152 (QUEUE §113, D48 (2) reading α): the ir-CSN-152 above with IPCSNBottleneck in every stage — the same
    stem, pooling, stage depths (3, 8, 36, 3) and key layout, so mmaction2's converted VMZ checkpoint
    `vmz_ipcsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-c3be9793.pth` (IG-65M → K400, BN frozen; NOTICE)
    loads strict=True through the same strip_prefix path. 32,196,992 backbone params (the file's own count, BN buffers
    excluded) against the ir-CSN-152's 28,883,968 — ratio 1.1147 (≈ 64.4 MB fp16 for the backbone alone)."""

    BLOCK = IPCSNBottleneck


def strip_prefix(state: dict, prefix: str = "backbone.") -> dict:
    """The K400-ft file namespaces the backbone; the pure IG-65M file does not."""
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
