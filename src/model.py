"""ResNet-18 + TSM (Temporal Shift Module), one branch per modality.

TSM buys temporal modelling for ZERO extra parameters and ZERO extra FLOPs: it
moves 1/4 of the channels along the time axis before each residual branch, so a
2D backbone sees neighbouring frames. That matters here because the artefact
budget is 100 MB and a 3D backbone would eat it.

TRAPS THIS FILE EXISTS TO HANDLE.

1. A SILENT NO-OP TSM IS THE WORST OUTCOME. If the shift is misplaced, or the
   (B*T) reshape does not match what shift() assumes about segment layout, the
   network still trains, still converges, and is simply TSN -- and the "TSM
   on/off" A/B then measures nothing while looking like a clean null result.
   The decisive test is temporal asymmetry: reversing frame order MUST change
   the output. A pure TSN averages per-frame logits and is exactly invariant to
   frame order. tests/check_model.py asserts both directions.

2. PLACEMENT IS RESIDUAL, NOT IN-PLACE. Wrapping conv1 of each BasicBlock puts
   the shift inside the residual branch, leaving the identity path carrying
   unshifted activations. Shifting in-place instead measures -2.6 points below
   the plain TSN baseline in the paper's own ablation -- a change that looks
   like a bad idea rather than a bug.

3. shift_div=8 MEANS A QUARTER OF THE CHANNELS. 1/8 shift forward plus 1/8
   backward. Reading it as "1/8 of channels total" halves the temporal capacity.

4. THE 4-CHANNEL STEM MUST NOT CHANGE ACTIVATION SCALE. Depth(3) + IR(1) needs a
   4-channel conv1. Copying the RGB kernel and filling the 4th with the mean of
   the three raises the response by 4/3, which shifts every downstream BN's
   input distribution away from what the pretrained statistics expect. Scaling
   the whole kernel by 0.75 restores it.

5. PARTIAL BN. With ~2,900 training clips, unfreezing every BN overfits its
   statistics. TSN's remedy -- freeze stats AND affine everywhere except the
   first BN -- is what partial_bn does, and it must survive .train() being
   called again, which is why it is re-applied there rather than once at build.

6. `pretrained` DEFAULTS TO FALSE, AND THE DEFAULT IS THE SAFETY PROPERTY.
   It defaulted True until 2026-08-15 (§J-DQ defect b). Every caller that must
   NOT reach the network -- predict.py, check_swa.py -- already passed False
   explicitly, and every caller that must fetch weights -- train.py,
   check_model.py -- already passed True explicitly, so flipping the default
   changed the behaviour of exactly one caller (tools/cache_logits.py, which
   overwrites the weights with a state dict one line later and was therefore
   downloading 45 MB for nothing). The default only ever governed the callers
   nobody had thought about, and on the committee's cold machine that is a
   download inside the inference path. Defaults should fail closed.
   tests/test_inference_offline.sh runs the real entry point in a network
   namespace with no route and an empty TORCH_HOME, so this cannot regress.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import re
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18, resnet50

NUM_CLASSES = 40


class TemporalShift(nn.Module):
    """Shift 1/shift_div of channels forward and 1/shift_div back, then apply net.

    Input is (B*T, C, H, W) with time varying fastest -- i.e. the flatten of a
    (B, T, C, H, W) tensor. Getting that layout wrong silently mixes different
    clips' frames together, which is trap 1.
    """

    def __init__(self, net: nn.Module, n_segment: int, shift_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.shift(x, self.n_segment, self.shift_div))

    @staticmethod
    def shift(x: torch.Tensor, n_segment: int, shift_div: int) -> torch.Tensor:
        nt, c, h, w = x.size()
        if nt % n_segment != 0:
            raise RuntimeError(f"batch {nt} is not a multiple of n_segment {n_segment}")
        b = nt // n_segment
        x = x.view(b, n_segment, c, h, w)
        fold = c // shift_div  # trap 3: fold forward + fold back = 2c/8 = c/4
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # borrow from the future
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # from the past
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # the rest stays put
        return out.view(nt, c, h, w)


# THE PREDICATE IS _BatchNorm, NEVER BatchNorm2d, AND THAT IS THE WHOLE POINT.
# `isinstance(m, nn.BatchNorm2d)` is correct for ResNet-18 and matches NOTHING on
# s3d, which has 77 BatchNorm3d and zero BatchNorm2d. It does not raise; it
# quietly iterates over an empty set. partial_bn would then freeze no layer at
# all, and ~3k clips of depth-colormap would overwrite the Kinetics statistics
# the pretraining exists to provide -- a run that trains, converges, and reports
# a number that is simply wrong. nn.modules.batchnorm._BatchNorm is the base of
# all three, which is why train.py:recompute_bn_stats was already right.
# PRETRAINING-GAP.md §4 lists all six sites; this is the one they all funnel to.
BN_TYPES = nn.modules.batchnorm._BatchNorm


def bn_modules(net: nn.Module) -> list[nn.Module]:
    """Every BatchNorm in `net`, 1d/2d/3d alike, in module order."""
    return [m for m in net.modules() if isinstance(m, BN_TYPES)]


def _freeze_bn_except_first(net: nn.Module) -> None:
    """TSN's partial-BN: freeze statistics AND affine everywhere but the first.

    Shared by both backbones so the policy cannot drift between them, and so a
    single predicate governs a 2D and a 3D network alike."""
    for i, m in enumerate(bn_modules(net)):
        if i == 0:
            continue  # the stem's BN stays trainable
        m.eval()
        if m.weight is not None:
            m.weight.requires_grad_(False)
        if m.bias is not None:
            m.bias.requires_grad_(False)


def _adapt_stem(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Grow conv1 from 3 to in_channels, preserving activation scale (trap 4)."""
    if in_channels == conv.in_channels:
        return conv
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        w = conv.weight.data  # (64, 3, 7, 7)
        if in_channels < 3:
            raise ValueError(f"cannot adapt a 3-channel stem down to {in_channels}")
        new.weight.data[:, :3] = w
        for c in range(3, in_channels):
            new.weight.data[:, c] = w.mean(dim=1)
        # 3 real channels spread over in_channels: keep the summed response equal.
        new.weight.data.mul_(3.0 / in_channels)
        if conv.bias is not None:
            new.bias.data = conv.bias.data.clone()
    return new


class TSMResNet18(nn.Module):
    """One branch. Consumes (B, T, C, H, W), returns (B, NUM_CLASSES) logits.

    Consensus is the mean of per-frame LOGITS (TSN's convention, and F4's
    within-model rule). Averaging probabilities instead is a different model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_segment: int = 8,
        num_classes: int = NUM_CLASSES,
        shift_div: int = 8,
        tsm: bool = True,
        dropout: float = 0.5,
        pretrained: bool = False,
        partial_bn: bool = True,
    ) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.tsm_enabled = tsm
        self.partial_bn = partial_bn

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        net.conv1 = _adapt_stem(net.conv1, in_channels)

        if tsm:
            # Trap 2: wrapping conv1 of each BasicBlock places the shift INSIDE
            # the residual branch. The identity path keeps unshifted features.
            for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
                for block in layer:
                    block.conv1 = TemporalShift(block.conv1, n_segment, shift_div)

        net.fc = nn.Identity()
        self.backbone = net
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.001)
        nn.init.zeros_(self.fc.bias)

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        if mode and self.partial_bn:
            self._freeze_bn_except_first()
        return self

    def _freeze_bn_except_first(self) -> None:
        """Trap 5. Re-applied on every .train() -- a one-shot freeze at build
        time is silently undone the first time a trainer calls model.train()."""
        _freeze_bn_except_first(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t != self.n_segment:
            raise ValueError(f"model built for n_segment={self.n_segment}, got T={t}")
        # Time varies fastest -- the layout TemporalShift.shift assumes.
        # channels_last belongs HERE, on the rank-4 tensor: the backbone is 2D,
        # so channels_last_3d (rank 5 only) is the wrong format and raises.
        flat = x.reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        feats = self.backbone(flat)
        logits = self.fc(self.dropout(feats)).view(b, t, -1)
        return logits.mean(dim=1)  # consensus over T

TSM_SSV2_R50 = "tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb_20230317-ec6696ad.pth"   # §61 (D22 b)


def _mmaction_tsm_to_torchvision(sd: dict) -> dict:
    """mmaction2 ResNetTSM keys -> torchvision resnet50 keys with TemporalShift on every block's conv1.

    conv1.conv/bn -> conv1/bn1 · layerL.B.conv1.conv.net -> layerL.B.conv1.net (the shift wrapper's inner
    conv) · layerL.B.convK.conv -> convK, .convK.bn -> bnK · downsample.conv -> downsample.0, .bn -> downsample.1.
    The 174-way SSv2 head is dropped. Any other key RAISES -- a remap that swallows keys is the silent
    failure the strict load below exists to prevent.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("cls_head."):
            continue
        if not k.startswith("backbone."):
            raise KeyError(f"unexpected checkpoint key {k}")
        k = k[len("backbone."):]
        if k.startswith("conv1.conv."):
            nk = "conv1." + k[len("conv1.conv."):]
        elif k.startswith("conv1.bn."):
            nk = "bn1." + k[len("conv1.bn."):]
        else:
            m = re.fullmatch(r"(layer\d\.\d+)\.(conv[123]|downsample)\.(conv|bn)\.(.+)", k)
            if m is None:
                raise KeyError(f"unmapped checkpoint key backbone.{k}")
            blk, part, kind, rest = m.groups()
            if part == "downsample":
                nk = f"{blk}.downsample.{0 if kind == 'conv' else 1}.{rest}"
            elif kind == "conv":
                nk = f"{blk}.{part}.{rest}"
            else:
                nk = f"{blk}.bn{part[-1]}.{rest}"
        out[nk] = v
    return out


class TSMResNet50(nn.Module):
    """§61 (D22 b): TSM on torchvision's ResNet-50, loaded from mmaction2's Something-Something V2
    TSM-R50 checkpoint (ImageNet-1k init, 16 segments, shift_div 8; the 174-way head is dropped).

    Same external contract as TSMResNet18 -- (B,T,C,H,W) in, (B,NUM_CLASSES) out, consensus = mean of
    per-frame logits. NORMALISATION IS READ OFF THE CHECKPOINT'S META CFG: mean [123.675, 116.28, 103.53]
    / std [58.395, 57.12, 57.375] at 0-255 = ImageNet after /255, this repo's `thermal`/`depthir` keys
    (the same as ir-CSN). The shift is parameter-free, so the 16-segment training transfers to any
    n_segment. A 2D backbone: rank-4 inside, channels_last applies (NOT a RANK5 arch).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_segment: int = 16,
        num_classes: int = NUM_CLASSES,
        shift_div: int = 8,
        tsm: bool = True,
        dropout: float = 0.5,
        pretrained: bool = False,
        partial_bn: bool = True,
        weights: str | None = None,
    ) -> None:
        super().__init__()
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.tsm_enabled = tsm
        self.partial_bn = partial_bn
        net = resnet50(weights=None)
        net.fc = nn.Identity()
        if tsm:
            for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
                for block in layer:
                    block.conv1 = TemporalShift(block.conv1, n_segment, shift_div)
        if pretrained:
            path = CSN_WEIGHT_DIR / (weights or TSM_SSV2_R50)
            if not path.is_file():
                raise FileNotFoundError(f"TSM SSv2 weights not found at {path}; set CSN_WEIGHT_DIR to override.")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sd = _mmaction_tsm_to_torchvision(ck["state_dict"])
            if not tsm:   # TSN: no wrapper, so the inner-conv key collapses
                sd = {k.replace(".conv1.net.", ".conv1."): v for k, v in sd.items()}
            net.load_state_dict(sd, strict=True)
        net.conv1 = _adapt_stem(net.conv1, in_channels)   # after loading: acts on the pretrained kernels
        self.backbone = net
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.001)
        nn.init.zeros_(self.fc.bias)

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        if mode and self.partial_bn:
            _freeze_bn_except_first(self.backbone)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t != self.n_segment:
            raise ValueError(f"model built for n_segment={self.n_segment}, got T={t}")
        flat = x.reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        feats = self.backbone(flat)
        logits = self.fc(self.dropout(feats)).view(b, t, -1)
        return logits.mean(dim=1)


# §96 (d) (D29): the shipped depth+IR IR-plane statistics, dataset.NORM["depthir"] channel 3 (the mean of the three ImageNet RGB
# values — baseline.yaml's convention). Duplicated here rather than imported: model.py must not import dataset.py (the two import
# contexts, see IRCSNBranch.__init__); tests/check_p9_input_arms.py asserts the two agree.
IRNORM_IR_MEAN, IRNORM_IR_STD = 0.449, 0.226


def _widen_stem3d_zero(conv: nn.Conv3d, extra: int) -> nn.Conv3d:
    """§63: add `extra` input channels to a Conv3d stem with ZERO kernels -- the pretrained channels are
    copied unchanged, so the widened network's step-0 output equals the original's on the original channels."""
    new = nn.Conv3d(conv.in_channels + extra, conv.out_channels, kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight.data[:, : conv.in_channels] = conv.weight.data
        if conv.bias is not None:
            new.bias.data = conv.bias.data.clone()
    return new


def _adapt_stem3d(conv: nn.Conv3d, in_channels: int) -> nn.Conv3d:
    """_adapt_stem's formula on a Conv3d. Trap 4, one dimension up.

    THE FORMULA TRANSFERS, THE FUNCTION DOES NOT. _adapt_stem constructs an
    nn.Conv2d unconditionally, so calling it on s3d's stem returns a Conv2d
    holding a rank-5 weight and the forward pass raises
    `Expected 3D or 4D input to conv2d, but got [1,4,16,112,112]`. The x0.75
    rescale is identical and is verified response-preserving on a
    channel-uniform input to ~1e-05 (float32 round-off; PRETRAINING-GAP.md §9).
    """
    if in_channels == conv.in_channels:
        return conv
    if in_channels < 3:
        raise ValueError(f"cannot adapt a 3-channel stem down to {in_channels}")
    new = nn.Conv3d(in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        w = conv.weight.data                      # (64, 3, t, k, k)
        new.weight.data[:, :3] = w
        for c in range(3, in_channels):
            new.weight.data[:, c] = w.mean(dim=1)
        new.weight.data.mul_(3.0 / in_channels)   # keep the summed response equal
        if conv.bias is not None:
            new.bias.data = conv.bias.data.clone()
    return new


class S3DBranch(nn.Module):
    """Separable 3D CNN, Kinetics-400 pretrained. One branch per modality.

    Same external contract as TSMResNet18 -- consumes (B, T, C, H, W), returns
    (B, NUM_CLASSES) -- so predict.py, train.py and the cached-logit tooling need
    no branch-specific code. The permute to (B, C, T, H, W) happens inside.

    WHY THIS EXISTS: 7.95 M params against ResNet-18's 11.20 M, Kinetics-400
    pretraining at 68.4% top-1 (clip-level, clips_per_video=1, clip_len=128,
    crop 224 -- quote the protocol or not at all), and it runs on the existing
    120x160 cache with no re-preprocessing. It is SMALLER than the model we ship
    today. PRETRAINING-GAP.md §3.

    THREE THINGS THAT DIFFER FROM THE 2D BRANCH AND EACH BITE SILENTLY.

    1. 77 BatchNorm3d AND ZERO BatchNorm2d. Every `isinstance(m,
       nn.BatchNorm2d)` in this repository matches NOTHING on this model and
       raises nothing while doing it. partial_bn would freeze no layer at all and
       3k clips of depth-colormap would overwrite the Kinetics statistics the
       pretraining exists to provide. Fixed at all six sites by predicating on
       nn.modules.batchnorm._BatchNorm, which catches 1d/2d/3d alike.

    2. avgpool is AvgPool3d((2,7,7)), hard-wired to 224x224 pretraining, and
       raises below it: `input image (T:2 H:3 W:3) smaller than kernel size`.
       Replaced with AdaptiveAvgPool3d(1), which runs at every shape tested
       (120x160, 112x112, 96x192, 224x224, T=8 and T=16).

    3. TSM IS NOT A KNOB HERE. A 3D convolution IS the temporal operator; there
       is nothing for a channel shift to add. `tsm` is accepted so that build()
       stays uniform, and deliberately ignored -- tsm_enabled is False and
       check_model.py asserts it, so the fact is pinned by a test rather than by
       this comment. The A/B is between ARCHITECTURES, which the checkpoint's
       `arch` field names; it is not TSM on/off, and the row cannot be misread
       because `arch` is on it.

    THE WEIGHTS DECLARE min_temporal_size = 14. T=16 is this backbone's stated
    operating point, not a knee -- which is why A/B #6 folds into J1 rather than
    being measured separately on a backbone that may not survive the week.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_segment: int = 16,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        pretrained: bool = False,
        partial_bn: bool = True,
        tsm: bool = True,  # accepted, ignored — see 3 above
    ) -> None:
        super().__init__()
        from torchvision.models.video import S3D_Weights, s3d

        self.n_segment = n_segment
        self.in_channels = in_channels
        self.tsm_enabled = False
        self.partial_bn = partial_bn

        net = s3d(weights=S3D_Weights.KINETICS400_V1 if pretrained else None)
        net.features[0][0][0] = _adapt_stem3d(net.features[0][0][0], in_channels)
        net.avgpool = nn.AdaptiveAvgPool3d(1)                      # see 2 above
        net.classifier[0] = nn.Dropout(p=dropout)                  # recipe's 0.5, not s3d's 0.2
        head = nn.Conv3d(1024, num_classes, kernel_size=1)
        nn.init.normal_(head.weight, 0, 0.001)
        nn.init.zeros_(head.bias)
        net.classifier[1] = head
        self.backbone = net

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        if mode and self.partial_bn:
            _freeze_bn_except_first(self.backbone)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t != self.n_segment:
            raise ValueError(f"model built for n_segment={self.n_segment}, got T={t}")
        # (B,T,C,H,W) -> (B,C,T,H,W). s3d's own forward already means over the
        # trailing three dims, so the head returns (B, num_classes) directly.
        return self.backbone(x.permute(0, 2, 1, 3, 4))


class MC3Branch(nn.Module):
    """Mixed 2D/3D ResNet-18, Kinetics-400 pretrained. The DIVERSE member.

    Same external contract as the other two -- (B,T,C,H,W) in, (B,NUM_CLASSES)
    out -- so no caller needs branch-specific code.

    WHY THIS EXISTS AND WHY IT IS NOT "ANOTHER s3d". m55 is explicit that
    more-of-same buys nothing in the ensemble, and after J1-J7 our stack is TWO
    BRANCHES OF ONE ARCHITECTURE. Diversity is the one ensemble axis this project
    has never spent. M-01 already cleared this model on the two things that kill
    candidates here: it PASSES D1's 100 MB fp32 fence at 11.70 M params
    (46.8 MB fp32 / 23.39 MB fp16, measured not computed), and it is torchvision
    + Kinetics-400 under M-04's already-verified BSD-3, so it adds no dependency
    and no licence surface.

    ITS NORMALISATION IS BYTE-IDENTICAL TO s3d's, AND THAT IS VERIFIED RATHER
    THAN ASSUMED. MC3_18_Weights.KINETICS400_V1.transforms() returns
    mean [0.43216, 0.394666, 0.37645] / std [0.22803, 0.22145, 0.216989] -- the
    same values S3D_Weights declares. So it reuses the *_kinetics NORM keys
    legitimately. The registry comment below warns that a wrong `norm` does not
    raise and quietly costs accuracy; that is exactly why this was read off the
    weights object instead of inherited from the neighbouring class.

    BUT ITS NATIVE SCALE IS NOT s3d's, AND THAT DECIDES ITS OPERATING POINT.
    MC3_18's transforms declare resize 128x171 / crop 112x112, against s3d's
    256x256 / 224x224. m69's mechanism for why resolution paid at all was that
    "the K400 weights were trained near 224, so moving toward their native scale
    helps them specifically". Applied honestly, that argument says this model
    wants ~112x112 -- i.e. our NATIVE 120x160 cache, with no upscale at all --
    and NOT the 168x224 the s3d members ship at. Running it at 168x224 would be
    inheriting an operating point rather than deriving one.

    Unlike s3d it has 20 BatchNorm3d, not 77, and torchvision already gives it
    an AdaptiveAvgPool3d(1), so neither of S3DBranch's shape fixes is needed here.
    Verified by introspection, not assumed from the family name.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_segment: int = 16,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        pretrained: bool = False,
        partial_bn: bool = True,
        tsm: bool = True,  # accepted, ignored — a 3D conv IS the temporal operator
    ) -> None:
        super().__init__()
        from torchvision.models.video import MC3_18_Weights, mc3_18

        self.n_segment = n_segment
        self.in_channels = in_channels
        self.tsm_enabled = False
        self.partial_bn = partial_bn

        net = mc3_18(weights=MC3_18_Weights.KINETICS400_V1 if pretrained else None)
        net.stem[0] = _adapt_stem3d(net.stem[0], in_channels)
        head = nn.Linear(net.fc.in_features, num_classes)
        nn.init.normal_(head.weight, 0, 0.001)
        nn.init.zeros_(head.bias)
        net.fc = nn.Sequential(nn.Dropout(p=dropout), head)   # the recipe's 0.5
        self.backbone = net

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        if mode and self.partial_bn:
            _freeze_bn_except_first(self.backbone)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t != self.n_segment:
            raise ValueError(f"model built for n_segment={self.n_segment}, got T={t}")
        return self.backbone(x.permute(0, 2, 1, 3, 4))


# Where B0 put the vendored-CSN weights. An env var overrides it so a fresh
# clone can point elsewhere; NOTHING here downloads. pretrained=False is the
# inference path and never touches the filesystem (D12: offline-reproducible).
CSN_WEIGHT_DIR = Path(os.environ.get("CSN_WEIGHT_DIR", Path.home() / "csn"))
CSN_K400_FT = "ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth"
CSN_IG65M_PURE = "ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth"
# §82 (D26): the R152 from-scratch IG-65M file — the K400 stage dropped; a probe-gated pretraining-axis ablation START,
# never a shipped byte unless promoted through R4 (NOTICE entry at download).
CSN_IG65M_PURE_R152 = "ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth"
# §49 (D19 a): ir-CSN-152, IG-65M → K400 bnfrozen, mmaction2 (Apache-2.0); sha256 a02af4a3a1258bb8… (NOTICE).
CSN_K400_FT_R152 = "ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-7d1dacde.pth"
# §113 (D48 (2), reading α): ip-CSN-152, IG-65M → K400 bnfrozen — mmaction2's conversion of the VMZ release (Apache-2.0 over
# Apache-2.0; metafile.yml row at its lines 124-145; NOTICE, fetched 2026-09-08); sha256 c3be979346a7fb45…, 133,006,791 B.
# A bare `backbone.`-prefixed state dict with NO meta cfg — the normalisation was read off the zoo's configs, not the file.
CSN_K400_FT_IP152 = "vmz_ipcsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-c3be9793.pth"
# §68 (D23): VideoMAE V2 ViT-S/16, K710-distilled from V2-g, K400 fine-tuned — the mmaction2 re-host (NOTICE; m225).
VMAE2_VITS_K400 = "vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth"
# §74 (D24): the ViT-B sibling (86.23 M backbone params — ~172 MB fp16; a D6-gated, int8-only pricing arm; NOTICE).
VMAE2_VITB_K400 = "vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"
VMAE2_VARIANTS = {"s": (384, 6, VMAE2_VITS_K400), "b": (768, 12, VMAE2_VITB_K400)}   # embed_dims, num_heads, weights
VMAE2_DEPTH = 12    # both zoo checkpoints carry 12 blocks; §90 (D27) builds ViT-B[11] / ViT-B[10] from the same file


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR'21) on a (B, C, T, H, W) feature map — TRAIN MODE ONLY (QUEUE §35 / T8).

    With probability p per batch, each instance's per-channel mean/std over (T, H, W) is mixed with a
    random other instance's under lambda ~ Beta(alpha, alpha); the content is re-normalised with the
    mixed statistics. Eval is untouched; the shipped path never constructs this module.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.shape[0] < 2 or float(torch.rand(1)) > self.p:
            return x
        B = x.shape[0]
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        sig = (x.var(dim=(2, 3, 4), keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - mu) / sig
        lam = self.beta.sample((B, 1, 1, 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(B, device=x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


class CSNBranch(nn.Module):
    """ir-CSN-R50, IG-65M pretrained. The CORPUS-AXIS member. QUEUE.md §4.

    Same external contract as the others -- (B,T,C,H,W) in, (B,NUM_CLASSES) out.
    The network itself is src/csn.py; this class is the adaptation layer.

    WHY THIS EXISTS. m69 decomposed our gap and found architecture ~ 0 and the
    PRETRAINING CORPUS worth +38 clips. Every IG-65M artefact the record knew of
    was the 63.75 M-param r2plus1d-34 teacher, which D1 excludes from the
    inference path on its own 243 MB fp32 bulk. ir-CSN-R50 is the first
    licence-clean IG-65M checkpoint that PASSES D1'S FENCE UNCHANGED: 12.31 M
    backbone params, 47.22 MB fp32 on disk, well inside the 100 MB fp32 bar.

    IT IS A HYPOTHESIS, NOT A RESULT. Its native clip is 32 frames at stride
    2; we run T=16, a 2x temporal squeeze, into a 4-channel depth+IR stem the
    corpus never saw. QUEUE.md §4 pre-registers the gate and the null reading;
    the board's record on cross-family members is 0/2, both -1 clip (m109 (1)).

    NORMALISATION IS READ, NOT INHERITED. The checkpoint's own embedded config
    declares mean [123.675, 116.28, 103.53] / std [58.395, 57.12, 57.375] on the
    0-255 scale. Divided by 255 those are (0.485, 0.456, 0.406) / (0.229, 0.224,
    0.225) -- IMAGENET statistics, byte-identical to this repo's `thermal` /
    `depthir` NORM keys and NOT the `*_kinetics` keys s3d and mc3_18 use. The
    registry below maps it accordingly. A wrong NORM does not raise; it just
    costs accuracy quietly, which is why this was read off the artefact.

    TWO S3D TRAPS DO NOT APPLY, AND ONE DOES.
      * Head pooling: AdaptiveAvgPool3d(1) means T=16 needs no re-specified
        pool kernel. At T=16 the x8 temporal stride leaves T'=2 and the adaptive
        pool takes it; nothing to configure, nothing to get wrong.
      * BatchNorm: this backbone is all BatchNorm3d, so `partial_bn` must keep
        predicating on _BatchNorm (it does -- _freeze_bn_except_first is shared).
      * `tsm` is accepted and ignored, as on the other 3D branches: a 3D
        convolution IS the temporal operator.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_segment: int = 16,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        pretrained: bool = False,
        partial_bn: bool = True,
        tsm: bool = True,  # accepted, ignored -- a 3D conv IS the temporal operator
        weights: str | None = None,
        load_k400_head: bool = False,
        mixstyle: bool = False,
        depth: int = 50,
        input_adapt: str | None = None,   # §63 (D22 d): None | "clipnorm" | "motion"
        freeze_stages: int = 0,           # §71 (D24): 0 = none; n = the stem + layer1..layer_n frozen (params AND BN)
        ip: bool = False,                 # §113 (D48 (2)): the 'ip' bottleneck (depth 152 only) with its own IG-65M→K400 file
    ) -> None:
        super().__init__()
        # BOTH IMPORT CONTEXTS, AND THIS IS NOT DEFENSIVE PADDING. package.py
        # imports this file as `src.model` (package), while predict.py inserts
        # src/ on sys.path and imports it as `model` (top-level) -- so a plain
        # relative import raises "attempted relative import with no known parent
        # package" INSIDE THE INFERENCE PATH ONLY. check_model.py runs in the
        # package context and passed 55/55 while that was broken; the offline and
        # robustness suites are what caught it. dataset/predict already use the
        # absolute form for exactly this reason.
        try:
            from .csn import IPCSNResNet152, IRCSNResNet50, IRCSNResNet152, strip_prefix
        except ImportError:                              # script context
            from csn import IPCSNResNet152, IRCSNResNet50, IRCSNResNet152, strip_prefix

        self.n_segment = n_segment
        self.in_channels = in_channels
        self.tsm_enabled = False
        self.partial_bn = partial_bn

        if depth not in (50, 152):
            raise ValueError(f"ir-CSN depth must be 50 or 152, got {depth}")
        if ip and depth != 152:
            raise ValueError(f"the ip-CSN is registered at depth 152 only (§113), got depth={depth}")
        self.depth = depth
        self.ip = ip
        if weights is None:
            weights = CSN_K400_FT_IP152 if ip else (CSN_K400_FT if depth == 50 else CSN_K400_FT_R152)
        net = (IPCSNResNet152 if ip else (IRCSNResNet50 if depth == 50 else IRCSNResNet152))(in_channels=3)
        if pretrained:
            path = CSN_WEIGHT_DIR / weights
            if not path.is_file():
                raise FileNotFoundError(
                    f"CSN weights not found at {path}. B0 fetches them foreground and "
                    f"single-stream (GOAL-SPRINT §9); set CSN_WEIGHT_DIR to override."
                )
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sd = strip_prefix(ck["state_dict"] if "state_dict" in ck else ck)
            head_sd = {k[len("cls_head.fc_cls."):]: v for k, v in sd.items()
                       if k.startswith("cls_head.fc_cls.")}
            sd = {k: v for k, v in sd.items() if not k.startswith("cls_head.")}
            # strict=True. src/csn.py is named so that this needs no remap, and
            # a remap is exactly the silent failure this load is here to prevent.
            net.load_state_dict(sd, strict=True)

        # Stem AFTER loading: _adapt_stem3d rescales by 3/in_channels to hold the
        # summed response, and it must act on the pretrained kernels, not on
        # random ones. (Same order as S3DBranch. Getting it backwards is silent.)
        net.conv1.conv = _adapt_stem3d(net.conv1.conv, in_channels)
        # §63 (D22 d): input adapters live in the model so the cache, the loaders and the packer are untouched;
        # the checkpoint stamps `input_adapt` and every rebuild passes it back (model.adapt_kw).
        if input_adapt not in (None, "clipnorm", "motion", "irnorm", "valid"):
            raise ValueError(f"input_adapt must be None, 'clipnorm', 'motion', 'irnorm' or 'valid', got {input_adapt!r}")
        self.input_adapt = input_adapt
        if input_adapt == "motion":
            net.conv1.conv = _widen_stem3d_zero(net.conv1.conv, 1)   # 4th channel zero-initialised: step 0 == the incumbent
        # §96 (d) (D29): `irnorm` maps the IR plane — channel 3 of the 4-channel depth+IR input — alone; §96 (c): `valid` takes the
        # depth-validity plane as a FIFTH channel; a 3-channel branch has neither, so both refuse rather than act on a colour plane.
        if input_adapt in ("irnorm", "valid") and in_channels != 4:
            raise ValueError(f"{input_adapt} is the 4-channel depth+IR branch's adapter; got in_channels={in_channels}")
        if input_adapt == "valid":
            # §96 (c) (D29): the plane arrives FROM THE LOADER (dataset.depth_validity, under the `depthir_valid` norm key) as the
            # fifth channel; the stem gains one ZERO kernel, so step 0 is BIT-IDENTICAL to the incumbent (tests/check_p9_input_arms.py).
            net.conv1.conv = _widen_stem3d_zero(net.conv1.conv, 1)
        self.IR_MEAN, self.IR_STD = IRNORM_IR_MEAN, IRNORM_IR_STD

        self.backbone = net
        # §71 (D24): partial freezing — the stem and the first n stages neither train nor update BN statistics;
        # stamped into every checkpoint (`freeze_stages`) and passed back by adapt_kw so a rebuild (swad_finalize,
        # the cacher, predict) carries the same frozen set and the SWA BN refit stays a no-op on them.
        if freeze_stages < 0 or freeze_stages > 4:
            raise ValueError(f"freeze_stages must be 0..4, got {freeze_stages}")
        self.freeze_stages = freeze_stages
        self._frozen_modules = ([net.conv1] + [getattr(net, f"layer{i}") for i in range(1, freeze_stages + 1)]
                                if freeze_stages else [])
        for mod in self._frozen_modules:
            for p_ in mod.parameters():
                p_.requires_grad_(False)
            mod.eval()
        # T8 (QUEUE §35): MixStyle after layer1 and layer2, train mode only, via forward hooks so the
        # backbone's key names (the strict-load contract, src/csn.py) are untouched. Off => no hooks.
        self.mixstyle = MixStyle() if mixstyle else None
        if self.mixstyle is not None:
            for layer in (net.layer1, net.layer2):
                layer.register_forward_hook(lambda _m, _i, out: self.mixstyle(out))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        head = nn.Linear(2048, num_classes)
        nn.init.normal_(head.weight, 0, 0.001)
        nn.init.zeros_(head.bias)
        # PROBE PATH ONLY (tools/csn_probe.py). m102 read our label off the
        # teacher's 400 KINETICS logits; reading it off a 2048-d penultimate
        # vector instead would compare a wider linear map to a narrower one and
        # inflate the number. Loading the checkpoint's own K400 head keeps the
        # readout at m102's width, so the two are the same measurement with the
        # backbone swapped. Never used by a shipping member (num_classes=40).
        if load_k400_head:
            if not pretrained:
                raise ValueError("load_k400_head needs pretrained=True")
            if num_classes != 400:
                raise ValueError(f"the K400 head is 400-way, got num_classes={num_classes}")
            head.load_state_dict(head_sd)
        self.fc = head

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        if mode and self.partial_bn:
            _freeze_bn_except_first(self.backbone)
        for mod in getattr(self, "_frozen_modules", []):   # §71: frozen stages stay in eval (their BNs never move)
            mod.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate 2048-d vector -- what the frozen-readout probe reads."""
        if x.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t != self.n_segment:
            raise ValueError(f"model built for n_segment={self.n_segment}, got T={t}")
        x = self._adapt_input(x)
        return self.pool(self.backbone(x.permute(0, 2, 1, 3, 4))).flatten(1)

    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        """§63. clipnorm: per-clip, per-channel standardisation of the model input (the embedded-norm space)
        over (T,H,W). motion: the signed per-frame temporal difference, channel-averaged, appended as a 4th
        channel (t=0 gets a zero frame). None: identity."""
        if self.input_adapt == "clipnorm":
            mu = x.mean(dim=(1, 3, 4), keepdim=True)
            sd = x.std(dim=(1, 3, 4), keepdim=True, unbiased=False)
            return (x - mu) / (sd + 1e-5)
        if self.input_adapt == "motion":
            d = (x[:, 1:] - x[:, :-1]).mean(dim=2, keepdim=True)
            d = torch.cat([torch.zeros_like(d[:, :1]), d], dim=1)
            return torch.cat([x, d], dim=2)
        if self.input_adapt == "irnorm":
            # §96 (d) (D29, QUEUE §96): the IR plane ALONE, mapped [p1, p99] -> [0, 1] PER CLIP from the clip's own pixels over
            # (T, H, W). The plane arrives in the embedded-norm space; it is de-embedded with the shipped depthir IR constants
            # (dataset.NORM["depthir"], channel 3), mapped, clamped, and re-embedded with the same constants, so a full-range
            # plane lands where the incumbent's did and the depth planes are untouched. kthvalue rather than quantile (no
            # element cap, deterministic). Per-clip statistics of THIS clip's own input and nothing else (L7; T-L7 holds):
            # predict.py, the cacher, the finaliser and the packers all inherit it through the `input_adapt` stamp.
            ir = x[:, :, 3:4]
            raw = ir * self.IR_STD + self.IR_MEAN
            flat = raw.reshape(raw.shape[0], -1)
            n = flat.shape[1]
            k1, k99 = int(round(0.01 * (n - 1))) + 1, int(round(0.99 * (n - 1))) + 1
            p1 = flat.kthvalue(k1, dim=1).values.view(-1, 1, 1, 1, 1)
            p99 = flat.kthvalue(k99, dim=1).values.view(-1, 1, 1, 1, 1)
            mapped = ((raw - p1) / (p99 - p1).clamp_min(1e-3)).clamp_(0.0, 1.0)
            return torch.cat([x[:, :, :3], (mapped - self.IR_MEAN) / self.IR_STD], dim=2)
        # "valid" (§96 c) is the identity here: its plane is the loader's fifth channel, already in x.
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.features(x)))


class SkeletonTCN(nn.Module):
    """D8's skeleton branch — a 1-D temporal ConvNet over pose features (P2).

    configs/baseline.yaml's skeleton block, followed verbatim: channels
    [102, 128, 256, 256, 256], kernel 5, GAP over T, dropout 0.3, fc 256→40.
    Trained FROM SCRATCH (D10 — the NTU pretraining ask was declined by name).
    Input (B, T, 102): 17 H36M joints × (xyz + first difference), built by
    src/skeleton_data.py under D14(d)'s pose-CONTENT-only clause.

    Recorded, not hidden: the config's own `params_m: 0.35` does not follow
    from its channel/kernel spec — [102,128,256,256,256] at k=5 is 0.90 M
    params = 1.79 MB fp16. The spec's STRUCTURE is followed; the stale params
    line is corrected in the ledger (byte fence unaffected: 75.03 + 1.79 MB is
    far inside D6's 95).
    """

    def __init__(self, in_channels: int = 102, n_segment: int = 24,
                 n_classes: int = 40, dropout: float = 0.3, **_) -> None:
        super().__init__()
        chs = (in_channels, 128, 256, 256, 256)
        layers: list[nn.Module] = []
        for a, b in zip(chs[:-1], chs[1:]):
            layers += [nn.Conv1d(a, b, kernel_size=5, padding=2, bias=False),
                       nn.BatchNorm1d(b), nn.ReLU(inplace=True)]
        self.tcn = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(chs[-1], n_classes)
        self.n_segment = n_segment
        self.in_channels = in_channels  # train.py stamps this into checkpoints

    def features(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        if x.dim() != 3:
            raise ValueError(f"SkeletonTCN wants (B, T, F) rank-3 input, got rank {x.dim()}")
        return self.tcn(x.transpose(1, 2)).mean(dim=2)     # GAP over T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.features(x)))


# ── §68 (D23): VideoMAE V2 ViT-S as the thermal member — the third backbone family ──────────────
class VMAEBranch(nn.Module):
    """VideoMAE V2 ViT-S/16 (K710-distilled, K400-fine-tuned) as a branch. QUEUE §68 / D23.

    Same external contract as CSNBranch -- (B, T, C, H, W) in, (B, NUM_CLASSES) out; the network itself is
    src/vmae.py (strict-load key contract). Native input: T = n_segment frames at img_size² (the ViT's
    position table is built for exactly that grid — T = 16, 224² for the checkpoint; a different grid RAISES
    rather than interpolating silently). No BatchNorm anywhere (LayerNorm) — partial_bn is accepted and
    meaningless; `tsm` accepted and ignored (a 3-D patch embedding is the temporal operator).

    NORMALISATION IS READ, NOT INHERITED: the zoo config's data_preprocessor declares mean [123.675, 116.28,
    103.53] / std [58.395, 57.12, 57.375] at 0-255 -- ImageNet after /255, this repo's `thermal`/`depthir`
    keys, the same as ir-CSN (m225). The 400-way K400 head is dropped at load; the branch head is a fresh
    Linear(384, 40) behind dropout 0.5 (CSNBranch's head recipe). drop_path 0.1 = VideoMAE V2's ViT-S
    fine-tune default (a model property, declared for both optimiser arms of §68).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_segment: int = 16,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        pretrained: bool = False,
        partial_bn: bool = True,      # accepted, meaningless: no BN exists
        tsm: bool = True,             # accepted, ignored
        weights: str | None = None,
        img_size: int = 224,
        drop_path_rate: float = 0.1,
        input_adapt: str | None = None,
        variant: str = "s",           # §74: "b" = ViT-B (embed 768, 12 heads)
        depth: int | None = None,     # §90 (D27): blocks 0 … depth−1 only (ViT-B[11] / ViT-B[10]); None = the zoo's 12
    ) -> None:
        super().__init__()
        try:
            from .vmae import VisionTransformer, strip_backbone, truncate_backbone
        except ImportError:                              # script context (predict.py inserts src/ on sys.path)
            from vmae import VisionTransformer, strip_backbone, truncate_backbone
        if variant not in VMAE2_VARIANTS:
            raise ValueError(f"vmae2 variant must be one of {sorted(VMAE2_VARIANTS)}, got {variant!r}")
        embed_dims, num_heads, default_weights = VMAE2_VARIANTS[variant]
        self.variant = variant
        self.depth = VMAE2_DEPTH if depth is None else int(depth)
        if not 0 < self.depth <= VMAE2_DEPTH:
            raise ValueError(f"vmae2 depth must lie in [1, {VMAE2_DEPTH}], got {depth!r}")
        if input_adapt is not None:
            raise ValueError(f"input_adapt is not wired for vmae2_vits (got {input_adapt!r}); §63's adapters are CSN-only")
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.tsm_enabled = False
        self.partial_bn = False       # nothing to freeze; train.py's BN refit is a no-op (no BN modules)
        self.input_adapt = None
        net = VisionTransformer(in_channels=3, num_frames=n_segment, img_size=img_size, drop_path_rate=drop_path_rate,
                                embed_dims=embed_dims, num_heads=num_heads, depth=self.depth)
        if pretrained:
            path = CSN_WEIGHT_DIR / (weights or default_weights)
            if not path.is_file():
                raise FileNotFoundError(f"VideoMAE V2 weights not found at {path}; set CSN_WEIGHT_DIR to override.")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sd = ck["state_dict"] if "state_dict" in ck else ck
            backbone_sd, _head_sd = strip_backbone(sd)       # the 400-way head is dropped here
            if self.depth < VMAE2_DEPTH:
                # §90: the shorter module is loaded STRICT after the dropped block's keys are removed; the dropped
                # names are printed — the load test's evidence (QUEUE §90 (2)).
                backbone_sd, dropped = truncate_backbone(backbone_sd, self.depth)
                print(f"[vmae] ViT-{variant.upper()}[{self.depth}]: {self.depth}/{VMAE2_DEPTH} blocks consumed from "
                      f"{path.name}; {len(dropped)} tensors dropped by name: {dropped}; the "
                      f"{tuple(_head_sd.get('cls_head.fc_cls.weight', torch.empty(0, 0)).shape)[0]}-way head dropped")
            # strict=True: src/vmae.py is named so that this needs no remap; a remap is the silent failure
            # this load exists to prevent. The position table is a non-persistent buffer, so it is neither
            # expected nor missing.
            net.load_state_dict(backbone_sd, strict=True)
        # Stem AFTER loading (trap 4): copy RGB, mean-fill the extra channel(s), ×3/in_channels — response-
        # preserving on a channel-uniform input; acts on the PRETRAINED kernels (§68 b, the depth+IR side).
        net.patch_embed.projection = _adapt_stem3d(net.patch_embed.projection, in_channels)
        self.backbone = net
        self.dropout = nn.Dropout(p=dropout)
        head = nn.Linear(net.embed_dims, num_classes)
        nn.init.normal_(head.weight, 0, 0.001)
        nn.init.zeros_(head.bias)
        self.fc = head

    def layer_id(self, name: str) -> int:
        """Layer-wise lr decay groups (§68 arm b): patch embedding 0, block i → i + 1, fc_norm/head → depth + 1."""
        if name.startswith("backbone.patch_embed"):
            return 0
        if name.startswith("backbone.blocks."):
            return int(name.split(".")[2]) + 1
        return self.backbone.depth + 1

    @property
    def n_layers(self) -> int:
        return self.backbone.depth + 2

    def features(self, x: torch.Tensor) -> torch.Tensor:        # (B, T, C, H, W)
        if x.dim() != 5:
            raise ValueError(f"VMAEBranch wants (B, T, C, H, W) rank-5 input, got rank {x.dim()}")
        return self.backbone(x.permute(0, 2, 1, 3, 4).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.features(x)))


# ── §66 (D23): the IMU member — three small 1-D architectures over 40 channels × T = 24 ─────────
# (B, T, 40) in — src/imu_data.py's tensor: 5 sites × [accel xyz, gyro xyz, pitch, roll] — (B, 40) out.
# Per-site, per-channel z-scoring from the fold's TRAIN subjects lives in the model as BUFFERS
# (in_mean, in_std): train.py fits them on the fold's train split and they travel inside the state
# dict — "stamped into the checkpoint, never a dataset.NORM key" (QUEUE §66; the skeleton_raw sentinel
# pattern below). An ABSENT site (a block that is exactly zero over every T and channel — real data is
# never exactly zero 192 times) stays exactly zero after z-scoring, so it contributes nothing: the
# presence mask is never an input (L5); the block's own content is the criterion.
class _IMUBase(nn.Module):
    def __init__(self, in_channels: int = 40, n_segment: int = 24, n_classes: int = 40,
                 dropout: float = 0.3, **_) -> None:
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(in_channels))
        self.register_buffer("in_std", torch.ones(in_channels))
        self.dropout = nn.Dropout(dropout)
        self.n_segment = n_segment
        self.in_channels = in_channels   # train.py stamps this into checkpoints
        self.n_sites = 5

    @torch.no_grad()
    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.in_mean.copy_(mean.to(self.in_mean))
        self.in_std.copy_(std.clamp_min(1e-6).to(self.in_std))

    def normalise(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 40)
        if x.dim() != 3:
            raise ValueError(f"{type(self).__name__} wants (B, T, F) rank-3 input, got rank {x.dim()}")
        B, T, F = x.shape
        per_site = F // self.n_sites
        absent = (x.abs().reshape(B, T, self.n_sites, per_site).sum(dim=(1, 3)) == 0)   # (B, 5)
        z = (x - self.in_mean) / self.in_std
        keep = (~absent).to(z.dtype).repeat_interleave(per_site, dim=1)[:, None, :]      # (B, 1, 40)
        return z * keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.features(self.normalise(x))))


class IMUConv1D(_IMUBase):
    """`cnn1d_imu` — the SkeletonTCN shape verbatim: [40,128,256,256,256], k = 5, BN, ReLU, GAP over T,
    dropout 0.3, fc → 40. From scratch; every BN live."""

    def __init__(self, in_channels: int = 40, n_segment: int = 24, n_classes: int = 40,
                 dropout: float = 0.3, dilations: tuple[int, ...] = (1, 1, 1, 1), **kw) -> None:
        super().__init__(in_channels, n_segment, n_classes, dropout, **kw)
        chs = (in_channels, 128, 256, 256, 256)
        layers: list[nn.Module] = []
        for a, b, d in zip(chs[:-1], chs[1:], dilations):
            layers += [nn.Conv1d(a, b, kernel_size=5, padding=2 * d, dilation=d, bias=False),
                       nn.BatchNorm1d(b), nn.ReLU(inplace=True)]
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(chs[-1], n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) normalised
        return self.tcn(x.transpose(1, 2)).mean(dim=2)


class IMUDilatedTCN(IMUConv1D):
    """`dtcn_imu` — the same widths with dilations 1/2/4/8: the one architectural axis with a mechanism
    at 10.87 Hz over ~2.2 s (a receptive field spanning the clip)."""

    def __init__(self, in_channels: int = 40, n_segment: int = 24, n_classes: int = 40,
                 dropout: float = 0.3, **kw) -> None:
        super().__init__(in_channels, n_segment, n_classes, dropout, dilations=(1, 2, 4, 8), **kw)


class IMUTransformer(_IMUBase):
    """`xf_imu` — Linear 40 → 128, learned positional embedding over T, a 2-layer 4-head
    TransformerEncoder (d = 128, ff 256, LayerNorm — NO BatchNorm anywhere), mean over T, dropout, fc."""

    def __init__(self, in_channels: int = 40, n_segment: int = 24, n_classes: int = 40,
                 dropout: float = 0.3, d_model: int = 128, **kw) -> None:
        super().__init__(in_channels, n_segment, n_classes, dropout, **kw)
        self.proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_segment, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=2 * d_model, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(d_model, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) normalised
        h = self.proj(x)
        if h.shape[1] == self.pos.shape[1]:
            h = h + self.pos
        else:  # a cropped or re-sampled T: interpolate the positions (only ever T = n_segment in practice)
            h = h + torch.nn.functional.interpolate(self.pos.transpose(1, 2), size=h.shape[1],
                                                    mode="linear", align_corners=True).transpose(1, 2)
        return self.encoder(h).mean(dim=1)


IMU_ARCHS = frozenset({"cnn1d_imu", "dtcn_imu", "xf_imu"})


# THE ARCH REGISTRY. Every architecture a checkpoint may name, and the
# dataset.NORM key it was trained under. A checkpoint records BOTH (train.py) and
# predict.py dispatches on BOTH; an unknown value raises rather than defaulting.
#
# Why a registry rather than a branch->model mapping: J1 puts two architectures
# and two normalisations in play at once, and predict.py is forbidden from
# reading YAML, so a checkpoint is the only channel through which the loader can
# learn how the weights were trained. A wrong ARCH raises on load_state_dict and
# is therefore self-announcing. A wrong NORM does not raise -- it feeds the
# network a distribution it never saw and quietly costs accuracy, which in a
# mixed ensemble is near-certain and undetectable from the CSV. That asymmetry is
# why `norm` is persisted even though it is currently derivable from `branch`.

class FrameNetBlock(nn.Module):
    """§109 (D44): skomuro's residual block, verbatim -- two 3x3 convs with BN and a 1x1 projection shortcut when the shape
    changes (the public notebook `skomuro/cuhk-x-14th-place-0-8-thermal-baseline`, its `Block`)."""

    def __init__(self, a: int, b: int, s: int = 1) -> None:
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(a, b, 3, s, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(),
                               nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b))
        self.d = (nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b))
                  if (a != b or s != 1) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c(x) + self.d(x))


class FrameNet(nn.Module):
    """§109 (D44; QUEUE §109): the from-scratch 2D per-frame CNN of skomuro's public notebook
    `skomuro/cuhk-x-14th-place-0-8-thermal-baseline` -- stem Conv2d(3->32, 7, s2) + BN + ReLU + MaxPool(3, s2), four residual
    blocks 32 -> 32 -> 64 -> 128 -> 256 (strides 1, 2, 2, 2), global average pool, Linear(256, 40): 1,240,520 parameters at
    in_channels 3 (G §4 counted it to the unit). Consumes (B, T, C, H, W) like every image branch here and returns
    (B, NUM_CLASSES): the per-frame logits are averaged over T (skomuro's `.mean(1)` = TSN's consensus; F4's within-model rule),
    so the model is frame-order INVARIANT by construction (tests/check_model.py asserts it). No pretrained weights exist for it
    (`pretrained` is accepted and ignored); `tsm` is accepted and ignored (no temporal operator); `partial_bn` behaves as on
    TSMResNet18 and the recipe turns it OFF (--no-partial-bn: every BN live, from scratch); `dropout` 0.0 keeps the published
    head exactly. The adapters it cannot honour (input_adapt, freeze_stages, mixstyle, weights) raise rather than pass silently.
    Grade the citation correctly: the notebook is "the deliberately pre-optimization version", trains 8 epochs and prints no
    validation accuracy -- its link to the 0.83582 board score is inferred, not shown (G §4 defect 1)."""

    def __init__(self, in_channels: int = 3, n_segment: int = 8, num_classes: int = NUM_CLASSES, dropout: float = 0.0,
                 pretrained: bool = False, partial_bn: bool = False, tsm: bool = False, mixstyle: bool = False,
                 input_adapt: str | None = None, freeze_stages: int = 0, weights: str | None = None) -> None:
        super().__init__()
        if mixstyle or input_adapt or freeze_stages or weights:
            raise ValueError("framenet has no MixStyle, input adapter, frozen stages or pretrained weights "
                             f"(got mixstyle={mixstyle}, input_adapt={input_adapt!r}, freeze_stages={freeze_stages}, weights={weights!r})")
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.partial_bn = partial_bn
        self.f = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            FrameNetBlock(32, 32), FrameNetBlock(32, 64, 2), FrameNetBlock(64, 128, 2), FrameNetBlock(128, 256, 2),
            nn.AdaptiveAvgPool2d(1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def train(self, mode: bool = True):  # noqa: A003
        super().train(mode)
        if mode and self.partial_bn:
            _freeze_bn_except_first(self.f)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (B,T,C,H,W), got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if t != self.n_segment:
            raise ValueError(f"model built for n_segment={self.n_segment}, got T={t}")
        flat = x.reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        z = self.f(flat).flatten(1)
        logits = self.fc(self.dropout(z)).view(b, t, -1)
        return logits.mean(dim=1)  # consensus over T


DEFAULT_ARCH = "tsm_resnet18"

# arch -> callable(in_channels, n_segment, **kw).
_ARCH_BUILDERS = {
    "tsm_resnet18": TSMResNet18,
    "s3d": S3DBranch,
    "mc3_18": MC3Branch,
    "ircsn_r50": CSNBranch,
    "ircsn_r152": functools.partial(CSNBranch, depth=152),   # §49 (D19 a)
    "ipcsn_r152": functools.partial(CSNBranch, depth=152, ip=True),   # §113 (D48 (2), reading α): ip-CSN-152, the thermal-member capacity arm — the TWELFTH gated contrast
    "tsm_r50_ssv2": TSMResNet50,                              # §61 (D22 b): SSv2-pretrained TSM-R50
    "vmae2_vits": VMAEBranch,                                 # §68 (D23): VideoMAE V2 ViT-S, the third family
    "vmae2_vitb": functools.partial(VMAEBranch, variant="b"), # §74 (D24): ViT-B, a D6-gated pricing arm
    "vmae2_vitb11": functools.partial(VMAEBranch, variant="b", depth=11),   # §90 (D27): ViT-B with block 11 dropped, under the bytes
    "vmae2_vitb10": functools.partial(VMAEBranch, variant="b", depth=10),   # §90: the declared fallback if depth 11 weighs ≥ 95,000,000 B
    "tcn_1d": SkeletonTCN,
    "cnn1d_imu": IMUConv1D,                                   # §66 (D23): the IMU member, arm (i)
    "dtcn_imu": IMUDilatedTCN,                                # §66: arm (ii)
    "xf_imu": IMUTransformer,                                 # §66: arm (iii)
    "framenet": FrameNet,                                    # §109 (D44): skomuro's from-scratch 2D FrameNet, the thermal third member
}

# arch -> {branch: dataset.NORM key}. ImageNet statistics for the 2D backbone,
# Kinetics-400 for the video-pretrained one -- read off
# S3D_Weights.KINETICS400_V1.transforms(), never borrowed from the R3D family.
_ARCH_NORM = {
    "tsm_resnet18": {"thermal": "thermal", "depthir": "depthir", "depthir_crop": "depthir",
                     "depthir_cropsq": "depthir"},
    "s3d": {"thermal": "thermal_kinetics", "depthir": "depthir_kinetics",
            "depthir_crop": "depthir_kinetics", "depthir_cropsq": "depthir_kinetics"},
    # mc3_18 declares the SAME Kinetics statistics as s3d -- read off
    # MC3_18_Weights.KINETICS400_V1.transforms(), not inherited by family.
    "mc3_18": {"thermal": "thermal_kinetics", "depthir": "depthir_kinetics",
               "depthir_crop": "depthir_kinetics", "depthir_cropsq": "depthir_kinetics"},
    # ir-CSN-R50 is the ONE video-pretrained arch here that does NOT take the
    # Kinetics keys. Its checkpoint's embedded config declares mean
    # [123.675, 116.28, 103.53] / std [58.395, 57.12, 57.375] at 0-255, i.e.
    # ImageNet after /255 -- read off the artefact, not inherited from the
    # neighbouring 3D rows. Inheriting by family here would have been silent.
    "ircsn_r50": {"thermal": "thermal", "depthir": "depthir",
                  "depthir_crop": "depthir", "depthir_cropsq": "depthir"},   # A3: a framing change only
    # ir-CSN-152: the same checkpoint family and the same embedded normalisation (read off the R152 file's meta cfg at §49's build).
    "ircsn_r152": {"thermal": "thermal", "depthir": "depthir",
                   "depthir_crop": "depthir", "depthir_cropsq": "depthir"},
    # ip-CSN-152 (§113): the converted VMZ file carries NO meta cfg, so the statistics were READ off the zoo's two configs on
    # 2026-09-08 — mmaction2 main `configs/_base_/models/ircsn_r152.py` (data_preprocessor mean [123.675, 116.28, 103.53] /
    # std [58.395, 57.12, 57.375], the base the ip config inherits) and the v0.24.1 ir-CSN-152 bnfrozen base's `img_norm_cfg`
    # (the same values, to_bgr=False) — i.e. ImageNet after /255, the ir-CSN keys. Not inherited by family.
    "ipcsn_r152": {"thermal": "thermal", "depthir": "depthir",
                   "depthir_crop": "depthir", "depthir_cropsq": "depthir"},
    # TSM-R50 SSv2 (§61): the checkpoint's meta cfg declares the same ImageNet statistics at 0-255.
    "tsm_r50_ssv2": {"thermal": "thermal", "depthir": "depthir",
                     "depthir_crop": "depthir", "depthir_cropsq": "depthir"},
    # VideoMAE V2 ViT-S (§68): the zoo config's data_preprocessor declares the same ImageNet statistics (m225).
    "vmae2_vits": {"thermal": "thermal", "depthir": "depthir",
                   "depthir_crop": "depthir", "depthir_cropsq": "depthir"},
    "vmae2_vitb": {"thermal": "thermal", "depthir": "depthir",
                   "depthir_crop": "depthir", "depthir_cropsq": "depthir"},
    # §90 (D27): the truncated ViT-B members are the same checkpoint family — the same statistics, read off the same config.
    "vmae2_vitb11": {"thermal": "thermal", "depthir": "depthir",
                     "depthir_crop": "depthir", "depthir_cropsq": "depthir"},
    "vmae2_vitb10": {"thermal": "thermal", "depthir": "depthir",
                     "depthir_crop": "depthir", "depthir_cropsq": "depthir"},
    # Skeleton features are root-relative pose values used RAW. "skeleton_raw"
    # is deliberately NOT a NORM key: nothing may apply image statistics to
    # pose data, and an accidental NORM["skeleton_raw"] lookup must KeyError
    # loudly instead of normalising silently.
    "tcn_1d": {"skeleton": "skeleton_raw"},
    # §66: the IMU tensor is z-scored INSIDE the model from stamped fold-train buffers; "imu_raw" is the
    # same deliberate non-NORM sentinel as skeleton_raw — an accidental NORM["imu_raw"] must KeyError.
    "cnn1d_imu": {"imu": "imu_raw"},
    "dtcn_imu": {"imu": "imu_raw"},
    "xf_imu": {"imu": "imu_raw"},
    "framenet": {"thermal": "thermal", "depthir": "depthir", "depthir_crop": "depthir", "depthir_cropsq": "depthir", "depthir_ord3": "depthir_ord3", "depthir_ordgrad": "depthir_ordgrad"},   # §109: from scratch, the branches' standard statistics (the same table as ir-CSN); §109 trains the thermal member only
}

# Architectures whose forward takes a rank-5 tensor end to end. channels_last is
# a rank-4 memory format and RAISES on these ("required rank 4 tensor to use
# channels_last"), so train.py must ask before applying it. Loud rather than
# silent, but it still has to be asked.
RANK5_ARCHS = frozenset({"s3d", "mc3_18", "ircsn_r50", "ircsn_r152", "ipcsn_r152", "vmae2_vits", "vmae2_vitb",
                         "vmae2_vitb11", "vmae2_vitb10"})   # rank-5 inputs; channels_last is rank-4
# Image archs with no BatchNorm at all (LayerNorm): partial_bn and the SWA BN refit are no-ops on them.
NO_BN_ARCHS = frozenset({"vmae2_vits", "vmae2_vitb", "vmae2_vitb11", "vmae2_vitb10"})
# Rank-3 (B, T, F) archs, trained from scratch with every BN live: no channels_last, no partial_bn,
# no pretrained weights. tcn_1d (P2) and the three §66 IMU archs.
NON_IMAGE_ARCHS = frozenset({"tcn_1d"}) | IMU_ARCHS

BRANCH_CHANNELS = {"thermal": 3, "depthir": 4, "depthir_crop": 4, "depthir_cropsq": 4, "depthir_ord3": 4, "depthir_ordgrad": 4,
                   "skeleton": 102,  # 17 H36M joints × (xyz + Δxyz)
                   "imu": 40}        # 5 sites × (accel xyz + gyro xyz + pitch + roll), src/imu_data.py


# §96 (a)/(b) (D29): the ordinal branches take their arch's depth+IR statistics — the ImageNet-statistics archs name the NEW keys
# (the same values; the stamp names the ordinal input), the Kinetics-statistics archs their own depthir key.
for _a, _t in _ARCH_NORM.items():
    if "depthir" in _t:
        for _b in ("depthir_ord3", "depthir_ordgrad"):
            _t.setdefault(_b, _b if _t["depthir"] == "depthir" else _t["depthir"])


def norm_key(branch: str, arch: str = DEFAULT_ARCH) -> str:
    """The dataset.NORM key this (arch, branch) pair trains and infers under.

    Single source of truth for train.py's checkpoint stamp and predict.py's
    lookup, so the two cannot drift. Raises on an unknown arch or branch.
    """
    if arch not in _ARCH_NORM:
        raise ValueError(f"unknown arch {arch!r}; known: {sorted(_ARCH_NORM)}")
    table = _ARCH_NORM[arch]
    if branch not in table:
        raise ValueError(f"branch must be one of {sorted(table)}, got {branch!r}")
    return table[branch]


def build(branch: str, n_segment: int = 8, arch: str = DEFAULT_ARCH, **kw) -> nn.Module:
    """thermal -> 3 channels, depthir -> Depth(3) + IR(1) = 4.

    depthir_crop is the SAME network as depthir -- A/B #3 changes the framing of
    the input, nothing about the model. ResNet-18 is fully convolutional and
    global-average-pools, so 96x192 needs no architectural change; keeping the two
    identical is what makes the A/B a measurement of the crop.

    `arch` is dispatched, never guessed. An unrecognised value raises here so
    that it cannot become a silently-wrong model three call frames later.
    """
    if branch not in BRANCH_CHANNELS:
        raise ValueError(f"branch must be one of {sorted(BRANCH_CHANNELS)}, got {branch!r}")
    if arch not in _ARCH_BUILDERS:
        raise ValueError(f"unknown arch {arch!r}; known: {sorted(_ARCH_BUILDERS)}")
    return _ARCH_BUILDERS[arch](in_channels=BRANCH_CHANNELS[branch], n_segment=n_segment, **kw)


def adapt_kw(ck: dict) -> dict:
    """§63: the build() kwargs a checkpoint's stamped input adapter needs -- empty for every checkpoint that
    predates the flag, so no loader changes behaviour for the existing artefacts."""
    kw = {"input_adapt": ck["input_adapt"]} if ck.get("input_adapt") else {}
    if ck.get("freeze_stages"):                     # §71: the frozen set travels with the checkpoint
        kw["freeze_stages"] = int(ck["freeze_stages"])
    return kw


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fp16_megabytes(model: nn.Module) -> float:
    """Decimal MB at the packaging dtype -- the units the 100 MB cap uses."""
    return param_count(model) * 2 / 1e6
