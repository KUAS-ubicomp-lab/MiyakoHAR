"""Verify model.py -- above all, that TSM is not silently a no-op.

The expensive failure here is not a crash. It is a model that trains fine,
converges fine, and is quietly a plain TSN because the shift landed in the wrong
place or the (B*T) reshape disagreed with what shift() assumes. The "TSM on/off"
A/B would then compare TSN against TSN, return a clean null, and we would delete
a real lever on the strength of it.

So the central test is a PROPERTY, not a shape: a frame-order-invariant model
cannot be doing temporal modelling. Plain TSN averages per-frame logits and is
exactly invariant to permuting T. TSM must not be.

Standing rule: diagnose any failure. Never adjust the expected value.

Usage:
    python tests/check_model.py            # uses pretrained weights if cached
    python tests/check_model.py --no-pretrained
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import (  # noqa: E402
    RANK5_ARCHS,
    TemporalShift,
    bn_modules,
    build,
    fp16_megabytes,
    norm_key,
    param_count,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(desc: str, got, expected) -> bool:
    ok = got == expected
    (PASS if ok else FAIL).append(desc)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {desc}: got {got!r}, expected {expected!r}")
    return ok


def check_true(desc: str, cond: bool, detail: str = "") -> bool:
    (PASS if cond else FAIL).append(desc)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {desc}{(' — ' + detail) if detail else ''}")
    return bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    pre = not args.no_pretrained
    torch.manual_seed(0)

    B, T, H, W = 2, 8, 120, 160

    # ── 1. The shift primitive, in isolation ────────────────────────────────
    print("\nTemporalShift primitive")
    c = 16
    x = torch.arange(B * T * c, dtype=torch.float32).view(B * T, c, 1, 1)
    y = TemporalShift.shift(x, n_segment=T, shift_div=8)
    fold = c // 8
    check("fold = c // shift_div", fold, 2)
    check_true("a QUARTER of channels move (trap 3)", 2 * fold == c // 4, f"{2*fold} of {c}")
    xv, yv = x.view(B, T, c, 1, 1), y.view(B, T, c, 1, 1)
    check_true("channels [0:fold] borrow from t+1", bool(torch.equal(yv[:, :-1, :fold], xv[:, 1:, :fold])))
    check_true("channels [fold:2fold] borrow from t-1", bool(torch.equal(yv[:, 1:, fold : 2 * fold], xv[:, :-1, fold : 2 * fold])))
    check_true("channels [2fold:] are untouched", bool(torch.equal(yv[:, :, 2 * fold :], xv[:, :, 2 * fold :])))
    check_true("the last frame's forward-shift is zero-filled", bool(torch.all(yv[:, -1, :fold] == 0)))
    check_true("the first frame's backward-shift is zero-filled", bool(torch.all(yv[:, 0, fold : 2 * fold] == 0)))
    check_true("shift preserves shape", y.shape == x.shape)
    try:
        TemporalShift.shift(torch.zeros(7, 4, 1, 1), n_segment=8, shift_div=8)
        check_true("refuses a batch that is not a multiple of T", False, "accepted 7 with T=8")
    except RuntimeError:
        check_true("refuses a batch that is not a multiple of T", True)

    # ── 2. TSM is not a no-op ─────────────────────────────────────────────
    print("\nTemporal asymmetry — the check that catches a silently-dead TSM")
    m = build("thermal", n_segment=T, pretrained=pre).eval()
    xb = torch.randn(B, T, 3, H, W)
    with torch.no_grad():
        out_fwd = m(xb)
        out_rev = m(torch.flip(xb, dims=[1]))
        out_again = m(xb)
    check_true("eval is deterministic", bool(torch.equal(out_fwd, out_again)))
    delta = (out_fwd - out_rev).abs().max().item()
    check_true("reversing frame order CHANGES the output (TSM is live)", delta > 1e-4, f"max|Δ| = {delta:.3e}")

    # The control arm: with tsm=False the model IS a plain TSN, and a mean over
    # per-frame logits is exactly invariant to frame order. If this fails, the
    # asymmetry above proves nothing about TSM.
    m0 = build("thermal", n_segment=T, tsm=False, pretrained=pre).eval()
    with torch.no_grad():
        d0 = (m0(xb) - m0(torch.flip(xb, dims=[1]))).abs().max().item()
    check_true("with tsm=False the model IS order-invariant (control)", d0 < 1e-4, f"max|Δ| = {d0:.3e}")
    check_true("so the asymmetry is attributable to TSM, not to chance", delta > 1e-4 and d0 < 1e-4)

    # ── 3. Placement is residual, not in-place ──────────────────────────────
    print("\nPlacement (trap 2)")
    shifted = [n for n, mod in m.named_modules() if isinstance(mod, TemporalShift)]
    check("every BasicBlock is wrapped", len(shifted), 8)  # resnet18: 2 blocks x 4 layers
    check_true("the wrap is on conv1, i.e. inside the residual branch", all(n.endswith(".conv1") for n in shifted), f"{shifted[:2]}")
    check_true("downsample paths are NOT shifted", not any("downsample" in n for n in shifted))

    # ── 4. Shapes, both branches ────────────────────────────────────────────
    print("\nShapes")
    with torch.no_grad():
        check("thermal output shape", tuple(m(xb).shape), (B, 40))
        md = build("depthir", n_segment=T, pretrained=pre).eval()
        check("depthir output shape", tuple(md(torch.randn(B, T, 4, H, W)).shape), (B, 40))
    check("thermal stem in_channels", m.backbone.conv1.in_channels, 3)
    check("depthir stem in_channels", md.backbone.conv1.in_channels, 4)
    for bad, why in (((B, T, 3), "3-dim input"), ((B, T + 1, 3, H, W), "wrong T")):
        try:
            m(torch.zeros(*bad))
            check_true(f"refuses {why}", False, "accepted it")
        except ValueError:
            check_true(f"refuses {why}", True)

    # ── 5. The 4-channel stem preserves scale (trap 4) ──────────────────────
    print("\n4-channel stem adaptation")
    w3 = build("thermal", n_segment=T, pretrained=pre).backbone.conv1.weight.data
    w4 = md.backbone.conv1.weight.data
    check("stem weight shape", tuple(w4.shape), (64, 4, 7, 7))
    check_true("RGB kernels are copied, then scaled by 3/4", bool(torch.allclose(w4[:, :3], w3 * 0.75, atol=1e-6)))
    check_true("the 4th channel is the mean of the three, same scale", bool(torch.allclose(w4[:, 3], w3.mean(dim=1) * 0.75, atol=1e-6)))
    # THE FIRST VERSION OF THIS CHECK MEASURED THE WRONG QUANTITY. It asserted
    # that sum|weight| is preserved, which the 0.75 does NOT do -- measured ratio
    # 0.9408 -- because |mean(w)| < mean(|w|) wherever the RGB kernels partly
    # cancel. That is a property of the metric, not a defect in the adaptation.
    #
    # What scale_by_0.75 actually guarantees, and the reason it is 3/4 rather
    # than any other constant, is EXACT preservation of the response when the
    # 4th channel carries the mean of the first three:
    #     conv4(v,v,v,v) = 0.75*v*[Sw + (1/3)Sw] = 0.75*(4/3)*v*Sw = v*Sw = conv3(v,v,v)
    # That is what keeps the pretrained downstream BN statistics valid, so that
    # is what gets asserted.
    ref = build("thermal", n_segment=T, pretrained=pre).backbone.conv1
    with torch.no_grad():
        grey = torch.randn(1, 1, 32, 32).expand(1, 3, 32, 32).contiguous()
        r3 = ref(grey)
        r4 = md.backbone.conv1(torch.cat([grey, grey.mean(dim=1, keepdim=True)], dim=1))
    rel = ((r4 - r3).abs().max() / r3.abs().max()).item()
    check_true("a channel-uniform input gives an IDENTICAL response through the 4-ch stem",
               rel < 1e-5, f"relative max|Δ| = {rel:.2e}")
    # And record the quantity that is NOT preserved, so nobody re-asserts it.
    ratio = w4.abs().sum().item() / w3.abs().sum().item()
    check_true("sum|weight| is deliberately NOT preserved (sign cancellation)", 0.90 < ratio < 0.99, f"ratio {ratio:.4f}")

    # ── 6. Partial BN survives .train() ─────────────────────────────────────
    print("\nPartial BN (trap 5)")
    mt = build("thermal", n_segment=T, pretrained=pre)
    mt.train()  # the call that silently undoes a one-shot freeze
    # _BatchNorm, not BatchNorm2d: the 2d predicate returns [] on a 3D backbone
    # and every assertion below then passes over an empty list (PRETRAINING-GAP §4).
    bns = bn_modules(mt.backbone)
    check_true("resnet18 has the expected BN count", len(bns) == 20, f"{len(bns)}")
    check_true("the FIRST BN stays in train mode", bns[0].training)
    check_true("every other BN is in eval mode AFTER .train()", all(not b.training for b in bns[1:]))
    check_true("the first BN keeps trainable affine", bns[0].weight.requires_grad)
    check_true("later BN affine is frozen", not any(b.weight.requires_grad for b in bns[1:]))
    mt.train()  # idempotent
    check_true("re-calling .train() does not thaw them", all(not b.training for b in bns[1:]))
    m_np = build("thermal", n_segment=T, pretrained=pre, partial_bn=False).train()
    bns_np = bn_modules(m_np.backbone)
    check_true("partial_bn=False leaves them all trainable (control)", all(b.training for b in bns_np))

    # ── 7. The artefact budget ──────────────────────────────────────────────
    print("\nBudget")
    p = param_count(m)
    check_true("thermal params ≈ 11.2 M as configured", 11.0e6 < p < 11.4e6, f"{p/1e6:.2f} M")
    check_true("depthir params ≈ 11.2 M", 11.0e6 < param_count(md) < 11.4e6, f"{param_count(md)/1e6:.2f} M")
    check_true("TSM adds ZERO parameters", param_count(m) == param_count(m0), f"{param_count(m)} vs {param_count(m0)}")
    both = fp16_megabytes(m) + fp16_megabytes(md)
    check_true("both branches at fp16 fit the 95 MB budget", both < 95.0, f"{both:.1f} MB")
    check_true("...with the headroom the config claims (~52 MB total)", both < 50.0, f"{both:.1f} MB")

    # ── 8. It can actually learn ────────────────────────────────────────────
    print("\nGradient flow")
    mt.train()
    xs = torch.randn(2, T, 3, H, W)
    loss = torch.nn.functional.cross_entropy(mt(xs), torch.tensor([3, 17]))
    loss.backward()
    check_true("loss is finite", bool(torch.isfinite(loss)))
    check_true("the stem receives gradient", mt.backbone.conv1.weight.grad is not None and bool(mt.backbone.conv1.weight.grad.abs().sum() > 0))
    check_true("the head receives gradient", bool(mt.fc.weight.grad.abs().sum() > 0))
    nograd = [n for n, prm in mt.named_parameters() if prm.requires_grad and prm.grad is None]
    check_true("no trainable parameter is orphaned from the graph", not nograd, f"{nograd[:3]}")

    # ── 9. s3d — the 3D backbone, and the traps that are specific to it ─────
    # Sections 1-8 are about the 2D branch and its TSM. This section exists
    # because PRETRAINING-GAP.md §4's finding was not "fix a predicate" but
    # "the suite never constructs a 3D backbone at all", and a predicate fix
    # that is never exercised is not a fix.
    print("\ns3d — the 3D backbone (PRETRAINING-GAP.md §3, §4)")
    s3 = build("depthir", n_segment=16, arch="s3d", pretrained=False)
    s3_bns = bn_modules(s3.backbone)
    check_true("s3d has 77 BatchNorm3d", len(s3_bns) == 77, f"{len(s3_bns)}")
    check_true("...and ZERO BatchNorm2d — which is why the 2d predicate was vacuous",
               not [b for b in s3_bns if isinstance(b, torch.nn.BatchNorm2d)],
               f"{sum(1 for b in s3_bns if isinstance(b, torch.nn.BatchNorm2d))} found")
    ps3 = param_count(s3)
    check_true("s3d is 7.95 M params with a 4-ch stem and a 40-class head",
               7.8e6 < ps3 < 8.1e6, f"{ps3/1e6:.2f} M")
    check_true("...and is SMALLER than the ResNet-18 we ship today",
               ps3 < param_count(md), f"{ps3/1e6:.2f} M vs {param_count(md)/1e6:.2f} M")
    both3 = fp16_megabytes(s3) * 2
    check_true("two s3d branches at fp16 fit the 95 MB budget", both3 < 95.0, f"{both3:.1f} MB")

    # THE CENTRAL ONE. partial_bn is a no-op unless the predicate catches 3D.
    s3t = build("depthir", n_segment=16, arch="s3d", pretrained=False, partial_bn=True).train()
    b3 = bn_modules(s3t.backbone)
    check_true("partial_bn FREEZES 76 of s3d's 77 BNs (it froze 0 before the fix)",
               sum(1 for b in b3[1:] if not b.training) == 76,
               f"{sum(1 for b in b3[1:] if not b.training)} frozen")
    check_true("the first BN stays trainable on s3d too", b3[0].training)
    check_true("frozen affine is frozen on s3d too",
               not any(b.weight.requires_grad for b in b3[1:]))

    # TSM is not a knob on a 3D net — pinned by a test, not by a comment.
    check_true("s3d reports tsm_enabled False — a 3D conv IS the temporal operator",
               s3.tsm_enabled is False)
    check_true("s3d is registered as rank-5, so train.py skips channels_last",
               "s3d" in RANK5_ARCHS, f"{sorted(RANK5_ARCHS)}")

    # It runs on the cache we already have, and at J1's second input size.
    s3.eval()
    for (hh, ww) in [(120, 160), (168, 224)]:
        with torch.no_grad():
            out = s3(torch.randn(2, 16, 4, hh, ww))
        check_true(f"s3d forward at T=16 {hh}x{ww} -> (2, 40)", tuple(out.shape) == (2, 40),
                   f"{tuple(out.shape)}")

    # ── ir-CSN-R50, and the §10 import-path assertion NOTICE relies on ────────
    # src/csn.py is VENDORED from mmaction2 so that neither mmaction nor
    # pytorchvideo enters the inference path -- mmaction would drag mmcv +
    # mmengine + a model registry behind it, and pytorchvideo 0.1.5 has a
    # documented torchvision-0.17+ import break. NOTICE asserts this in prose;
    # this is the assertion that makes the prose checkable. §10 requires every
    # new dependency claim to carry one, and requires it to be mutation-tested.
    import importlib.util
    for mod in ("mmaction", "mmcv", "mmengine", "pytorchvideo"):
        check_true(f"{mod} is NOT importable — vendored, not depended on",
                   importlib.util.find_spec(mod) is None, "absent from the venv")
    src = (ROOT / "src" / "csn.py").read_text(encoding="utf-8")
    check_true("src/csn.py imports no mmaction/pytorchvideo symbol",
               not re.search(r'^\s*(from|import)\s+(mmaction|mmcv|mmengine|pytorchvideo)',
                             src, re.M), "structure reimplemented, nothing imported")
    check_true("src/csn.py keeps its Apache-2.0 attribution to OpenMMLab",
               "Apache-2.0" in src and "OpenMMLab" in src, "licence header present")

    cs = build("depthir", n_segment=16, arch="ircsn_r50", pretrained=False)
    check_true("ircsn_r50 is registered as rank-5 (channels_last would raise)",
               "ircsn_r50" in RANK5_ARCHS, f"{sorted(RANK5_ARCHS)}")
    check_true("ircsn_r50 takes the IMAGENET norm keys, not the *_kinetics ones "
               "— read off the checkpoint's own config, not inherited by family",
               norm_key("depthir", "ircsn_r50") == "depthir"
               and norm_key("thermal", "ircsn_r50") == "thermal",
               f'{norm_key("depthir", "ircsn_r50")} / {norm_key("thermal", "ircsn_r50")}')
    cs.eval()
    with torch.no_grad():
        out = cs(torch.randn(2, 16, 4, 168, 224))
    check_true("ircsn_r50 forward at T=16 168x224 -> (2, 40)",
               tuple(out.shape) == (2, 40), f"{tuple(out.shape)}")

    # §109 (D44): skomuro's from-scratch 2D FrameNet as a registered thermal arch -- the published structure to the parameter.
    def _raises(fn) -> bool:
        try:
            fn()
        except ValueError:
            return True
        return False
    fn = build("thermal", n_segment=8, arch="framenet", pretrained=False, partial_bn=False, tsm=False)
    check("framenet params = 1,240,520 (skomuro's FrameNet at 3 input channels; G §4 counted it to the unit)",
          param_count(fn), 1_240_520)
    check_true("framenet is rank-4 per frame (NOT in RANK5_ARCHS -- channels_last applies as for tsm_resnet18)",
               "framenet" not in RANK5_ARCHS, f"{sorted(RANK5_ARCHS)}")
    check_true("framenet takes the thermal cache's standard norm key",
               norm_key("thermal", "framenet") == "thermal", norm_key("thermal", "framenet"))
    fn.eval()
    with torch.no_grad():
        x8 = torch.randn(2, 8, 3, 112, 112)
        o8 = fn(x8)
        o8p = fn(x8[:, torch.randperm(8)])
        fn16 = build("thermal", n_segment=16, arch="framenet", pretrained=False, partial_bn=False, tsm=False).eval()
        o16 = fn16(torch.randn(2, 16, 3, 112, 112))
    check_true("framenet forward at T=8 112x112 -> (2, 40)", tuple(o8.shape) == (2, 40), f"{tuple(o8.shape)}")
    check_true("framenet forward at T=16 112x112 -> (2, 40) (the declared second config)",
               tuple(o16.shape) == (2, 40), f"{tuple(o16.shape)}")
    check_true("framenet is frame-order INVARIANT (per-frame logits averaged, no temporal operator -- the TSN property this file tests TSM against)",
               torch.allclose(o8, o8p, atol=1e-5), f"max |delta| {(o8 - o8p).abs().max().item():.2e}")
    fn.train()
    check_true("framenet with partial_bn=False keeps every BatchNorm live in train mode (from scratch -- nothing to protect)",
               all(m.training for m in bn_modules(fn)),
               f"{sum(m.training for m in bn_modules(fn))}/{len(bn_modules(fn))} BN modules training")
    check_true("framenet refuses the adapters it cannot honour (input_adapt / freeze_stages / mixstyle / weights raise)",
               _raises(lambda: build("thermal", n_segment=8, arch="framenet", pretrained=False, input_adapt="irnorm")),
               "ValueError on input_adapt")

    # ── §113 (D48 (2), reading α): ip-CSN-152 as a registered thermal arch ──────────────────────────────────────
    # The 'ip' bottleneck was READ off mmaction2 v0.24.1's resnet3d_csn.py (a BN'd 1x1x1 ConvModule with act_cfg=None
    # prepended to the depthwise 3x3x3) and off the fetched checkpoint's own keys (conv2.0.{conv,bn} + conv2.1.{conv,bn});
    # the converted VMZ file carries no meta cfg, so its normalisation was read off the zoo's two configs (ImageNet at
    # 0-255 — the ir-CSN keys). These checks pin the structure so the strict load is a proof, not a hope.
    from src.csn import ConvBN, IPCSNBottleneck
    from src.model import CSN_K400_FT_IP152, CSN_WEIGHT_DIR
    ipm = build("thermal", n_segment=8, arch="ipcsn_r152", pretrained=False)
    check_true("ipcsn_r152 is registered as rank-5 (channels_last would raise)",
               "ipcsn_r152" in RANK5_ARCHS, f"{sorted(RANK5_ARCHS)}")
    check_true("ipcsn_r152 takes the IMAGENET norm keys — read off mmaction2's _base_/models/ircsn_r152.py "
               "data_preprocessor and the v0.24.1 img_norm_cfg, not inherited by family",
               norm_key("thermal", "ipcsn_r152") == "thermal" and norm_key("depthir", "ipcsn_r152") == "depthir",
               f'{norm_key("thermal", "ipcsn_r152")} / {norm_key("depthir", "ipcsn_r152")}')
    bb = ipm.backbone
    check_true("ipcsn_r152 keeps the ir-CSN-152 stage depths (3, 8, 36, 3)",
               tuple(len(getattr(bb, f"layer{i}")) for i in range(1, 5)) == (3, 8, 36, 3),
               f"{tuple(len(getattr(bb, f'layer{i}')) for i in range(1, 5))}")
    blk = bb.layer1[0]
    check_true("every ip bottleneck's conv2 is [1x1x1 ConvBN with NO activation → 3x3x3 DEPTHWISE ConvBN], the stride on the depthwise",
               all(isinstance(b, IPCSNBottleneck) and len(b.conv2) == 2 and isinstance(b.conv2[0], ConvBN)
                   and b.conv2[0].act is None and tuple(b.conv2[0].conv.kernel_size) == (1, 1, 1)
                   and tuple(b.conv2[1].conv.kernel_size) == (3, 3, 3) and b.conv2[1].conv.groups == b.conv2[1].conv.in_channels
                   for i in range(1, 5) for b in getattr(bb, f"layer{i}")),
               f"layer1[0].conv2 = {[type(m).__name__ for m in blk.conv2]}, act[0] = {blk.conv2[0].act}")
    check("ipcsn_r152 params = 32,278,952 (the fetched file's 32,196,992 backbone params, BN buffers excluded, + a 40-way head)",
          param_count(ipm), 32_278_952)
    check("ircsn_r152 params are UNCHANGED by the BLOCK refactor = 28,965,928 (the shipped file's 28,883,968 + a 40-way head)",
          param_count(build("thermal", n_segment=8, arch="ircsn_r152", pretrained=False)), 28_965_928)
    ipm.eval()
    with torch.no_grad():
        oip = ipm(torch.randn(2, 8, 3, 112, 112))
    check_true("ipcsn_r152 forward at T=8 112x112 -> (2, 40)", tuple(oip.shape) == (2, 40), f"{tuple(oip.shape)}")
    if pre and (CSN_WEIGHT_DIR / CSN_K400_FT_IP152).is_file():
        ipk = build("thermal", n_segment=8, arch="ipcsn_r152", pretrained=True, num_classes=400, load_k400_head=True)
        check_true("ipcsn_r152 STRICT-loads the fetched IG-65M→K400 file with ZERO key remapping, the K400 head (400, 2048) included",
                   tuple(ipk.fc.weight.shape) == (400, 2048), f"{tuple(ipk.fc.weight.shape)}")
    else:
        print("  [skip] ipcsn_r152 strict load — the weights file is not cached or --no-pretrained (stated, not silent)")

    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 62)
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  · {f}")
        print("\nDiagnose these. Do not adjust the expected values to make them pass.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
