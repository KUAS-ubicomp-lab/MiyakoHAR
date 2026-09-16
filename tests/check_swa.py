"""Verify SWA -- above all, that the average is an average and that recomputing
BN statistics does not destroy the frozen BatchNorms partial_bn preserves.

THIS SUITE RUNS ONCE PER REGISTERED ARCHITECTURE, AND THAT IS NOT COSMETIC.
Until 2026-08-15 it did two things that made it structurally unable to fail on a
3D backbone, and PRETRAINING-GAP.md §4 found both:

  · bns() predicated on nn.BatchNorm2d. s3d has 77 BatchNorm3d and ZERO
    BatchNorm2d, so bns() returned [] and all 24 checks passed VACUOUSLY --
    including randomise_bn_buffers, which randomises nothing, which disables the
    tautology guard the docstring below says the file depends on.
  · every build() call named the 2D model. So fixing the predicate was necessary
    and NOT sufficient: the suite would still never have constructed a 3D
    backbone, and would have reported a confident green over a code path it had
    never executed.

Both are fixed by iterating over model._ARCH_BUILDERS rather than hardcoding
anything, so an architecture added to the registry is exercised here the day it
lands rather than the day someone remembers.

WHY THIS FILE EXISTS.

SWA has two failure modes here and neither one crashes.

  1. The "average" is really the last iterate. If update_parameters is called
     from the wrong place, or n_averaged never advances, the checkpoint is just
     the final weights wearing an SWA label. It still scores, it still looks
     reasonable, and every A/B measured against it is then measured against a
     baseline that is not the one the recipe specifies.

  2. update_bn resets the frozen BatchNorms. torch.optim.swa_utils.update_bn
     calls reset_running_stats() on EVERY BatchNorm and only then calls
     model.train(). Our train() re-applies partial_bn, which puts all but the
     first back into eval mode -- so they never recompute, and keep the (0, 1)
     that reset left behind. The pretrained statistics partial_bn exists to
     preserve are gone, silently, and the model still runs. (19 of 20 on
     ResNet-18; 76 of 77 on s3d, where there is 3.85x as much to lose.)

     Section 3's control arm runs the stock function on purpose and asserts it
     DOES corrupt them, so this file proves the trap is real rather than
     asserting that our replacement is merely present.

TAUTOLOGY GUARD. A fresh BatchNorm initialises running stats to exactly
(mean 0, var 1) -- which is also what reset_running_stats() produces. So on an
untrained model "the frozen stats were not reset" passes whether or not they
were reset, and proves nothing. Section 3 therefore randomises every BN buffer
to non-trivial values FIRST. That is the difference between a test and a
decoration (HANDOFF.md 9.7).

No corpus, no cache, no network -- synthetic tensors only, so this runs on
either machine. Small spatial size on purpose: BN statistics do not care about
the geometry, and 64x64 keeps the suite under a minute on CPU.

Standing rule: diagnose any failure. Never adjust the expected value.

Usage:
    python tests/check_swa.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import update_bn as stock_update_bn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import _ARCH_BUILDERS, bn_modules, build  # noqa: E402
from src.swad import EpochSums, load_average, swad_window, window_average  # noqa: E402
from src.train import recompute_bn_stats, swa_start_iter  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []

B, T, C, H, W = 2, 8, 3, 64, 64

# The head parameter each backbone exposes -- the one section 1 fills with known
# values to prove the average is an average. Named per arch rather than found by
# heuristic: "the last 2-D parameter" would silently pick a different tensor if a
# backbone changed, and section 1 would then be averaging something else while
# still passing.
HEAD_PARAM = {"tsm_resnet18": "fc.weight", "s3d": "backbone.classifier.1.weight",
              "mc3_18": "backbone.fc.1.weight",
              # ircsn_r50 was registered in _ARCH_BUILDERS (m118) without an
              # entry here, so per_arch() silently skipped THE ARCH WE SHIP and
              # only the "names its head parameter" assertion caught it. Its head
              # is a top-level fc, not backbone.fc -- same name as tsm_resnet18's,
              # a different module.
              "ircsn_r50": "fc.weight",
              # P2's skeleton branch (D8's build): top-level fc, like the CSN.
              "tcn_1d": "fc.weight",
              # §49's ircsn_r152 (CSNBranch(depth=152) -- the same top-level fc
              # as ircsn_r50) and §61's tsm_r50_ssv2 (TSMResNet50: top-level fc,
              # model.py:306) were registered in _ARCH_BUILDERS without an entry
              # here; the assertion below went RED at the Phase-3 boot
              # (2026-08-31, m219) -- the gap it exists to catch, the second time.
              # Entries ADDED (the expected values are untouched); each arch now
              # runs sections 1/3/4/5 like the others.
              "ircsn_r152": "fc.weight",
              # §109's framenet (dff3791, m340) was registered in _ARCH_BUILDERS without an entry here — the
              # assertion below went RED at the Phase-8 boot (2026-09-08, m353), the THIRD time it catches the gap it
              # exists to catch. Entry ADDED (FrameNet's head is a top-level fc, model.py:1080); the expected values
              # are untouched. §113's ipcsn_r152 (CSNBranch(depth=152, ip=True), the same top-level fc) registered
              # WITH its entry in the same commit.
              "framenet": "fc.weight",
              "ipcsn_r152": "fc.weight",
              "tsm_r50_ssv2": "fc.weight",
              # §66's IMU archs (D23): top-level fc on all three; from scratch,
              # every BN live (the tcn_1d cases); xf_imu has NO BatchNorm at all.
              "cnn1d_imu": "fc.weight", "dtcn_imu": "fc.weight", "xf_imu": "fc.weight",
              # §68's VideoMAE V2 ViT-S branch: top-level fc; an IMAGE arch with NO BatchNorm (LayerNorm).
              "vmae2_vits": "fc.weight", "vmae2_vitb": "fc.weight",
              # §90 (D27): the truncated ViT-B members (VMAEBranch(variant="b", depth=11 / 10)): the same top-level fc.
              "vmae2_vitb11": "fc.weight", "vmae2_vitb10": "fc.weight"}

# Rank-3, from-scratch archs: no partial_bn, nothing frozen — sections 3–5 (the
# partial-BN preservation properties) do not apply and are skipped LOUDLY.
NON_IMAGE = {"tcn_1d", "cnn1d_imu", "dtcn_imu", "xf_imu"}
# Archs with no BatchNorm anywhere (LayerNorm transformers; §68's ViT will join):
# section 1 (the average IS an average) still runs; recompute_bn_stats must be a
# no-op that returns 0 on them — m213 (i)'s live-BN predicate re-verified on a
# module with no `.backbone` and no BN.
NO_BN = {"xf_imu", "vmae2_vits", "vmae2_vitb", "vmae2_vitb11", "vmae2_vitb10"}


def check(desc: str, got, expected) -> bool:
    ok = got == expected
    (PASS if ok else FAIL).append(desc)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {desc}: got {got!r}, expected {expected!r}")
    return ok


def check_true(desc: str, cond: bool, detail: str = "") -> bool:
    (PASS if cond else FAIL).append(desc)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {desc}{(' — ' + detail) if detail else ''}")
    return bool(cond)


def bns(model: nn.Module) -> list[nn.Module]:
    """Every BatchNorm on the backbone, 1d/2d/3d alike.

    THE PREDICATE USED TO BE nn.BatchNorm2d AND THAT IS THE BUG THIS FILE
    NOW EXISTS TO HAVE CAUGHT. On s3d it matched nothing, returned [], and every
    loop below iterated zero times while reporting [ok]. Delegated to
    model.bn_modules so the test and the code under test cannot drift apart.
    tcn_1d has no .backbone — its BNs live on the module itself."""
    return bn_modules(getattr(model, "backbone", model))


def randomise_bn_buffers(model: nn.Module, gen: torch.Generator) -> None:
    """Give every BN non-trivial running stats, so a reset to (0, 1) is visible.

    Without this, section 3's central assertion is a tautology: an untrained
    BN's running stats ARE (0, 1), which is exactly what a reset produces.
    """
    for m in bns(model):
        m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=gen) * 0.5 + 1.0)
        m.running_var.copy_(torch.rand(m.running_var.shape, generator=gen) * 2.0 + 0.5)
        m.num_batches_tracked.fill_(4321)


def synthetic_batches(n: int, device, gen: torch.Generator) -> list[dict]:
    """Batches in collate()'s shape -- a dict, which is itself part of the trap."""
    return [{"x": torch.randn(B, T, C, H, W, generator=gen).to(device),
             "y": torch.randint(0, 40, (B,), generator=gen).to(device)} for _ in range(n)]


def arch_branch(arch: str) -> str:
    """The branch each arch actually trains on — tcn_1d exists only for pose, the *_imu archs for imu."""
    if arch == "tcn_1d":
        return "skeleton"
    if arch.endswith("_imu"):
        return "imu"
    return "thermal"


def per_arch(arch: str, device, amp_dtype) -> None:
    """Sections 1, 3, 4 and 5 — everything that constructs a network.

    Section 2 is pure arithmetic on swa_start_iter and is arch-independent, so it
    stays in main() and runs once."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(20260813)
    branch = arch_branch(arch)
    n_bn = len(bns(build(branch, n_segment=T, arch=arch, pretrained=False)))
    n_frozen = 0 if (arch in NON_IMAGE or arch in NO_BN) else n_bn - 1
    print("\n" + "─" * 62)
    print(f"  ARCH: {arch}  ({n_bn} BatchNorms, {n_frozen} frozen by partial_bn"
          + (" — none: trains from scratch, all BNs live)" if arch in NON_IMAGE else ")"))
    print("─" * 62)

    if arch in NO_BN:
        # Declared BN-free: the vacuity guard below would misreport this as a failure. Assert the
        # declaration BOTH ways (a BN appearing in a "no-BN" arch is a wrong declaration) and that the
        # BN refit is a no-op on it — then only section 1 applies.
        check_true(f"{arch}: declared BN-free and actually has NO BatchNorm", n_bn == 0, f"{n_bn} found")
        m0 = build(branch, n_segment=T, arch=arch, pretrained=False).to(device)
        seen = recompute_bn_stats(m0, [{"x": torch.randn(B, T, 40).to(device)}], device, amp_dtype)
        check_true(f"{arch}: recompute_bn_stats is a no-op on a BN-free module (returns 0, consumes nothing)",
                   seen == 0, f"consumed {seen}")
    # A suite that reports [ok] over an empty list is the exact failure mode
    # this parameterisation exists to close. Assert there is something to check
    # BEFORE checking it.
    elif not check_true(f"{arch}: the backbone actually exposes BatchNorms to test",
                        n_bn > 0, f"{n_bn} found — if 0, every assertion below is vacuous"):
        # Return rather than continue: the sections below index bns(...)[0] and
        # would raise IndexError, which reports the failure worse than this line
        # already has. The point is made; keep the summary readable.
        print("  [skip] remaining sections for this arch — there is nothing to assert over")
        return

    # ── 1. The average is an average, not the last iterate ────────────────
    print("\nSWA averages — the check that catches a 'mean' that is really a copy")
    m = build(branch, n_segment=T, arch=arch, pretrained=False).to(device)
    swa = AveragedModel(m)
    probe = HEAD_PARAM[arch]
    values = [1.0, 2.0, 6.0, 11.0]  # mean 5.0, and the LAST value is not the mean
    for v in values:
        with torch.no_grad():
            dict(m.named_parameters())[probe].fill_(v)
        swa.update_parameters(m)
    got = float(dict(swa.module.named_parameters())[probe].detach().flatten()[0])
    check("n_averaged counts every snapshot", int(swa.n_averaged), len(values))
    check_true("SWA weight is the arithmetic mean of the snapshots",
               abs(got - 5.0) < 1e-5, f"{got:.6f} vs mean {sum(values)/len(values)}")
    # The control: a "copy the latest" bug reproduces every other property of
    # this test and would pass without this line.
    check_true("SWA weight is NOT simply the final snapshot (control)",
               abs(got - values[-1]) > 1.0, f"{got:.4f} vs last {values[-1]}")

    # ── 3. Recomputing BN respects partial_bn ─────────────────────────────
    print("\nrecompute_bn_stats vs partial_bn — the silent-corruption check")
    if arch in NON_IMAGE or arch in NO_BN:
        # Not vacuous-pass, not silent-skip: the partial-BN preservation
        # sections assert a property tcn_1d deliberately does not have (D10:
        # from scratch, every BN trainable — there is nothing frozen to
        # preserve). Its BN refit is exercised live by train.py's
        # recompute_bn_stats on every skeleton run and printed in the run log.
        # The §66 IMU archs are the same class (xf_imu has no BN at all).
        print(f"  [skip] sections 3-5 for {arch} — partial_bn does not apply "
              "(from-scratch arch, all BNs live or none); stated, not silent")
        return

    mp = build("thermal", n_segment=T, arch=arch, pretrained=False, partial_bn=True).to(device)
    with torch.no_grad():
        randomise_bn_buffers(mp, gen)          # kills the (0,1) tautology
    mp = mp.to(device)
    before = [(b.running_mean.detach().clone(), b.running_var.detach().clone()) for b in bns(mp)]
    check_true("the fixture starts with NON-trivial frozen stats",
               bool((before[5][0].abs().sum() > 0) and not torch.allclose(before[5][1], torch.ones_like(before[5][1]))),
               "otherwise every assertion below is a tautology")

    momentum_before = bns(mp)[0].momentum
    batches = synthetic_batches(4, device, gen)
    seen = recompute_bn_stats(mp, batches, device, amp_dtype)
    after = [(b.running_mean.detach().clone(), b.running_var.detach().clone()) for b in bns(mp)]

    check("it consumed every batch", seen, len(batches))
    check_true("BN[0] — the one training updates — was RECOMPUTED",
               not torch.allclose(before[0][0], after[0][0]),
               f"max|Δmean| = {(before[0][0] - after[0][0]).abs().max():.4f}")
    check_true("BN[0]'s num_batches_tracked restarted from the reset",
               int(bns(mp)[0].num_batches_tracked) == len(batches),
               f"{int(bns(mp)[0].num_batches_tracked)}")
    frozen_changed = [i for i in range(1, len(before))
                      if not (torch.equal(before[i][0], after[i][0]) and torch.equal(before[i][1], after[i][1]))]
    check_true(f"all {n_frozen} frozen BNs are BIT-IDENTICAL afterwards",
               not frozen_changed, f"changed: {frozen_changed[:5]}")
    # THE EXPECTED VALUE IS THE ONE THE MODEL CAME WITH, NOT THE LITERAL 0.1.
    # This asserted `== 0.1` until 2026-08-15 and went red the first time a
    # non-ResNet backbone ran it: torchvision's S3D builds its BatchNorm3d with
    # momentum=0.001, eps=0.001 (its own values, from the TF original), and
    # recompute_bn_stats correctly restores whatever it found. The code was
    # right and the expectation was a ResNet-18 constant.
    #
    # AND 0.1 IS ALSO TORCH'S DEFAULT, which is what made the old form weak
    # even on ResNet-18: a bug that reset momentum to the default instead of
    # restoring the saved value would have passed. Capturing it beforehand tests
    # the property the function actually promises -- restoration -- rather than a
    # number that coincided with it. (Standing rule 5: re-derive the standard,
    # never move the expected value to fit.)
    check_true("BN[0]'s momentum was restored to what the model came with",
               bns(mp)[0].momentum == momentum_before,
               f"{bns(mp)[0].momentum} vs {momentum_before} before the call")
    check_true("running stats stayed finite", all(bool(torch.isfinite(a).all() and torch.isfinite(v).all())
                                                  for a, v in after))

    # THE CONTROL ARM. Stock update_bn on the same fixture must corrupt the
    # frozen BNs -- if it does not, the trap this function exists for is not
    # real and recompute_bn_stats is pointless indirection.
    mc = build("thermal", n_segment=T, arch=arch, pretrained=False, partial_bn=True).to(device)
    with torch.no_grad():
        randomise_bn_buffers(mc, torch.Generator().manual_seed(20260813))
    mc = mc.to(device)
    c_before = [(b.running_mean.detach().clone(), b.running_var.detach().clone()) for b in bns(mc)]
    stock_update_bn([b["x"] for b in batches], mc, device=device)   # tensors, as it demands
    c_after = [(b.running_mean.detach().clone(), b.running_var.detach().clone()) for b in bns(mc)]
    stock_wrecked = [i for i in range(1, len(c_before))
                     if not torch.equal(c_before[i][0], c_after[i][0])]
    check_true("CONTROL: stock torch update_bn DOES wreck the frozen BNs",
               len(stock_wrecked) == n_frozen, f"{len(stock_wrecked)} of {n_frozen} corrupted")
    check_true("CONTROL: and it leaves them at the (0, 1) reset, never recomputed",
               bool(torch.allclose(c_after[5][0], torch.zeros_like(c_after[5][0]))
                    and torch.allclose(c_after[5][1], torch.ones_like(c_after[5][1]))),
               "so the corruption is a silent reset, not a wrong recomputation")

    # ── 4. The end-to-end shape: averaging changes the weights ──────────────
    print("\nAveraging over a real (tiny) optimisation")
    # The fixture carries the recipe's OWN stabilisers -- lr well below 0.05 and
    # clip_grad_norm 20.0 -- because without them this loop diverges: measured
    # loss 3.69 -> 248 -> 1.3e9 -> NaN by step 3, gradient norm 8.4e9, on a
    # random-init network fed pure noise. SWA then averages NaN faithfully and
    # the delta below reports `nan`, which is not a result in either direction.
    # Hence the finiteness gate: a future divergence must NAME itself rather
    # than hide inside a comparison that silently cannot pass.
    mt = build("thermal", n_segment=T, arch=arch, pretrained=False).to(device)
    swa2 = AveragedModel(mt)
    opt = torch.optim.SGD(mt.parameters(), lr=0.005, momentum=0.9)
    mt.train()
    start = swa_start_iter(8, 0.5)
    for it in range(8):
        bt = synthetic_batches(1, device, gen)[0]
        loss = nn.functional.cross_entropy(mt(bt["x"]), bt["y"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mt.parameters(), 20.0)
        opt.step()
        if it >= start:
            swa2.update_parameters(mt)
    check("snapshots taken = total - start", int(swa2.n_averaged), 8 - start)
    check_true("the fixture optimisation stayed finite",
               all(bool(torch.isfinite(p).all()) for p in mt.parameters()),
               "if this fails the delta below is meaningless — fix the fixture, not the assertion")
    d = max(float((a - b).abs().max())
            for a, b in zip(swa2.module.parameters(), mt.parameters()))
    check_true("SWA weights DIFFER from the final iterate", d > 1e-7, f"max|Δ| = {d:.3e}")

    # torch's documented use_buffers=False behaviour: buffers are copied from
    # the live model, never averaged. That is the whole reason update_bn is
    # mandatory, so it is asserted rather than assumed -- if a future torch
    # changes it, this line is what tells us.
    same_buf = all(torch.equal(a, b) for a, b in zip(swa2.module.buffers(), mt.buffers()))
    check_true("SWA buffers track the live model, so BN must be recomputed", same_buf)

    # ── 5. The zero-snapshot guard ──────────────────────────────────────────
    print("\nThe guard against reporting an SWA number that was never averaged")
    swa3 = AveragedModel(build("thermal", n_segment=T, arch=arch, pretrained=False))
    check("a never-updated AveragedModel reports n_averaged 0", int(swa3.n_averaged), 0)
    check_true("...which is the condition train.py refuses to write swa.pt on",
               int(swa3.n_averaged) == 0)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    # ── 2. Snapshot window arithmetic — arch-independent, so it runs once ────
    print("\nstart_frac -> the iteration averaging begins at")
    check("40 epochs x 121 steps at 0.75", swa_start_iter(4840, 0.75), 3630)
    check_true("...which averages the last quarter of the schedule",
               4840 - swa_start_iter(4840, 0.75) == 1210,
               f"{4840 - swa_start_iter(4840, 0.75)} snapshots")
    check("start_frac 0.0 averages the whole run", swa_start_iter(4840, 0.0), 0)
    check("start_frac 1.0 collects nothing", swa_start_iter(4840, 1.0), 4840)
    for bad in (-0.1, 1.5):
        try:
            swa_start_iter(100, bad)
            check_true(f"refuses start_frac={bad}", False, "accepted it")
        except ValueError:
            check_true(f"refuses start_frac={bad}", True)

    # ── 6. SWAD (T1, QUEUE §24) — the window rule and the epoch-sum average ──
    # MUTATION-TESTED ON LANDING (m140's class): an off-by-one in the rule's
    # convergence test must turn the U-shape case RED. The run that proved it is
    # recorded in the ledger row, not assumed from a green first pass.
    print("\nSWAD window rule — loss-valley at epoch granularity (n_converge 3, n_tolerance 6, ratio 0.3)")
    u = [5, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ts, te, conv, thr = swad_window(u)
    check("U-shape: the window opens at the valley floor (epoch 3)", ts, 3)
    check_true("U-shape: threshold = mean(2,3,4) × 1.3 = 3.9", thr is not None and abs(thr - 3.9) < 1e-9, f"{thr}")
    check("U-shape: it closes before six epochs all above 3.9 (epoch 4)", te, 4)
    check("U-shape: reported converged", conv, True)
    ts, te, conv, thr = swad_window([1.0] * 10)
    check("flat curve: opens at epoch 0", ts, 0)
    check("flat curve: never dies — closes at the last epoch", te, 9)
    ts, te, conv, thr = swad_window(list(range(10, 0, -1)))
    check("still falling at the end: NOT converged (the flagged fallback)", conv, False)
    check("...and the fallback is the last n_converge epochs", (ts, te), (7, 9))
    ts, te, conv, thr = swad_window([3, 2, 1, 1.1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
    check("valley then plateau under threshold: opens at 2, runs to the end", (ts, te), (2, 9))
    ts, te, conv, thr = swad_window([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    check("rising from the start: opens at 0 and dies quickly", ts, 0)
    check_true("...closing before the first six-above-threshold window", te < 9, f"te={te}")

    print("\nSWAD epoch sums — the window average is the arithmetic mean of the iterates")
    import tempfile
    lin = nn.Linear(3, 2)
    with tempfile.TemporaryDirectory() as td:
        sums = EpochSums(lin, Path(td))
        vals = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]          # 3 epochs × 2 iterations
        for e, vs in enumerate(vals):
            for v in vs:
                with torch.no_grad():
                    lin.weight.fill_(v); lin.bias.fill_(-v)
                sums.add()
            sums.flush(e)
        check("every epoch flushed with its iteration count", sums.flushed, [(0, 2), (1, 2), (2, 2)])
        avg, n = window_average(Path(td), 1, 2)
        check("window [1,2] counts 4 iterates", n, 4)
        check_true("window [1,2] mean = (3+4+5+6)/4 = 4.5",
                   abs(float(avg["weight"].flatten()[0]) - 4.5) < 1e-6, f"{float(avg['weight'].flatten()[0])}")
        check_true("...and the bias averages independently (−4.5)",
                   abs(float(avg["bias"][0]) + 4.5) < 1e-6, f"{float(avg['bias'][0])}")
        avg0, n0 = window_average(Path(td), 0, 0)
        check_true("window [0,0] mean = 1.5 (control: not the last iterate)",
                   abs(float(avg0["weight"].flatten()[0]) - 1.5) < 1e-6)
        load_average(lin, avg)
        check_true("load_average writes the mean into the model's parameters",
                   abs(float(lin.weight.flatten()[0]) - 4.5) < 1e-6)
        try:
            load_average(lin, {"weight": avg["weight"]})
            check_true("load_average refuses a partial average", False, "accepted it")
        except KeyError:
            check_true("load_average refuses a partial average", True)

    # EVERY REGISTERED ARCHITECTURE, not a hardcoded list. An arch added to
    # model._ARCH_BUILDERS is exercised here automatically; one that is added
    # without a HEAD_PARAM entry fails loudly on the next line rather than being
    # quietly skipped.
    missing = sorted(set(_ARCH_BUILDERS) - set(HEAD_PARAM))
    check_true("every registered arch names its head parameter", not missing,
               f"no HEAD_PARAM for {missing}" if missing else f"{sorted(_ARCH_BUILDERS)}")
    for arch in sorted(_ARCH_BUILDERS):
        if arch in HEAD_PARAM:
            per_arch(arch, device, amp_dtype)

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
