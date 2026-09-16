"""Verify that a checkpoint says what it IS, and that the loader believes it.

WHY THIS FILE EXISTS (§J-DQ defect (c), 2026-08-15).

Until this landed, a checkpoint recorded `branch, fold, T, epoch, val_acc,
in_channels` and nothing about how the weights were built. predict.py inferred
the architecture by being the only one there was, and the normalisation from the
branch name. Both inferences were correct and both were about to stop being so:
J1 puts two architectures and two pretraining corpora in play at once, and
predict.py is forbidden from reading YAML, so the checkpoint file is the ONLY
channel through which the loader can learn how to feed the weights.

THE ASYMMETRY THIS FILE IS BUILT AROUND:

  · a wrong ARCH raises inside load_state_dict. It is self-announcing.
  · a wrong NORM raises NOTHING. The tensors are the right shape, the forward
    pass succeeds, the CSV is well-formed, and the model has simply been fed a
    distribution it never saw. In a mixed ensemble that is near-certain, and the
    only symptom is accuracy that is quietly worse than it should be.

So `norm` is persisted even though it is currently derivable from `branch`. The
derivable case is exactly the case where nobody notices it stopped being.

WHAT IS ASSERTED, AND WHAT IS NOT. The reader half is asserted by execution
here. The writer half -- that train.py stamps both keys at both save sites -- is
asserted structurally below and end-to-end by every real run: section 4 walks
runs/ and requires that any checkpoint carrying `arch` carries `norm` too.

Section 4 needs runs/, which is gitignored. It reports how many checkpoints it
saw; zero is reported as a SKIPPED section, never as a pass.

Standing rule: diagnose any failure. Never adjust the expected value.

Usage:
    python tests/check_ckpt_meta.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.dataset import NORM  # noqa: E402
from src.model import (  # noqa: E402
    _ARCH_BUILDERS,
    _ARCH_NORM,
    BRANCH_CHANNELS,
    DEFAULT_ARCH,
    build,
    norm_key,
)

PASS: list[str] = []
FAIL: list[str] = []
SKIPPED: list[str] = []


def check(desc: str, got, expected) -> bool:
    ok = got == expected
    (PASS if ok else FAIL).append(desc)
    print(f"  [{'ok  ' if ok else 'FAIL'}] {desc}: got {got!r}, expected {expected!r}")
    return ok


def check_true(desc: str, cond: bool, detail: str = "") -> bool:
    (PASS if cond else FAIL).append(desc)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {desc}{(' — ' + detail) if detail else ''}")
    return bool(cond)


def check_raises(desc: str, fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return check_true(desc, True, "raised as required")
    except Exception as e:  # noqa: BLE001
        return check_true(desc, False, f"raised {type(e).__name__}, expected {exc.__name__}")
    return check_true(desc, False, "did NOT raise — it resolved to something")


def main() -> int:
    print("=" * 62)
    print("  CHECKPOINT METADATA — arch and norm must be recorded and dispatched")
    print("=" * 62)

    # ── 1. the registry is internally consistent ────────────────────────────
    # This is the section that will fail the day `s3d` is added to one table and
    # not the other -- which is the single most likely way J1 produces a
    # confidently wrong number.
    print("\n1 · The arch registry agrees with itself")
    check_true("every buildable arch has a norm table",
               set(_ARCH_BUILDERS) <= set(_ARCH_NORM),
               f"missing: {sorted(set(_ARCH_BUILDERS) - set(_ARCH_NORM))}")
    check_true("every norm table names a buildable arch",
               set(_ARCH_NORM) <= set(_ARCH_BUILDERS),
               f"orphaned: {sorted(set(_ARCH_NORM) - set(_ARCH_BUILDERS))}")
    check_true("DEFAULT_ARCH is buildable", DEFAULT_ARCH in _ARCH_BUILDERS, DEFAULT_ARCH)
    # RE-DERIVED 2026-08-25, not relaxed. P2 (m151) registered `tcn_1d`, a POSE
    # arch whose only branch is `skeleton` and whose norm key `skeleton_raw` is
    # DELIBERATELY absent from dataset.NORM (model.py: an accidental image-
    # statistics lookup on pose data must KeyError loudly). The expectation
    # "every arch covers every branch" was written when every arch was an image
    # arch; it crashed this suite on `NORM['skeleton_raw']` from the day P2
    # landed, and nobody re-ran it (m151 lists the other three suites). The
    # property that was always meant: an arch covers every branch OF ITS
    # MODALITY, and names only norms that exist -- except the pose sentinel,
    # which must NOT exist. Both halves are asserted.
    # §66 (D23) generalised the pose case to every NON-IMAGE modality: arch → (its one branch, its
    # sentinel, its raw feature width). tcn_1d = pose; the three IMU archs = imu (40 = 5 sites × 8).
    NON_IMAGE = {"tcn_1d": ("skeleton", "skeleton_raw", 102),
                 "cnn1d_imu": ("imu", "imu_raw", 40), "dtcn_imu": ("imu", "imu_raw", 40), "xf_imu": ("imu", "imu_raw", 40)}
    NON_IMAGE_BRANCHES = {b for b, _s, _w in NON_IMAGE.values()}
    SENTINELS = {s for _b, s, _w in NON_IMAGE.values()}
    IMAGE_BRANCHES = set(BRANCH_CHANNELS) - NON_IMAGE_BRANCHES
    POSE_SENTINEL = "skeleton_raw"
    for a, table in sorted(_ARCH_NORM.items()):
        want = {NON_IMAGE[a][0]} if a in NON_IMAGE else IMAGE_BRANCHES
        check_true(f"{a}: covers every branch of its modality",
                   set(table) == want,
                   f"branches {sorted(set(want) ^ set(table))} differ"
                   if set(table) != want else f"{sorted(table)}")
        bad = sorted(v for v in table.values() if v not in NORM and v not in SENTINELS)
        check_true(f"{a}: every norm it names exists in dataset.NORM", not bad,
                   f"unknown: {bad}" if bad else f"{sorted(set(table.values()))}")
    for s in sorted(SENTINELS):
        check_true(f"the non-image sentinel {s!r} is NOT a NORM key (must KeyError, by design)", s not in NORM)
    # The channel count and the norm vector must agree, or normalisation
    # broadcasts against the wrong number of planes and raises at runtime.
    for a, table in sorted(_ARCH_NORM.items()):
        for br, key in sorted(table.items()):
            if key in SENTINELS:
                check_true(f"{a}/{br}: non-image features are used RAW ({BRANCH_CHANNELS[br]} dims, no NORM entry)",
                           BRANCH_CHANNELS[br] == NON_IMAGE[a][2] and key == NON_IMAGE[a][1] and key not in NORM)
                continue
            check(f"{a}/{br}: norm '{key}' has {BRANCH_CHANNELS[br]} channels",
                  len(NORM[key]["mean"]), BRANCH_CHANNELS[br])

    # ── 2. unknown values raise, they never resolve ─────────────────────────
    print("\n2 · An unrecognised arch or branch REFUSES, it does not default")
    check_raises("build() raises on an unknown arch", lambda: build("depthir", arch="nope"))
    check_raises("build() raises on an unknown branch", lambda: build("nope"))
    check_raises("norm_key() raises on an unknown arch", lambda: norm_key("depthir", "nope"))
    check_raises("norm_key() raises on an unknown branch", lambda: norm_key("nope"))
    check("norm_key is the same function train.py stamps with",
          norm_key("depthir", DEFAULT_ARCH), "depthir")
    check("...and it is branch-sensitive", norm_key("thermal", DEFAULT_ARCH), "thermal")

    # ── 3. the loader dispatches on the FILE, not on the branch name ────────
    # Constructed rather than trained: the property under test is what Ensemble
    # does with the keys, and a real run costs 12 minutes to assert the same thing.
    print("\n3 · predict.Ensemble reads arch and norm off the checkpoint")
    from predict import Ensemble  # noqa: PLC0415  (needs src/ on the path)

    m = build("depthir", n_segment=8, arch=DEFAULT_ARCH, pretrained=False)
    base = {"model": m.state_dict(), "branch": "depthir", "fold": 0, "T": 8,
            "epoch": 0, "val_acc": 0.0, "in_channels": 4}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        stamped = td / "stamped.pt"
        torch.save({**base, "arch": DEFAULT_ARCH, "norm": "depthir"}, stamped)
        e = Ensemble([stamped], device_pref="cpu")
        mem = e.by_branch["depthir"][0]
        check("a stamped checkpoint reports its own arch", mem["arch"], DEFAULT_ARCH)
        check("a stamped checkpoint reports its own norm", mem["norm"], "depthir")
        check("a stamped checkpoint is not flagged legacy", mem["legacy"], False)

        legacy = td / "legacy.pt"
        torch.save(dict(base), legacy)  # no arch, no norm — a pre-2026-08-15 file
        e = Ensemble([legacy], device_pref="cpu")
        mem = e.by_branch["depthir"][0]
        check("an unstamped checkpoint resolves to the legacy arch", mem["arch"], DEFAULT_ARCH)
        check("an unstamped checkpoint resolves to the legacy norm", mem["norm"], "depthir")
        check_true("...and SAYS it did so rather than resolving silently", mem["legacy"] is True)

        # The central assertion of this file. A norm that is present but
        # unrecognised must stop the run. Defaulting here is the silent-accuracy
        # failure the whole §J-DQ (c) fix exists to prevent.
        bad_norm = td / "badnorm.pt"
        torch.save({**base, "arch": DEFAULT_ARCH, "norm": "kinetics_typo"}, bad_norm)
        check_raises("an UNKNOWN norm raises rather than falling back",
                     lambda: Ensemble([bad_norm], device_pref="cpu"))

        bad_arch = td / "badarch.pt"
        torch.save({**base, "arch": "resnet_from_the_future", "norm": "depthir"}, bad_arch)
        check_raises("an UNKNOWN arch raises rather than falling back",
                     lambda: Ensemble([bad_arch], device_pref="cpu"))

        # ── §96 (a)/(b) (D29): the depth inverse-LUT stamp (a fold-TRAIN statistic, L7's second clause) travels through BOTH
        # packers as a fixed member key, is None where absent, and predict REFUSES a LUT-bearing member until the per-clip
        # ordinal decode path exists — the key can never be a silent passenger. Committed BEFORE any §96 (a)/(b) cell.
        from tools import package as P, package_int8 as Q  # noqa: PLC0415
        lut = torch.arange(254 * 3, dtype=torch.int64).remainder(256).to(torch.uint8).view(254, 3)
        lut_ck = td / "lut_run" / "swa.pt"; lut_ck.parent.mkdir()
        torch.save({**base, "arch": DEFAULT_ARCH, "norm": "depthir", "depth_lut": lut}, lut_ck)
        plain_ck = td / "plain_run" / "swa.pt"; plain_ck.parent.mkdir()
        torch.save({**base, "arch": DEFAULT_ARCH, "norm": "depthir"}, plain_ck)
        P.pack([lut_ck, plain_ck], td / "lut_pack.pt", half=True)
        mems = torch.load(td / "lut_pack.pt", map_location="cpu", weights_only=False)
        check_true("§96: package.pack carries a stamped depth_lut BITWISE into the container member",
                   mems[0].get("depth_lut") is not None and torch.equal(mems[0]["depth_lut"], lut))
        check_true("§96: … and None for a member that carries none", "depth_lut" in mems[1] and mems[1]["depth_lut"] is None)
        q0, q1 = Q.member_of(lut_ck), Q.member_of(plain_ck)
        check_true("§96: package_int8.member_of carries a stamped depth_lut BITWISE", q0.get("depth_lut") is not None and torch.equal(q0["depth_lut"], lut))
        check_true("§96: … and None for a member that carries none", "depth_lut" in q1 and q1["depth_lut"] is None)
        check_raises("§96: predict.Ensemble REFUSES a LUT-bearing member until the ordinal decode path exists",
                     lambda: Ensemble([lut_ck], device_pref="cpu"))
        check_true("§96: … and loads the plain member as before", len(Ensemble([plain_ck], device_pref="cpu").by_branch["depthir"]) == 1)

        # ── §105 (D34; P-A): the CROP COUNT is stamped metadata with an inert default — absent ⇒ 1, 3 ⇒ the 3-crop TTA, anything
        # else REFUSED; both packers carry the key (None when absent). Committed BEFORE any §105 number.
        check("§105: a member with no crops stamp resolves to 1 (the shipped view)", Ensemble([plain_ck], device_pref="cpu").by_branch["depthir"][0]["crops"], 1)
        c3 = td / "crop3_run" / "swa.pt"; c3.parent.mkdir()
        torch.save({**base, "arch": DEFAULT_ARCH, "norm": "depthir", "crops": 3}, c3)
        check("§105: a member stamped crops=3 reports 3", Ensemble([c3], device_pref="cpu").by_branch["depthir"][0]["crops"], 3)
        c2 = td / "crop2_run" / "swa.pt"; c2.parent.mkdir()
        torch.save({**base, "arch": DEFAULT_ARCH, "norm": "depthir", "crops": 2}, c2)
        check_raises("§105: an UNKNOWN crop count (2) raises rather than defaulting", lambda: Ensemble([c2], device_pref="cpu"))
        P.pack([c3, plain_ck], td / "crop_pack.pt", half=True)
        mems = torch.load(td / "crop_pack.pt", map_location="cpu", weights_only=False)
        check("§105: package.pack carries the crops stamp into the container member", mems[0].get("crops"), 3)
        check_true("§105: … and None for a member that carries none", "crops" in mems[1] and mems[1]["crops"] is None)
        check("§105: package_int8.member_of carries the crops stamp", Q.member_of(c3).get("crops"), 3)

        # ── §119 (D50; P-A): the FUSION FORMULA is the SECOND stamped decode key beside `crops` — absent ⇒ None ⇒ the
        # arithmetic mean (bitwise the pre-§119 decode, tests/check_fusion_key.py); a dict ⇒ §118's logit-adjusted
        # geometric pool; an unknown form, a malformed prior or members that DISAGREE are REFUSED. Both packers carry
        # it, from --fusion-json (it is the artefact's decode rule, not a member's training attribute), and the JSON is
        # validated through predict.parse_fusion at pack time — ONE definition of a legal stamp. Committed BEFORE any
        # §119 number. The prior below is SYNTHETIC: the twin's real π is a corpus-derived TRAIN statistic and never
        # enters this repository (.gitignore:1-2; the §96 depth_lut precedent).
        PI = [(i + 1) / 820.0 for i in range(40)]                      # 40 positive floats summing to 1
        FUSE = {"form": "logit_adjusted_geo", "lambda": 0.25, "prior": PI}
        check("§119: a member with no fusion stamp resolves to None (the arithmetic mean — the inert default)",
              Ensemble([plain_ck], device_pref="cpu").fusion, None)
        f1 = td / "fusion_run" / "swa.pt"; f1.parent.mkdir()
        torch.save({**base, "arch": DEFAULT_ARCH, "norm": "depthir", "fusion": FUSE}, f1)
        check("§119: a stamped member reports the twin's form and λ",
              Ensemble([f1], device_pref="cpu").fusion[:2], ("logit_adjusted_geo", 0.25))
        check_true("§119: … and its π travels in full, bitwise",
                   Ensemble([f1], device_pref="cpu").fusion[2] == tuple(PI), f"{len(PI)} floats")
        for tag, bad in (("an unknown FORM", {**FUSE, "form": "geometric_mean"}),
                         ("a 39-class prior", {**FUSE, "prior": PI[:39]}),
                         ("a negative λ", {**FUSE, "lambda": -1.0}),
                         ("a non-positive prior entry", {**FUSE, "prior": [0.0] + PI[1:]})):
            bp = td / f"fusion_bad_{abs(hash(tag))}.pt"
            torch.save({**base, "arch": DEFAULT_ARCH, "norm": "depthir", "fusion": bad}, bp)
            check_raises(f"§119: {tag} raises rather than defaulting", lambda q=bp: Ensemble([q], device_pref="cpu"))
        P.pack([f1, plain_ck], td / "fuse_pack.pt", half=True, fusion=FUSE)
        mems = torch.load(td / "fuse_pack.pt", map_location="cpu", weights_only=False)
        check_true("§119: package.pack stamps --fusion-json's formula on EVERY member",
                   all(m.get("fusion") == FUSE for m in mems), f"{len(mems)} members, form {mems[0]['fusion']['form']} λ {mems[0]['fusion']['lambda']}")
        mems0 = torch.load(td / "crop_pack.pt", map_location="cpu", weights_only=False)
        check_true("§119: … and None on every member when no formula is given (every container built before 2026-09-12)",
                   all("fusion" in m and m["fusion"] is None for m in mems0))
        check_true("§119: package_int8.member_of carries the fusion stamp", Q.member_of(f1, fusion=FUSE).get("fusion") == FUSE)
        check("§119: … and None when none is given", Q.member_of(f1).get("fusion"), None)
        fj = td / "fusion.json"; fj.write_text(json.dumps(FUSE), encoding="utf-8")
        check_true("§119: package.fusion_from_json reads the file the packers are given", P.fusion_from_json(fj) == FUSE)
        bj = td / "fusion_bad.json"; bj.write_text(json.dumps({**FUSE, "form": "geometric_mean"}), encoding="utf-8")
        check_raises("§119: … and REFUSES at pack time a stamp predict.py would refuse at load (one definition)",
                     lambda: P.fusion_from_json(bj))
        split = td / "fuse_split.pt"
        sm = torch.load(td / "fuse_pack.pt", map_location="cpu", weights_only=False)
        sm[1]["fusion"] = None; torch.save(sm, split)
        check_raises("§119: a container whose members DISAGREE about the fusion has no single decode rule — refused",
                     lambda: Ensemble([split], device_pref="cpu"))

    # ── 4. the writer half, on the artefacts that exist ─────────────────────
    print("\n4 · Every checkpoint on disk is self-consistent")
    ckpts = sorted((ROOT / "runs").glob("*/[bs]*.pt")) if (ROOT / "runs").is_dir() else []
    if not ckpts:
        SKIPPED.append("section 4: no runs/ on this clone")
        print("  [skip] no checkpoints under runs/ — nothing to walk (runs/ is gitignored)")
    else:
        stamped = [p for p in ckpts if "arch" in torch.load(p, map_location="meta",
                                                            weights_only=False, mmap=True)]
        half = []
        for p in ckpts:
            ck = torch.load(p, map_location="meta", weights_only=False, mmap=True)
            if ("arch" in ck) != ("norm" in ck):
                half.append(p.parent.name + "/" + p.name)
        check_true("no checkpoint carries arch without norm, or the reverse", not half,
                   f"{len(half)} half-stamped: {half[:5]}" if half else
                   f"{len(ckpts)} checkpoints, {len(stamped)} stamped, "
                   f"{len(ckpts) - len(stamped)} legacy")
        # A CHECK OVER AN EMPTY SET IS NOT A PASS. Every checkpoint on disk
        # today predates the stamp, so this cell has nothing to examine and would
        # otherwise report [ok] over zero files -- the same vacuous green that
        # PRETRAINING-GAP.md §4 found in check_swa.py's bns(). It goes live by
        # itself the moment any run trains under the new train.py.
        if not stamped:
            SKIPPED.append("section 4: every checkpoint on disk predates the arch/norm stamp")
            print("  [skip] no STAMPED checkpoint exists yet — this cell would pass vacuously")
        else:
            bad = []
            for p in stamped:
                ck = torch.load(p, map_location="meta", weights_only=False, mmap=True)
                # §96 (c) (D29, m284): a depth+IR member trained under --input-adapt valid stamps the 5-statistic key
                # `depthir_valid` (the validity plane is its fifth channel) — train.py's rule, mirrored here; every other
                # checkpoint's stamp is what norm_key derives from (branch, arch).
                expected = ("depthir_valid" if ck.get("input_adapt") == "valid" and ck["branch"] == "depthir"
                            else norm_key(ck["branch"], ck["arch"]))
                if ck["norm"] != expected:
                    bad.append(f"{p.parent.name}: {ck['norm']} != {expected}")
            check_true("every stamped norm matches what norm_key would derive (the §96 (c) valid rule included)", not bad,
                       "; ".join(bad[:3]) if bad else f"{len(stamped)} checked")

    # ── 5. the writer stamps BOTH save sites ───────────────────────────────
    # Structural, by AST: the alternative is two 12-minute training runs to
    # assert a property of the source. A torch.save in train.py that omits
    # either key is the regression this catches.
    print("\n5 · train.py stamps arch and norm at EVERY save site")
    tree = ast.parse((ROOT / "src" / "train.py").read_text(encoding="utf-8"))
    saves = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "save"]
    check_true("train.py has the expected number of save sites", len(saves) == 2,
               f"found {len(saves)} (best.pt and swa.pt)")
    for i, call in enumerate(saves):
        keys = set()
        if call.args and isinstance(call.args[0], ast.Dict):
            keys = {k.value for k in call.args[0].keys if isinstance(k, ast.Constant)}
        # m73 adds input_size to the self-describing set: it is the third field
        # a member needs to be run correctly, and the one whose absence reached
        # a submission rather than only a measurement.
        want = {"arch", "norm", "input_size"}
        check_true(f"save site {i + 1} stamps arch, norm and input_size", want <= keys,
                   f"missing {sorted(want - keys)}" if not want <= keys else f"{len(keys)} keys")

    # ── 6. the two entry points name the SAME four checkpoints ─────────────
    # m72. This is the drift that already happened twice. package.py kept its
    # own SHIPPING list of 2 while predict.py shipped 4, so `--shipping` weighed
    # an object nobody would send, and nothing caught it for a whole submission
    # cycle. package.py now IMPORTS the list, so the remaining copy that can
    # drift is train.sh's loop -- and a train.sh that rebuilds checkpoints under
    # names inference.sh does not load is a verification package that fails on
    # the committee's machine, not ours.
    print("\n6 · train.sh rebuilds exactly the checkpoints predict.py ships")
    sys.path.insert(0, str(ROOT / "src"))
    from predict import SHIPPING  # noqa: E402
    sh = (ROOT / "train.sh").read_text(encoding="utf-8")
    # train.sh's spec now STATES the run name rather than reconstructing it:
    # "<branch> <fold> <suffix> <arch> <resolution> <seed>". Reconstruction broke
    # twice — once on m80's mixed-arch member (name ends _n_s1, not _w_s1) and
    # once on m84's seed-2 twin — and each time this check caught it by matching
    # nothing and reporting "0 vs N" rather than a wrong name. Parsing a stated
    # name removes the whole class.
    # m110: train.sh is now a TWO-STAGE pipeline. The shipped members are
    # self-trained, so they need pseudo-labels generated from an ensemble of the
    # stage-1 members — and stage 1's own checkpoints are NOT shipped.
    #
    # The invariant is therefore sharpened, not relaxed: the SHIPPED stage must
    # match SHIPPING element by element (unchanged), AND the prerequisite stage
    # must exist (new). Parsing the STAGE2 block by name rather than globbing
    # every spec in the file is what keeps an intermediate from ever being
    # mistaken for a shipped member.
    def _block(name):
        m = re.search(rf'^{name}=\((.*?)\)$', sh, re.S | re.M)
        # THE SPEC GREW A 7th FIELD (lr) ON 2026-08-20, AND THIS PARSER
        # FOLLOWED IT RATHER THAN THE OTHER WAY AROUND. m124: the two backbone
        # families have optima 10x apart, so lr belongs to the spec instead of
        # being a global constant. The trailing `[^"]*` accepts any number of
        # fields AFTER the seed, so both the 6- and 7-field forms parse.
        # This relaxes the PARSER, never the ASSERTION: the checks below still
        # compare run names element-by-element against predict.SHIPPING, and a
        # divergence still fails. Mutation-verified when the field was added.
        # 022 (D20, 2026-08-29): the spec grew MACHINE, T and EXTRA fields after batch
        # size; the run name is now {branch}_{fold}_{machine}_{suffix}. Parsed as
        # tokens; machine defaults to ALIEN for the older 8-field form. The assertion
        # below is unchanged: derived names must equal SHIPPING element by element.
        out = []
        for q in (re.findall(r'"([^"]*)"', m.group(1)) if m else []):
            t = q.split()
            if len(t) >= 6 and t[0] in ('depthir', 'thermal'):
                out.append((t[0], t[1], t[2], (t[8] if len(t) >= 9 else 'ALIEN').lower()))
        return out

    stage1, specs = _block("STAGE1"), _block("STAGE2")
    # The released train.sh has no pseudo-label stage: neither shipped member is
    # trained on pseudo-labelled test clips (the fold field of every spec is
    # `all18`, never `all18pl`), so there is no stage-1 ensemble and no label step.
    needs_pl = any(f == "all18pl" for _b, f, *_ in specs)
    check_true("no shipped member is trained on pseudo-labels (no `all18pl` spec)",
               not needs_pl, f"needs_pl={needs_pl}")
    check_true("train.sh carries no STAGE1 block (the pseudo-label pipeline is not "
               "part of the release)", len(stage1) == 0, f"{len(stage1)} stage-1 specs")
    derived = [f"runs/{b}_{f}_{mc}_{suf}" for b, f, suf, mc in specs]
    # Assert the INVARIANT, not the count of the day. A hard-coded 4 re-broke the
    # moment 005 legitimately went to five members, which would have taught the
    # habit of editing expected values to make a suite pass — the one thing this
    # file's own footer forbids.
    check_true("train.sh's loop yields as many runs as SHIPPING",
               len(derived) == len(SHIPPING) and len(derived) > 0,
               f"{len(derived)} vs {len(SHIPPING)}")
    check_true("train.sh's run names == predict.py's SHIPPING",
               derived == list(SHIPPING),
               "match" if derived == list(SHIPPING)
               else f"train.sh {derived} != SHIPPING {list(SHIPPING)}")
    import tools.package as _pkg  # noqa: E402
    check_true("package.py weighs that same list (imported, not copied)",
               list(_pkg.SHIPPING) == list(SHIPPING), "one definition")

    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed"
          + (f", {len(SKIPPED)} section(s) skipped" if SKIPPED else ""))
    print("=" * 62)
    for s in SKIPPED:
        print(f"  [skipped] {s}")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  · {f}")
        print("\nDiagnose these. Do not adjust the expected values to make them pass.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
