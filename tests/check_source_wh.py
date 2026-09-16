"""§124 (P-A) — THE source_wh KEY'S BITWISE TEST (the m286 ② / check_fusion_key shape), CPU only (CUDA hidden).

THE CLAIM UNDER TEST, in the words P-A uses: *"the incumbent's CSV is bitwise unchanged with the flag off"*. `crops`
(§105) was the first stamped decode key, `fusion` (§119) the second; `source_wh` is the THIRD, and it is the first one
that touches the DECODE OF PIXELS rather than the arithmetic over probabilities — so its inertness has to be proved on
the frames themselves, not only on the decision.

WHY THE KEY EXISTS. Every shipped member trained on frames the cache decoded at 160x120 (src/preprocess.py's STREAMS
table) and then resized to the stamped input_size. A §124 member trained on frames decoded at 320x240 (`--size-wh
320x240`, cache_hr) and resized to 320x320. If inference decoded its thermal frames at 160x120 and resized those up,
the member would see a pixel distribution it never trained on — 76.8k source pixels replaced by 19.2k upsampled ones —
and NOTHING would raise. That is m73 with a different number, and m73 reached submission 004.

  1 · THE PARSER AND THE LOADER, on synthetic checkpoints — the semantics, without pixels:
      · absent  ⇒ Ensemble.source_wh[branch] is None (the shipped decode; the inert default)
      · [320,240] ⇒ reported as the tuple (320, 240)
      · malformed (one number, zero, a negative) ⇒ REFUSED, never defaulted
      · two members of ONE branch that disagree — INCLUDING stamped beside unstamped — ⇒ REFUSED
      · tools/package.py's --source-wh BRANCH=WxH parser: accepts, and refuses what predict.py would refuse
  2 · THE DECODE, on REAL test clips through the REAL path (Ensemble._frames, from the original files — R8; no cache):
      (a) with the key ABSENT the frames are BITWISE `preprocess.build_clip(sampled, stream)` — the pre-§124 call,
          written out here rather than asked of the code under test — on BOTH branches
      (b) with thermal stamped [320,240] the thermal frames are BITWISE `build_clip(sampled, "thermal",
          size_wh=(320,240))` AND BITWISE what the CLI path produces (STREAMS mutated, as the cache build did) —
          the two ways of asking for 320x240 must agree, or the member trains on one and infers on the other
      (c) the MUTATION: those frames must DIFFER from the 160x120 ones (the key reaches the decode; not cosmetic)
      (d) the depth+IR member of the SAME container is BITWISE unaffected when only thermal is stamped
      (e) the ABSENT-key container's DECISION on real clips is unchanged — the real models, the real _probs

§125 ADDS SECTION 3 — THE DEPTH+IR BRANCH. §124 stamped one branch; §125 trains the OTHER member on a 320x240
depthir_raw cache, and the candidate that pairs them (027) stamps BOTH. Everything §124 proved for thermal has to
hold for depthir on its own stream, and the two-branch container has to be a real shape rather than an assumed one:
      (f) with the key ABSENT — and with an EXPLICIT None, which is what tools/package.py writes for an unnamed
          branch — the depth+IR frames are BITWISE today's path
      (g) with depthir stamped [320,240] they are BITWISE build_clip(sampled, "depthir_raw", size_wh=(320,240))
          AND BITWISE the CLI path (STREAMS["depthir_raw"] mutated, as `--size-wh 320x240` does), at (T,240,320,4)
      (h) the MUTATION, and the THERMAL member of that container BITWISE unaffected — the mirror of (d)
      (i) THE RESIZE: those 240x320 frames go through the real _probs and are resized to the member's STAMPED
          input_size before the forward — a real 40-class distribution, and one that DIFFERS from the shipped
          decode's, so the key reaches the DECISION and not only the pixels
      (j) BOTH BRANCHES AT ONCE: package.py's --source-wh parses `thermal=320x240 depthir=320x240` into two
          entries, predict.py's PER-BRANCH dict reports both, and each branch's frames are its own reference
      (k) the branch-consistency refusal is unchanged, and an AGREEING pair of depth+IR members is still accepted

Usage:  CUDA_VISIBLE_DEVICES="" ./.venv/bin/python tests/check_source_wh.py
Standing rule: diagnose any failure. Never adjust the expected value.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import preprocess as PP                                  # noqa: E402
from dataset import _tsn_indices                         # noqa: E402
from predict import Ensemble                             # noqa: E402
from tools.package import source_wh_from_args            # noqa: E402

CONTAINER = ROOT / "checkpoints" / "model.pth"
TEST = Path(os.environ.get("CUHKX_TEST_ROOT", str(Path.home() / "cuhk-x" / "test_extracted"))) / "small_model_track_test"
SHA_022 = "218266cc3230e87f"
HR_WH = (320, 240)                       # §124's source cache: src/preprocess.py --size-wh 320x240
PASS: list[str] = []
FAIL: list[str] = []
SKIPPED: list[str] = []


def check(desc: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(desc)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {desc}{(' — ' + detail) if detail else ''}")


def skip(desc: str, detail: str) -> None:
    SKIPPED.append(desc)
    print(f"  [skip] {desc} — {detail}")


def check_raises(desc: str, fn, exc=ValueError) -> None:
    try:
        fn()
    except exc as e:
        check(desc, True, str(e)[:90])
    except Exception as e:                               # noqa: BLE001
        check(desc, False, f"raised {type(e).__name__}, expected {exc.__name__}")
    else:
        check(desc, False, "did NOT raise — it resolved to something")


def synth(td: Path, tag: str, **extra) -> Path:
    """A loadable one-member depthir checkpoint carrying whatever stamps `extra` names."""
    from model import DEFAULT_ARCH, build
    m = build("depthir", n_segment=8, arch=DEFAULT_ARCH, pretrained=False)
    p = td / f"synth_{tag}" / "swa.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "branch": "depthir", "T": 8, "in_channels": 4,
                "arch": DEFAULT_ARCH, "norm": "depthir", "input_size": "224x224", **extra}, p)
    return p


def stamped(td: Path, tag: str, value, which=lambda m: m["branch"] == "thermal") -> Path:
    """A COPY of the slot's container with source_wh written on the members `which` selects. The slot is never opened
    for write (P-B): torch.load reads it, torch.save writes into the temporary directory."""
    mems = torch.load(CONTAINER, map_location="cpu", weights_only=False)
    for m in mems:
        m["source_wh"] = value if which(m) else None
    p = td / f"model_source_wh_{tag}.pt"
    torch.save(mems, p)
    return p


def reference_frames(clip_dir: Path, branch: str, T: int, size_wh=None):
    """What the frames MUST be — the pre-§124 expression, written out here, not asked of predict.py.

    src/predict.py:_frames verbatim minus the new argument: the same path listing, the same TSN indices, the same
    build_clip, the same whole-clip repair fallback.
    """
    stream = "thermal" if branch == "thermal" else "depthir_raw"
    if branch == "thermal":
        paths, _ = PP.thermal_frames(clip_dir / "Thermal")
        paths = [(q,) for q in paths]
    else:
        ir, dp, _ = PP.depthir_frames(clip_dir / "IR", clip_dir / "Depth_Color")
        paths = list(zip(ir, dp))
    if not paths:
        return None
    idx = _tsn_indices(len(paths), T, train=False, rng=None)
    kw = {} if size_wh is None else {"size_wh": size_wh}
    frames, stats = PP.build_clip([paths[i] for i in idx], stream, **kw)
    if stats["bad"]:
        frames, stats = PP.build_clip(paths, stream, **kw)
        frames = frames[idx] if len(frames) else frames
    if "depth" in stats["absent"] or "rgb" in stats["absent"]:
        return None
    return np.asarray(frames)


def cli_path_frames(clip_dir: Path, branch: str, T: int):
    """The SAME 320x240 frames asked for the OTHER way — by mutating STREAMS, which is exactly what
    `src/preprocess.py --size-wh 320x240` did to build cache_hr (§124: --streams thermal; §125: --streams
    depthir_raw). Restored in a finally, so nothing leaks."""
    stream = "thermal" if branch == "thermal" else "depthir_raw"
    saved = PP.STREAMS[stream]["size_wh"]
    try:
        PP.STREAMS[stream]["size_wh"] = HR_WH
        return reference_frames(clip_dir, branch, T)
    finally:
        PP.STREAMS[stream]["size_wh"] = saved


def main() -> int:
    torch.set_num_threads(4)                 # a GPU cell may be training; this test is a CPU guest on the box
    print("§124 (P-A) — the source_wh key: inert when absent, exact when present\n")

    # ── 1 · THE PARSER AND THE LOADER ────────────────────────────────────────
    print("1 · THE PARSER AND THE LOADER (synthetic checkpoints; no pixels)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        plain = synth(td, "plain")
        check("a member with NO source_wh stamp resolves to None (the shipped 160x120 decode — the inert default)",
              Ensemble([plain], device_pref="cpu").source_wh["depthir"] is None)
        hr = synth(td, "hr", source_wh=[320, 240])
        check("a member stamped [320,240] reports the tuple (320, 240)",
              Ensemble([hr], device_pref="cpu").source_wh["depthir"] == HR_WH)
        for bad, why in (([320], "one number"), ([320, 0], "a zero"), ([-1, 240], "a negative")):
            check_raises(f"a malformed source_wh ({why}) is REFUSED, never defaulted",
                         lambda b=bad: Ensemble([synth(td, f"bad{abs(hash(str(b)))%9999}", source_wh=b)], device_pref="cpu"))
        check_raises("two members of ONE branch with DIFFERENT source_wh are REFUSED",
                     lambda: Ensemble([hr, synth(td, "hr2", source_wh=[256, 192])], device_pref="cpu"))
        check_raises("…and a stamped member beside an UNSTAMPED one is a disagreement too, not a default",
                     lambda: Ensemble([hr, plain], device_pref="cpu"))
        # §125 (k): the refusal is about DISAGREEMENT, not about the key. Two depth+IR members that AGREE at
        # [320,240] are one branch with one decode and must load — a §125 arm can ship more than one seed.
        check("two depth+IR members that AGREE at [320,240] are ACCEPTED — the refusal is about disagreement, not the key",
              Ensemble([hr, synth(td, "hr_agree", source_wh=[320, 240])], device_pref="cpu").source_wh["depthir"] == HR_WH)
        check("package.py --source-wh thermal=320x240 parses to {'thermal': (320, 240)}",
              source_wh_from_args(["thermal=320x240"]) == {"thermal": HR_WH})
        # §125 (j): the 027 container stamps BOTH branches, so the repeatable flag must carry two entries at once.
        check("package.py --source-wh thermal=320x240 depthir=320x240 parses to BOTH branches — the 027 shape",
              source_wh_from_args(["thermal=320x240", "depthir=320x240"]) == {"thermal": HR_WH, "depthir": HR_WH})
        check("…and no flag at all parses to {} — every member stamped None, today's containers rebuilt bitwise",
              source_wh_from_args(None) == {} and source_wh_from_args([]) == {})
        for bad in ("thermal", "thermal=320", "thermal=0x240"):
            check_raises(f"package.py refuses --source-wh {bad!r}", lambda b=bad: source_wh_from_args([b]), SystemExit)

    # ── 2 · THE DECODE, ON REAL CLIPS ────────────────────────────────────────
    print("\n2 · THE DECODE, on REAL test clips through Ensemble._frames (the original files — R8; no cache)")
    if not CONTAINER.is_file() or not TEST.is_dir():
        skip("the real-artefact section", f"{CONTAINER if not CONTAINER.is_file() else TEST} is absent")
        print(f"\n  {len(PASS)} passed, {len(FAIL)} failed, {len(SKIPPED)} skipped")
        return 1 if FAIL else 0

    sha = hashlib.sha256(CONTAINER.read_bytes()).hexdigest()[:16]
    mems = torch.load(CONTAINER, map_location="cpu", weights_only=False)
    print(f"      container sha16 {sha} · {len(mems)} members · source_wh {[m.get('source_wh') for m in mems]}")
    if sha == SHA_022:
        check("the shipped 022 container carries NO source_wh key on any member — the flag is off",
              all(m.get("source_wh") is None for m in mems))
    else:
        skip("the shipped 022 container carries no source_wh key",
             f"this tree's slot is {sha}, not 022 ({SHA_022}); the assertion is scoped to 022")

    clips = sorted(p for p in TEST.iterdir() if p.is_dir() and p.name.startswith("SM_test_"))[:3]
    print(f"      clips {[c.name for c in clips]}")
    e0 = Ensemble([CONTAINER], device_pref="cpu")
    base = {b: [e0._frames(c, b) for c in clips] for b in e0.by_branch}

    # (a) the key ABSENT ⇒ bitwise the pre-§124 expression, on BOTH branches
    for b in sorted(base):
        ref = [reference_frames(c, b, e0.T[b]) for c in clips]
        check(f"(a) {b}: with the key ABSENT the frames are BITWISE the pre-§124 build_clip call",
              all(x is not None and y is not None and np.array_equal(x, y) for x, y in zip(base[b], ref)),
              f"shape {None if base[b][0] is None else base[b][0].shape}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        eh = Ensemble([stamped(Path(td), "hr", list(HR_WH))], device_pref="cpu")
        check("the stamped container reports thermal at (320, 240) and depthir at None",
              eh.source_wh.get("thermal") == HR_WH and eh.source_wh.get("depthir") is None,
              str(eh.source_wh))
        hrf = [eh._frames(c, "thermal") for c in clips]
        # (b) the two ways of asking for 320x240 must agree, bitwise
        ref_kw = [reference_frames(c, "thermal", eh.T["thermal"], size_wh=HR_WH) for c in clips]
        ref_cli = [cli_path_frames(c, "thermal", eh.T["thermal"]) for c in clips]
        check("(b) the stamped thermal frames are BITWISE build_clip(..., size_wh=(320,240))",
              all(np.array_equal(x, y) for x, y in zip(hrf, ref_kw)),
              f"shape {hrf[0].shape}")
        check("(b) …and BITWISE the CLI path (STREAMS mutated) — the very frames src/preprocess.py --size-wh 320x240 "
              "wrote into cache_hr, so the member infers on what it trained on",
              all(np.array_equal(x, y) for x, y in zip(hrf, ref_cli)))
        check("(b) …and the shape is (T, 240, 320, 3), not (T, 120, 160, 3)",
              all(x.shape[1:] == (240, 320, 3) for x in hrf) and all(x.shape[1:] == (120, 160, 3) for x in base["thermal"]))
        # (c) the mutation
        check("(c) MUTATION: the 320x240 frames DIFFER from the 160x120 ones — the key reaches the decode",
              all(x.shape != y.shape for x, y in zip(hrf, base["thermal"])))
        # (d) the other branch is untouched
        dep = [eh._frames(c, "depthir") for c in clips]
        check("(d) the depth+IR member of the SAME container is BITWISE unaffected — the key is PER BRANCH",
              all(np.array_equal(x, y) for x, y in zip(dep, base["depthir"])))
        # (e) the decision, through the real models, with the key absent
        en = Ensemble([stamped(Path(td), "none", None, which=lambda m: False)], device_pref="cpu")
        d0 = [e0.predict(c) for c in clips[:2]]
        dn = [en.predict(c) for c in clips[:2]]
        check("(e) an EXPLICIT source_wh=None stamp decides exactly as the unstamped slot does — the CSV is unaffected",
              d0 == dn and all(v is not None for v in d0), f"{d0} vs {dn}")

    # ── 3 · THE DEPTH+IR BRANCH (§125) ───────────────────────────────────────
    # §124 stamped thermal; §125 trains the depth+IR member on a 320x240 depthir_raw cache and 027 stamps BOTH.
    # Same claims, the other stream — and then the two-branch container as a real shape, not an assumed one.
    print("\n3 · THE DEPTH+IR BRANCH (§125): the same key on the depthir_raw stream, and BOTH branches at once")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        TD = e0.T["depthir"]

        # (f) inert: an EXPLICIT None on depthir — what package.py writes for a branch --source-wh does not name.
        edn = Ensemble([stamped(td, "depnone", None, which=lambda m: m["branch"] == "depthir")], device_pref="cpu")
        check("(f) depthir: an EXPLICIT source_wh=None stamp decodes BITWISE the unstamped path",
              all(np.array_equal(edn._frames(c, "depthir"), b) for c, b in zip(clips, base["depthir"])),
              f"shape {base['depthir'][0].shape}")

        ed = Ensemble([stamped(td, "dephr", list(HR_WH), which=lambda m: m["branch"] == "depthir")], device_pref="cpu")
        check("(f) the stamped container reports depthir at (320, 240) and thermal at None — the key is PER BRANCH",
              ed.source_wh.get("depthir") == HR_WH and ed.source_wh.get("thermal") is None, str(ed.source_wh))
        hrd = [ed._frames(c, "depthir") for c in clips]
        # (g) the two ways of asking for 320x240 must agree, bitwise — the member infers on what it trained on
        ref_kw_d = [reference_frames(c, "depthir", TD, size_wh=HR_WH) for c in clips]
        ref_cli_d = [cli_path_frames(c, "depthir", TD) for c in clips]
        check("(g) the stamped depth+IR frames are BITWISE build_clip(..., 'depthir_raw', size_wh=(320,240))",
              all(np.array_equal(x, y) for x, y in zip(hrd, ref_kw_d)), f"shape {hrd[0].shape}")
        check("(g) …and BITWISE the CLI path (STREAMS['depthir_raw'] mutated) — the very frames "
              "src/preprocess.py --streams depthir_raw --size-wh 320x240 writes into cache_hr for §125",
              all(np.array_equal(x, y) for x, y in zip(hrd, ref_cli_d)))
        check("(g) …and the shape is (T, 240, 320, 4), not (T, 120, 160, 4)",
              all(x.shape[1:] == (240, 320, 4) for x in hrd) and all(x.shape[1:] == (120, 160, 4) for x in base["depthir"]))
        # (h) the mutation, and the other branch untouched — the mirror of (d)
        check("(h) MUTATION: the 320x240 depth+IR frames DIFFER from the 160x120 ones — the key reaches the decode",
              all(x.shape != y.shape for x, y in zip(hrd, base["depthir"])))
        check("(h) the THERMAL member of the SAME container is BITWISE unaffected when only depthir is stamped",
              all(np.array_equal(ed._frames(c, "thermal"), b) for c, b in zip(clips, base["thermal"])))

        # (i) THE RESIZE, through the real _probs on the real member: 240x320 decode → the member's STAMPED
        # input_size → the forward. This is the half a frame comparison cannot see, and it is where m73 was lost.
        want_size = tuple(ed.by_branch["depthir"][0]["input_size"])
        p_hr = ed._probs(hrd[0], "depthir")
        p_sh = e0._probs(base["depthir"][0], "depthir")
        check(f"(i) the 240x320 frames are resized to the member's STAMPED input_size {want_size} and the member runs "
              "— a real 40-class distribution",
              p_hr.shape == p_sh.shape == (40,) and abs(float(p_hr.sum()) - 1.0) < 1e-5, f"sum {float(p_hr.sum()):.6f}")
        check("(i) …and that distribution DIFFERS from the shipped decode's — the key reaches the DECISION, not only the pixels",
              not np.array_equal(p_hr, p_sh), f"max|Δp| {float(np.abs(p_hr - p_sh).max()):.4f}")

        # (j) BOTH BRANCHES AT ONCE — the 027 container's shape
        eb = Ensemble([stamped(td, "both", list(HR_WH), which=lambda m: True)], device_pref="cpu")
        check("(j) a container with BOTH branches stamped reports both at (320, 240) — predict.py's per-branch dict",
              eb.source_wh.get("thermal") == HR_WH and eb.source_wh.get("depthir") == HR_WH, str(eb.source_wh))
        check("(j) …and its THERMAL frames are BITWISE the 320x240 reference",
              all(np.array_equal(eb._frames(c, "thermal"), y)
                  for c, y in zip(clips, [reference_frames(c, "thermal", e0.T["thermal"], size_wh=HR_WH) for c in clips])))
        check("(j) …and its DEPTH+IR frames are BITWISE the 320x240 reference — one container, two decodes, neither borrowed",
              all(np.array_equal(x, y) for x, y in zip([eb._frames(c, "depthir") for c in clips], ref_kw_d)))

    print(f"\n  {len(PASS)} passed, {len(FAIL)} failed" + (f", {len(SKIPPED)} skipped" if SKIPPED else ""))
    if SKIPPED:
        print("  SKIPPED:\n    " + "\n    ".join(SKIPPED))
    if FAIL:
        print("FAILED:\n  " + "\n  ".join(FAIL) + "\n\nDiagnose these. Do not adjust the expected values to make them pass.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
