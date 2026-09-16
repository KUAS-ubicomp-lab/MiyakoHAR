"""Test-set prediction for the CUHK-X Small Model Track.

Contract, taken from the organisers' own files rather than assumed:

    Testing/test_file/test.csv          path,prediction   (prediction empty)
    Testing/test_file/sample_submission.csv               (prediction filled)

    path == "small_model_track_test/SM_test_0001/"   -- note the trailing slash
    prediction == action_id, an integer in [0, 39]

405 test clips, so 406 lines including the header.

Two rules govern this file and outrank tidiness:

1. It must never crash. The competition runs it on a hidden set we cannot
   inspect; a clip that is missing, one frame long, corrupt or an odd
   resolution must produce a fallback prediction, not a traceback. Every
   per-clip failure is caught and counted.
2. The row keys come from the organisers' test.csv whenever one is present,
   verbatim. Reconstructing them from the directory tree is the fallback, not
   the default -- a path string that differs by a trailing slash scores zero
   while looking perfectly correct.

Day 1 predicts a constant class. The point of that is not accuracy: it is to
prove the submission path end to end before any model exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import os
import sys
from pathlib import Path

# Most frequent class in training: 36_Walk, 365 of 3036 clips.
# Used both as the day-1 constant prediction and, permanently, as the
# per-clip fallback when prediction fails.
FALLBACK_ACTION_ID = 36

CLIP_DIR_PATTERN = re.compile(r"^SM_test_\d+$")

# THE SHIPPING SET — ONE definition, imported by tools/package.py.
# m72: this list lived here as a local and a SECOND, DIFFERENT list lived in
# package.py (2 checkpoints against these 4), so `package.py --shipping` had not
# weighed the set predict.py actually ships since submission 003. Two copies of
# a fact is one copy too many; the packer now imports this one.
#
# Submission 005 (m75): s3d-K400 @168x224, FIVE members -- depthir f0/f1/f2 and
# thermal f0/f1. Weighed at 80.47 MB fp16; a sixth measures 96.57 and BREACHES
# D6's 95, so five is the hard maximum for this architecture.
#
# 004 shipped four of these (2 folds x 2 branches) at 64.38 MB and deliberately
# left the slots unspent, because its only job was to test whether m64's
# CV - 4.90 pp constant survived leaving the ResNet-18 family. It did not: the
# s3d offset came back at +1.587 pp and 004 scored 0.61691 = 124/201. The slots
# are now spent on the FOLD axis.
#
# THE FOLD AXIS CANNOT BE PRICED ON VALIDATION AND THIS IS NOT A LOOPHOLE --
# it is stated in the Ensemble docstring below and in m52. Fold f's model saw
# fold g's val subjects, so a cross-fold ensemble is unscoreable on val BY
# CONSTRUCTION. Adding fold 2 buys wider SUBJECT coverage at test time, which is
# a variance argument; m58 prices it near +2 pp and only the board can confirm.
# CONSEQUENCE: 005's CV is NOT comparable to 004's in m64's convention (fold 2
# contributes one branch, not two), so 005 is NOT the n=2 calibration test. That
# test needs another 2x2 submission.
# REVERTED TO FOUR 2026-08-17 BY THE BOARD, NOT BY VALIDATION (m79).
# 005 shipped the five-member set above and scored 0.61194 = 123/201 against
# 004's 0.61691 = 124/201 -- EXACTLY ONE CLIP WORSE. The fold-2 member changed
# 31 of 405 predictions and netted -1.
#
# The honest reading is NOT "the fold axis hurts": one clip cannot support that.
# It is "the fold axis buys nothing measurable, and this member costs 16.09 MB
# of a 95 MB budget." So the slot goes back on parsimony, and the best VERIFIED
# configuration is what ships. m58's ~+2 pp fold-ensemble bookkeeping is the
# SEVENTH ResNet-18 result that failed to transfer to s3d.
#
# This was the whole point of submitting 005 with NO predicted LB: a cross-fold
# ensemble is unscoreable on validation by construction (m52), so the board was
# the only instrument that could price it. It has now priced it.
# SUBMISSION 006 (m80): 004's four members PLUS ONE mc3_18 — an ARCHITECTURE
# added to the ensemble, not another copy of the same one. 87.42 MB weighed.
#
# WHY THIS IS A SUBMISSION AND NOT A PROMOTION. mc3_18's marginal fusion gain
# measures +1.049 pp, which MEETS the >= +1 pp gate J8 pre-registered — but
# t = 2.14 on 5 df is NOT significant (crit 2.57), the 95% CI is
# [-0.211, +2.308], and 1.049 sits deep inside m66's +-3 pp fence. Honesty rule 5
# is explicit that the fence outranks the point estimate, so validation does NOT
# settle this and must not be read as though it had.
#
# The third seed the fence asks for costs ~5 h (a seed-3 cell for mc3 AND for
# both s3d branches, x3 folds, since the contrast is paired per cell). One
# submission costs ~2 minutes of GPU and asks an INDEPENDENT instrument. §6
# admits candidates "worth spending to LEARN" and 005 already proved the board
# can settle what validation structurally cannot.
#
# ONLY ONE mc3_18 FITS: 4x s3d (64.38) + 2x mc3 (46.08) = 110.5 MB and
# breaches D6. So this set is asymmetric across folds and, like 005, has NO
# CV comparable in m64's convention — it cannot be the n=2 calibration point.
# FINAL: FOUR MEMBERS. The ensemble is closed, and the board closed it (m83).
#
# The 95 MB cap allows a fifth checkpoint and TWO INDEPENDENT attempts were made
# to spend that slot. Both landed EXACTLY ONE CLIP BELOW this set:
#     004  4x s3d                     0.61691 = 124/201   <- ships
#     005  + a 5th s3d (fold 2)       0.61194 = 123/201
#     006  + one mc3_18 (a different
#          ARCHITECTURE, not a copy)  0.61194 = 123/201
# 005 and 006 differ on 37 of 405 rows and scored identically, so they lost
# different clips — this is a plateau, not one bad member.
#
# One clip cannot prove a fifth member HURTS. It comfortably supports the
# weaker, sufficient claim: no fifth member has produced a measurable gain, from
# either the same architecture or a different one, so the slot stays unspent and
# the best VERIFIED configuration ships. ~7.6 MB of the budget goes unused, on
# purpose.
# REVERTED to the seed-1 set after 007's CSV was built (m84). 004 remains the
# best VERIFIED configuration; 007 is a measurement, not a promotion.
# SUBMISSION 008 (m89/m91): the same four slots, but every member trained on
# ALL 18 subjects rather than a fold's 12 (m87's slope: +4.86 pp single-branch
# predicted), which means the FOLD axis is gone and the diversity comes from two
# SEEDS instead. Measured consequence, m91: member disagreement on the 405 test
# inputs falls from 40.2% (fold axis) to 23.2% (seed axis), so this trades
# ensemble diversity for single-member accuracy and only the board can price it.
#
# 004's four remain the best VERIFIED configuration (0.61691 = 124/201, m75).
# This list points at the CANDIDATE while it is gated and submitted, exactly as
# it pointed at 005's five in m79 — and m79's revert is the precedent if 008
# scores below 004: one commit restores the four names kept below.
#
#   004 (best verified):
#     runs/depthir_f0_alien_j1_s3dk4_w_s1   runs/thermal_f0_alien_j2t_s3dk4_w_s1
#     runs/depthir_f1_alien_j1_s3dk4_w_s1   runs/thermal_f1_alien_j2t_s3dk4_w_s1
# THE ENSEMBLE'S COMBINATION TEMPERATURE (m107). THIS CONSTANT IS PART OF THE
# SUBMITTED ARTEFACT and must never be supplied only by the environment: the
# committee runs inference.sh with no variables set, so an env-only value would
# make the submission unreproducible by the people who verify it. The env var
# exists for EXPERIMENTS; the shipped number is the default written here.
#
# REVERTED TO 1.0 BY THE BOARD (m111), not by validation. T=16 measured
# +1.147 pp on the frozen folds by leave-one-fold-out and LOST A CLIP twice:
# 009 (0.61691) against 008 (0.62189) on identical weights, and 010 (0.63681)
# against 011 (0.64179) on identical weights. Two independent -1 clip
# measurements, same magnitude and sign. The CV gain does not transfer, and
# eight levers have now failed this way (m79's class).
#
#   1.0  = SHIPPED. Reproduces 001-008 and 011, the best verified.
#   16.0 = shipped only in 009 and 010. Softmax at a temperature before averaging,
#          so a confident-but-wrong member cannot dominate. Measured on the
#          frozen folds by LEAVE-ONE-FOLD-OUT: +1.112 pp, sign 3/3; fixed at 16
#          for all folds: +1.147 pp, sign 3/3. The gain plateaus for any T >= 8,
#          which is why 16 (mid-plateau) ships rather than the in-sample argmax
#          of 32. Zero bytes, zero GPU.
#
# To reproduce 008 or earlier from this commit: CUHKX_FUSE_TEMP=1.0
FUSE_TEMP = float(os.environ.get("CUHKX_FUSE_TEMP", "1.0"))

# SUBMISSION 010 (m110): the all-18 members RETRAINED WITH SELF-TRAINING —
# 142 test clips whose own predicted confidence exceeded 0.7 joined the training
# set under this project's own predicted labels (rules-class PSEUDO-LABEL(R7); no
# test LABEL is read anywhere). It scored 0.63681 = 128/201, which is the
# DISTINCTION THRESHOLD met exactly, and is the BEST VERIFIED configuration.
#
# Self-training was worth +4.00 clips measured against 009 with the combiner held
# fixed. The combiner temperature was worth -1.00 clip measured against 008 on
# identical weights. Submitting 010 alone would have attributed +3 to the pair
# and been wrong about both.
#
#   008 (all-18, no pseudo, T=1)  0.62189 = 125/201
#   009 (all-18, no pseudo, T=16) 0.61691 = 124/201
#   010 (all-18, pseudo,    T=16) 0.63681 = 128/201
#   011 (all-18, pseudo,    T=1)  0.64179 = 129/201  <- ships, one clip
#                                                       ABOVE Distinction
# UPDATED 2026-08-20 (m123/Q8) TO THE ir-CSN-R50 SHAPE, AND THE REASON THIS
# LIST MOVED AT ALL IS TO STOP A SILENT DIVERGENCE. `checkpoints/model.pth` (named
# `model.pt` until 2026-08-31, P0 a) is what ships (R4); this list is only the no-container fallback. If the two named
# different ensembles, a fresh clone WITHOUT the container would quietly build a
# different model than the artefact we graded -- the exact class of mismatch m73
# and §J-DQ (c) were both about. So they are kept in agreement by hand.
#
# Shape D: 2 x depthir (seeds 1,2) + 1 x thermal. MEASURED 75.02 MB fp16, legal
# under D6's 95. The symmetric 4-member all-CSN shape MEASURES 100.01 MB --
# over D6's 95 AND over the organisers' own 100 -- so no byte-legal one-lever
# swap against 011 exists, and this asymmetry is forced, not chosen. It is safe
# because Ensemble.predict() averages ONE vector PER BRANCH (branch-balanced
# fusion), so 2-vs-1 members does not tilt the branch weighting.
#
# Superseded (011's members, 64.40 MB, LB 0.64179 = 129/201, m111) -- kept here
# because reproducing 011 must stay one command away:
#   runs/depthir_all18pl_alien_j10_s3dk4_s1  runs/thermal_all18pl_alien_j10_s3dk4_s1
#   runs/depthir_all18pl_alien_j10_s3dk4_s2  runs/thermal_all18pl_alien_j10_s3dk4_s2
# 022 (D20, 2026-08-29): the FENCE shape — ONE ir-CSN-R50 depth+IR member (018's, ALIEN-
# trained) + ONE ir-CSN-152 thermal member (LABPC-trained, T=32, SWAD, temporal crop),
# 83.5 MB fp16. Two R50 depth+IR members + the R152 would be ~109 MB (m199).
# Superseded (018's members, 75.03 MB, LB 0.71144 = 143/201, m137) -- kept so 018 stays
# one command away:
#   runs/depthir_all18_alien_q16_ircsn224_s1  runs/thermal_all18_alien_q16_ircsn224_s1
#   runs/depthir_all18_alien_q16_ircsn224_s2
SHIPPING = ["runs/depthir_all18_alien_q16_ircsn224_s1", "runs/thermal_all18_labpc_r152stack_s1"]

# THE R4 ARTEFACT. The host ruling is that everything loaded at inference --
# "including every model in an ensemble" -- ships as ONE file under 100 MB. The
# packer has written that container since it was first built; until 2026-08-19
# nothing could LOAD it, so `test_inference_offline.sh` passed while reporting
# "loaded 4 checkpoint(s)" and the artefact was compliant on disk and
# non-compliant in the only path a committee runs. `checkpoints/` is gitignored,
# like every other weight in this repo.
# P0 (a), 2026-08-31 (GOAL-SPRINT-3 §4; ledger m219+): the competition site names
# the verification deliverable `checkpoints/model.pth`; this repo's slot was
# `model.pt` since B1. DUAL-ACCEPT, `.pth` PREFERRED: the first name that exists
# is the artefact; with neither on disk the deliverable name is what the messages
# below tell the reader to build. verify.sh and tools/freshclone_rehearsal.sh
# resolve the same two names in the same order.
_CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"
SHIPPED_CONTAINER_NAMES = ("model.pth", "model.pt")


def shipped_container() -> Path:
    """The R4 single-file artefact: checkpoints/model.pth, else the legacy model.pt."""
    for name in SHIPPED_CONTAINER_NAMES:
        if (_CHECKPOINTS / name).is_file():
            return _CHECKPOINTS / name
    return _CHECKPOINTS / SHIPPED_CONTAINER_NAMES[0]


SHIPPED_CONTAINER = shipped_container()


def input_size_of(ck: dict, path: Path) -> tuple[int, int] | None:
    """The (H, W) this member was TRAINED at. None means the cache's native size.

    m73, AND IT REACHED THE SUBMISSION ITSELF. This file resized nothing: it
    fed cache-native 120x160 into checkpoints trained at 168x224, so submission
    004 as first built ran at the wrong scale on every clip. `arch` and `norm`
    were made self-describing by §J-DQ (c) and resolution was not, because when
    that was written there was only one resolution. m69 introduced a second.

    Resolved the same way §J-DQ (c) resolves the others -- from the file first --
    with the run's own `experiments-*.jsonl` record as the fallback for
    checkpoints written before train.py began stamping it. A member whose
    size cannot be established WARNS LOUDLY and falls back to native rather than
    guessing quietly: running at the wrong scale does not raise, it just costs
    ~4 pp, which is exactly how this went unnoticed.
    """
    if ck.get("input_size"):
        sz = ck["input_size"]
        return tuple(int(v) for v in sz.split("x")) if isinstance(sz, str) else tuple(sz)
    repo = Path(__file__).resolve().parent.parent
    for log in sorted(repo.glob("experiments-*.jsonl")):
        for line in log.read_text(encoding="utf-8").splitlines():
            if path.parent.name not in line:
                continue
            rec = json.loads(line)
            if rec.get("run") != path.parent.name:
                continue
            sz = rec.get("hparams", {}).get("input_size")
            return tuple(int(v) for v in sz.split("x")) if sz else None
    print(f"[predict] ⚠ {path.parent.name}: no stamped input_size and no log record — "
          f"assuming the cache's native size. If this member was trained at another "
          f"resolution it is now running at the wrong scale.", file=sys.stderr)
    return None


def _members(ckpts, torch):
    """Yield (member, path, packed) from EITHER artefact shape.

    R4 (host ruling, on the record since 08-12): everything loaded at
    inference must ship as ONE checkpoint file under 100 MB. `package.py`'s
    `pack()` has always WRITTEN that single container -- a list of member dicts
    -- but nothing could READ it: this class took a list of paths and loaded a
    run checkpoint from each, which is why `test_inference_offline.sh` reported
    "loaded 4 checkpoint(s)" in its own PASS line. The artefact was compliant on
    disk and non-compliant in the only path that matters.

    A container is a `list`; a run checkpoint is a `dict`. Nothing else about
    loading changes, because `pack()` writes members carrying the same keys this
    loader already reads. Both shapes stay supported on purpose: the four-file
    form is the dev path every run directory produces, and the container is what
    ships.
    """
    for p in ckpts:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, list):
            for mem in obj:
                yield mem, p, True
        else:
            yield obj, p, False


def _member_input_size(ck: dict, path: Path, packed: bool):
    """As input_size_of, but a CONTAINER member must say its own resolution.

    input_size_of's fallback reads the run's `experiments-*.jsonl` record keyed
    on `path.parent.name`. For a container that name is a directory, so the
    fallback silently yields native. On the dev path that warning is survivable;
    on the artefact that SHIPS it is m73 again, unnoticed, at ~4 pp. So a packed
    member without the key REFUSES rather than guessing. `None` is a legitimate
    value (it means the cache's native size) -- absence is not, which is why this
    tests key presence and not truthiness.
    """
    if packed:
        if "input_size" not in ck:
            raise ValueError(
                f"{path}: packed member {ck.get('run', '?')!r} carries no input_size. "
                f"Repack with a package.py that stamps it — refusing to guess, because "
                f"a wrong resolution does not raise, it silently costs ~4 pp (m73)."
            )
        sz = ck["input_size"]
        if sz is None:
            return None
        return tuple(int(v) for v in sz.split("x")) if isinstance(sz, str) else tuple(sz)
    return input_size_of(ck, path)


def natural_key(name: str) -> tuple[int, str]:
    """Sort SM_test_9 before SM_test_10 even if the padding ever changes."""
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else -1, name)


def find_row_keys(data_dir: Path, test_csv: Path | None) -> tuple[list[str], str]:
    """Return (ordered path strings, provenance).

    Prefers the organisers' row list. Falls back to walking the tree.
    """
    candidates = []
    if test_csv is not None:
        candidates.append(test_csv)
    candidates += [
        data_dir / "test.csv",
        data_dir / "test_file" / "test.csv",
        data_dir.parent / "test_file" / "test.csv",
        data_dir.parent / "Testing" / "test_file" / "test.csv",
        data_dir / "sample_submission.csv",
        data_dir / "test_file" / "sample_submission.csv",
        data_dir.parent / "test_file" / "sample_submission.csv",
    ]

    for path in candidates:
        if not path.is_file():
            continue
        # utf-8-sig: a BOM turns the first header into "﻿path" and every
        # lookup by name then fails. This corpus has BOM'd CSVs elsewhere.
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        keys = [r["path"] for r in rows if r.get("path")]
        if keys:
            return keys, f"row keys from {path}"

    # Fallback: discover clip directories.
    clip_dirs = [p for p in data_dir.iterdir() if p.is_dir() and CLIP_DIR_PATTERN.match(p.name)]
    container = data_dir
    if not clip_dirs:
        for child in sorted(data_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("__"):
                continue
            found = [p for p in child.iterdir() if p.is_dir() and CLIP_DIR_PATTERN.match(p.name)]
            if found:
                clip_dirs, container = found, child
                break

    if not clip_dirs:
        raise SystemExit(f"no SM_test_* clip directories found under {data_dir}")

    clip_dirs.sort(key=lambda p: natural_key(p.name))
    keys = [f"{container.name}/{p.name}/" for p in clip_dirs]
    return keys, f"row keys discovered from {container} (no test.csv found)"


def clip_dir_for(data_dir: Path, row_key: str) -> Path | None:
    """Map a submission path string back to a directory on disk, or None."""
    rel = row_key.strip().strip("/")
    for base in (data_dir, data_dir.parent):
        candidate = base / rel
        if candidate.is_dir():
            return candidate
    # The row key carries a container name that may not match this data_dir.
    leaf = rel.rsplit("/", 1)[-1]
    for base in (data_dir, *(p for p in data_dir.iterdir() if p.is_dir())):
        candidate = base / leaf
        if candidate.is_dir():
            return candidate
    return None


# §119 (D50; P-A) — THE FUSION FORMULA IS THE SECOND STAMPED KEY, BESIDE `crops`, AND ITS DEFAULT IS INERT.
# The shipped decode fuses the branches by the ARITHMETIC mean of their probability vectors (`Ensemble.predict`
# below, `:600` before this landed). §118 measured a family of one-parameter alternatives on the six cached 022
# cells and named ONE: the logit-adjusted geometric pool  s = 0.5·z_A + 0.5·z_B − λ·log π  (z = log p; π a TRAIN
# prior, never a test statistic — T-L7, D14 (d)). A formula is not a weight, so it cannot travel in the tensors:
# it travels as stamped CONTAINER METADATA, exactly as `crops` does, and predict.py reads it here.
#
#   absent, or None   ⇒ the arithmetic mean — BITWISE the pre-§119 path (tests/check_fusion_key.py proves it on
#                       synthetic tensors AND on the real decode path; the shipped 022 container carries no key).
#   a dict            ⇒ {"form": "logit_adjusted_geo", "lambda": float ≥ 0, "prior": [40 floats > 0]}.
#   anything else     ⇒ REFUSED. A fusion rule this code does not know is not defaulted, it stops — the `crops`
#                       precedent, and for the same reason: a wrong fusion does not raise, it silently decides.
#
# Applied ONLY where both branches are present. A clip seen by ONE branch passes that branch's probabilities
# through unchanged and is NOT prior-adjusted — the registered missing-modality convention
# (tools/p11_desk_closures.py:80-95, replicated in tools/p15_fusion_family.py, which is where the twin's numbers
# come from). Per clip, stateless, no test input read: T-L7 holds for the twin exactly as for the incumbent.
FUSION_FORMS = ("logit_adjusted_geo",)
FUSION_LOG_FLOOR = 1e-300          # tools/p15_fusion_family.py's `log_` floor, verbatim; a no-op on these probabilities


def parse_fusion(raw, where: str):
    """Normalise a stamped `fusion` value to a hashable spec, or None for the inert default.

    Returns None (absent/None) or ("logit_adjusted_geo", lam, (prior...)). Raises ValueError on anything else, so a
    malformed stamp can never be a silent passenger — the §96 depth_lut and §105 crops precedent.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: the fusion stamp is {type(raw).__name__}, not a dict — refusing to guess")
    unknown = set(raw) - {"form", "lambda", "prior"}
    if unknown:
        raise ValueError(f"{where}: the fusion stamp carries unknown key(s) {sorted(unknown)} — refusing")
    form = raw.get("form")
    if form not in FUSION_FORMS:
        raise ValueError(f"{where}: fusion form {form!r}; only {list(FUSION_FORMS)} exist — refusing to guess")
    try:
        lam = float(raw["lambda"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{where}: the fusion stamp has no readable 'lambda' — refusing") from None
    if not (lam == lam and lam != float("inf") and lam >= 0.0):     # NaN and inf are not a λ
        raise ValueError(f"{where}: fusion lambda={lam!r} is not a finite λ ≥ 0 — refusing")
    prior = raw.get("prior")
    if prior is None:
        raise ValueError(f"{where}: the fusion stamp has no 'prior' — refusing")
    prior = tuple(float(x) for x in prior)
    sys.path.insert(0, str(Path(__file__).resolve().parent))     # the same one-liner every method here uses
    from model import NUM_CLASSES  # noqa: PLC0415
    if len(prior) != NUM_CLASSES:
        raise ValueError(f"{where}: the fusion prior has {len(prior)} entries, not {NUM_CLASSES} — refusing")
    if not all(x > 0.0 and x == x and x != float("inf") for x in prior):
        raise ValueError(f"{where}: the fusion prior has a non-positive or non-finite entry — log π is undefined; refusing")
    return (form, lam, prior)


class Ensemble:
    """Every checkpoint that votes, grouped by branch. Built once, used per clip.

    R8: THE WHOLE PIPELINE RUNS HERE, FROM THE ORIGINAL FILES. Nothing is
    precomputed and the cache is not read -- the committee runs this on a hidden
    set that has no cache. Decode, resize, dead-strip, normalise and sample all
    happen per clip, using the SAME functions the training path used
    (preprocess.frame listing + build_clip, dataset._tsn_indices + NORM), never a
    reimplementation of them. A second copy of the preprocessing that drifts from
    the first is the failure mode this costs a little speed to avoid.

    WHAT VOTES, AND WHY. RE-MEASURED A SECOND TIME, ON ir-CSN-R50 (m141, R1).
    This block has now been re-derived across TWO backbone pivots, and the reason
    is the same both times: this docstring is on the SHIPPING path and is read as
    a justification for what votes, so it must describe the backbone that ships.
    On 2026-08-22 that is ir-CSN-R50 @ 224^2 / bs12 / lr 5e-4, not s3d.

    Measured on the 6-cell grid -- 3 SUBJECT-DISJOINT folds x {depthir, thermal},
    one seed per branch per fold (tools/r1_ensemble_cv.py; QUEUE.md section 8 is
    the pre-registration, written before the passes ran):

        2 branches  +3.084 pp vs the BETTER branch of that fold, sd 1.176,
                    t +4.54 on 2 df, sign 3/3, CI [+0.164, +6.005].
                    Against a FIXED branch: +3.296 pp vs depthir (3/3, CI
                    [-0.012, +6.603], n.s. at 2 df) and +5.071 pp vs thermal
                    (3/3, CI [+1.648, +8.495]).
                    "Better branch" is picked per fold on the same data it is
                    scored on, so that baseline is the max of two noisy estimates
                    and is biased UP -- the +3.084 is an UNDER-estimate.
        2 seeds      UNMEASURED ON THIS BACKBONE. The grid carries ONE seed per
                    branch, so R1 cannot price it. s3d's +0.57 pp does NOT
                    transfer by assumption -- that is the whole lesson of the
                    table in RESULTS.md. The shipped artefact carries two depthir
                    seeds anyway, on the standard-and-free argument, not a
                    measured one.
        2 flips     +0.809 pp, sd 0.729, t +2.72 on 5 df, sign 5/6,
                    CI [+0.043, +1.574] -- pooled over all 6 branch x fold cells.
                    SO IT IS NOT THE NULL IT IS ON s3d. Significant, and
                    still INSIDE m66's +-3 pp fence: real and small.
    Together 65.831% (one member, one branch, upright) -> 69.381% (full stack) on
    the UNION of clips with >=1 branch, which is the population this file faces on
    test; 69.510% on the intersection. Per fold: 71.311 / 69.152 / 67.680.

    SUPERSEDED, kept visible so a reader can see what was believed and that it was
    corrected -- and because tools/seed_axis_cv.py cross-references the s3d chain:
        on s3d (m72/m74): branches +4.19 [+3.18, +5.20] 6/6 · seeds +0.57 ·
        flips +0.15 sd 0.37 3/6 p = 1.000 (a NULL there) · 0.58201 -> 0.62870.
        on ResNet-18 (m49/51/52): seeds +1.41 · branches +2.56 · flips +0.84 ·
        0.44889 -> 0.48117.

    hflip TTA STAYS ON, and the reason is STILL bytes first: it reuses the same
    weights, so it costs two forward passes and ZERO of D6's 95 MB. What changed
    is that on ir-CSN-R50 it is no longer free-and-worthless -- it is free and
    worth about +0.8 pp. Nothing here may claim m51's +0.84 pp, which is a
    different backbone that happens to land on a similar number.

    FOLD CHECKPOINTS ARE A FOURTH AXIS THAT VALIDATION CANNOT PRICE. All three
    folds trained on data disjoint from TEST, so all three may vote here; but
    fold f's model saw fold g's val subjects, so their gain cannot be measured on
    val and is NOT included in the 69.381%. Passing all folds is a variance
    argument, not a measured one. It is the default because it is free and
    standard; it is labelled here so the report cannot quietly claim otherwise.

    AND THE 69.381% IS THE RECIPE'S, NOT THIS ARTEFACT'S. checkpoints/model.pt
    is trained on all 18 subjects and so has no held-out subjects left to score
    against (m87 (4)); the grid above is fold-trained on 12. The number describes
    the SHAPE that ships, which is the most that can honestly be said.
    """

    def __init__(self, ckpts: list[Path], device_pref: str = "auto") -> None:
        import numpy as np  # noqa: F401  (imported for the per-clip path)
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dataset import NORM
        from model import DEFAULT_ARCH, adapt_kw, build, norm_key

        self.torch = torch
        self.device = torch.device(
            "cuda" if (device_pref != "cpu" and torch.cuda.is_available()) else "cpu"
        )
        # Members carry their own (arch, norm) -- see the §J-DQ (c) note below.
        self.by_branch: dict[str, list[dict]] = {}
        self.lut: dict[str, "np.ndarray"] = {}   # §96 (a)/(b): branch -> the stamped depth LUT (uint8 (K, 3))
        self.T: dict[str, int] = {}
        # §124 (P-A): branch -> the stamped SOURCE (W, H) for the decode, or None = build_clip's own table (the
        # shipped 160x120 path). Recorded for EVERY member, including the absent case, so that a branch whose
        # members disagree — stamped vs unstamped included — is refused rather than decided by load order.
        self.source_wh: dict[str, tuple[int, int] | None] = {}
        containers: dict[Path, int] = {}
        fusions: list = []                       # §119 (P-A): every member's stamped fusion spec; they must agree
        for ck, p, packed in _members(ckpts, torch):
            branch = ck["branch"]
            if packed:
                containers[p] = p.stat().st_size

            # §J-DQ (c). A checkpoint must say what it is. Until 2026-08-15
            # none did, and predict.py inferred the architecture by being the
            # only one there was and the normalisation from the branch name.
            # J1 puts two of each in play, so both are now read off the file.
            #
            # ABSENT means "written before 2026-08-15", and every such file on
            # disk IS the legacy pair -- that is a fact about this repository,
            # not an assumption, and check_ckpt_meta.py asserts it over runs/.
            # So absence resolves to the legacy default and is REPORTED; a value
            # that is present but unrecognised RAISES, because that is a
            # checkpoint from a world this code does not know about.
            arch = ck.get("arch", DEFAULT_ARCH)
            legacy = "arch" not in ck
            norm = ck.get("norm") or norm_key(branch, arch)
            if norm not in NORM:
                raise ValueError(
                    f"{p}: checkpoint declares norm {norm!r}, which is not a key of "
                    f"dataset.NORM {sorted(NORM)}. Refusing to guess — a wrong "
                    f"normalisation does not raise, it silently costs accuracy."
                )

            # pretrained=False: the state dict overwrites every weight anyway, and
            # the True path would reach for ImageNet weights over the network on a
            # machine that may have none. (Now also the default — §J-DQ (b).)
            # §96 (a)/(b) (D29): an ordinal-depth member needs its `depth_lut` stamp (a fold-TRAIN statistic; never a repo constant) —
            # _frames decodes the ORIGINAL frames through the SAME build_clip path the cache used, with this table (R8; T-L7); a
            # member of any other branch must NOT carry one (a misplaced stamp is refused, never a silent passenger).
            from dataset import ORD_BRANCHES
            import numpy as _np
            lut = ck.get("depth_lut")
            if branch in ORD_BRANCHES and lut is None:
                raise ValueError(f"{p}: an ordinal-depth member ({branch}) needs its depth_lut stamp (§96 a/b) — refusing")
            if branch not in ORD_BRANCHES and lut is not None:
                raise ValueError(f"{p}: member {ck.get('run', '?')!r} carries a depth_lut but is not an ordinal-depth branch — a misplaced stamp; refusing")
            if lut is not None:
                lut_np = _np.asarray(lut.cpu().numpy() if hasattr(lut, "cpu") else lut, dtype=_np.uint8)
                if branch in self.lut and not _np.array_equal(self.lut[branch], lut_np):
                    raise ValueError(f"{p}: the members of branch {branch} carry DIFFERENT depth_lut tables — refusing")
                self.lut[branch] = lut_np
            # §105 (D34; P-A): the CROP COUNT is stamped checkpoint metadata — absent or None ⇒ 1 (the shipped single view; the
            # decode path below is then BITWISE the pre-§105 path, tests/check_crop_tta.py); 3 ⇒ the 3-crop spatial TTA (three
            # deterministic 224² crops of a 256² resize, each hflip-paired, six views averaged). Any other value is REFUSED — a
            # crop count this code does not know is not defaulted, it stops.
            crops = ck.get("crops")
            crops = 1 if crops is None else int(crops)
            if crops not in (1, 3):
                raise ValueError(f"{p}: member {ck.get('run', '?')!r} declares crops={crops!r}; only 1 (the shipped view) and 3 (§105) exist — refusing to guess")
            # §124 (P-A): the THIRD stamped decode key — the SOURCE resolution this member's frames were decoded at
            # BEFORE the resize to input_size. ABSENT ⇒ None ⇒ preprocess.build_clip's own STREAMS table (thermal
            # 160x120, depthir 160x120) — the shipped path, bitwise (tests/check_source_wh.py §1). PRESENT ⇒ that
            # (W, H), fed to the SAME build_clip/_open_resized code src/preprocess.py --size-wh WxH used to build the
            # cache the member trained on. This is not cosmetic: a §124 member trained on frames decoded at 320x240
            # and resized to 320x320 sees a DIFFERENT pixel distribution if inference decodes at 160x120 first, and
            # nothing downstream would raise — it is m73's failure with a different number.
            swh = ck.get("source_wh")
            if swh is not None:
                swh = tuple(int(v) for v in swh)
                if len(swh) != 2 or min(swh) < 1:
                    raise ValueError(f"{p}: member {ck.get('run', '?')!r} declares source_wh={ck.get('source_wh')!r}; it must be (W, H), both ≥ 1 — refusing to guess")
            # The members of ONE branch decode TOGETHER through one build_clip call, so they must agree — including
            # one stamped and one not, which is a disagreement between (W, H) and the table, not a missing option.
            if branch in self.source_wh and self.source_wh[branch] != swh:
                raise ValueError(f"{p}: the members of branch {branch} carry DIFFERENT source_wh stamps "
                                 f"({self.source_wh[branch]!r} vs {swh!r}) — one branch has one decode; refusing")
            self.source_wh[branch] = swh
            # §119 (D50; P-A): the SECOND stamped key. Absent ⇒ None ⇒ the arithmetic mean, bitwise the pre-§119 path.
            # The members of ONE artefact decode together, so a container whose members disagree about the fusion
            # formula has no single decode rule and is refused rather than resolved by position.
            fusions.append(parse_fusion(ck.get("fusion"), f"{p}: member {ck.get('run', '?')!r}"))
            m = build(branch, n_segment=ck["T"], arch=arch, pretrained=False, **adapt_kw(ck))   # §63
            # D27/D28: a torch.ao int8-STORAGE container (tools/package_int8.py) carries quantised weight
            # tensors; they are dequantised HERE, at load, into the same fp32 module -- the arithmetic
            # downstream is unchanged and stateless (T-L7). An fp16/fp32 container has no quantised tensor
            # and takes this line untouched.
            m.load_state_dict({k: (v.dequantize() if getattr(v, "is_quantized", False) else v)
                               for k, v in ck["model"].items()})
            m.eval().to(self.device)
            self.by_branch.setdefault(branch, []).append(
                {"model": m, "arch": arch, "norm": norm, "legacy": legacy, "path": p, "quant": ck.get("quant"),
                 "input_size": _member_input_size(ck, p, packed), "packed": packed, "crops": crops,
                 # A container's `path.parent.name` is a directory, not a run, so
                 # the load report below would name the same thing four times.
                 # pack() stamps the originating run; prefer it.
                 "run": ck.get("run") or p.parent.name}
            )
            self.T[branch] = ck["T"]
        self.n = sum(len(v) for v in self.by_branch.values())
        # §119 (P-A): one artefact, one fusion rule.
        if len(set(fusions)) > 1:
            raise ValueError(f"the members of this artefact declare DIFFERENT fusion stamps {sorted(set(map(str, fusions)))} — "
                             f"an artefact has ONE decode rule; refusing")
        self.fusion = fusions[0] if fusions else None
        self.fusion_lam = 0.0 if self.fusion is None else self.fusion[1]
        self.fusion_logpi = None if self.fusion is None else np.log(np.asarray(self.fusion[2], dtype=np.float64))
        # D28: a CONTAINER is weighed by its bytes on disk -- the object the cap applies to -- never by
        # params x 2, which overstates an int8-storage file by ~4x and would refuse a legal artefact.
        self.all_packed = bool(containers) and all(mem["packed"] for v in self.by_branch.values() for mem in v)
        self.packed_mb = sum(containers.values()) / 1e6 if containers else 0.0
        # params x 2 IS AN ESTIMATE, AND UNTIL 2026-08-15 NOBODY HAD WEIGHED
        # THE FILE IT ESTIMATES. A state dict also carries BN buffers --
        # running_mean and running_var halve, num_batches_tracked is int64 and
        # does not -- plus torch.save's zip container and a pickle header per
        # tensor. Measured with tools/package.py on the real artefact: 44.91 MB
        # against 44.79 predicted (2 members), 89.82 vs 89.59 (4), 134.73 vs
        # 134.38 (6). A consistent +0.3% for ResNet-18.
        #
        # THE OVERHEAD IS BACKBONE-DEPENDENT AND GROWS WITH BN COUNT: a
        # reviewer's 6-member s3d package measured 96.38 MB against 95.4
        # predicted (+1.03%), because s3d carries 77 BatchNorm3d to ResNet-18's
        # 20. That one BREACHES D6's 95 MB while its estimate passed. So the
        # guard applies the measured factor rather than the theoretical one --
        # an estimate that is 1% optimistic about a hard ceiling is the failure
        # standing rule 11 exists for.
        self.fp16_mb = sum(sum(q.numel() for q in mem["model"].parameters())
                           for v in self.by_branch.values() for mem in v) * 2 / 1e6 * 1.005

    # ── one clip, from the original files ───────────────────────────────────
    def _frames(self, clip_dir: Path, branch: str):
        """(T, H, W, C) uint8 for one branch, already sampled. None if absent.

        DECODES ONLY THE T FRAMES TSN WILL USE. The cache build decoded whole
        clips because it stored them; here they are consumed immediately, and a
        640x480 PNG costs 2-5 ms. The median depth clip is 24 frames against T=8,
        the median thermal clip 49 -- so decoding everything spent 2/3 to 5/6 of
        the wall clock on pixels that are thrown away. Measured: 2.76 s/clip
        before, and 405 clips did not finish in 10 minutes.

        THE REPAIR PATH IS THE CATCH, AND IT IS WHY THIS FALLS BACK RATHER
        THAN JUST SUBSAMPLING. build_clip repairs a bad frame from its
        neighbours; over a subset it would repair from the wrong neighbours and
        silently produce pixels the training path never would. So: sample first,
        and if ANY sampled frame came back bad, redo over the whole clip, which
        is exactly what the cache did. Fast in the common case, identical in the
        rare one -- 122 unreadable files exist in 4 test clips.
        """
        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dataset import _tsn_indices
        from preprocess import build_clip, depthir_frames, thermal_frames

        from dataset import ORD_BRANCHES
        stream = "thermal" if branch == "thermal" else ("depthir_ord" if branch in ORD_BRANCHES else "depthir_raw")
        lut = self.lut.get(branch)          # §96 (a)/(b): the member's stamped table; None for every other branch
        if branch == "thermal":
            paths, _ = thermal_frames(clip_dir / "Thermal")
            paths = [(p,) for p in paths]
        else:
            ir, dp, _ = depthir_frames(clip_dir / "IR", clip_dir / "Depth_Color")
            paths = list(zip(ir, dp))
        if not paths:
            return None

        # §124 (P-A): the member's stamped SOURCE size, or None = build_clip's table (the shipped decode, bitwise).
        swh = self.source_wh.get(branch)
        idx = _tsn_indices(len(paths), self.T[branch], train=False, rng=None)
        frames, stats = build_clip([paths[i] for i in idx], stream, lut=lut, size_wh=swh)
        if stats["bad"]:
            frames, stats = build_clip(paths, stream, lut=lut, size_wh=swh)  # exact path; repair needs neighbours
            frames = frames[idx] if len(frames) else frames
        # Trap 3: a channel group with no good frame anywhere is not an input.
        # Depth is 3 of depthir's 4 channels and is what the branch requires.
        if "depth" in stats["absent"] or "rgb" in stats["absent"]:
            return None
        return np.asarray(frames)

    def _probs(self, frames, branch: str):
        """Mean softmax over every member of one branch, upright and flipped."""
        import numpy as np
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dataset import NORM, ORD3_PLANES, ORIGINAL_WH, depth_validity

        # _frames already applied the TSN sampling, because it decodes only what
        # it samples. Asserted rather than re-sampled: _tsn_indices(T, T) happens
        # to be the identity, so a second call would look correct and silently
        # stop being so the moment T or the sampler changed.
        T = self.T[branch]
        if len(frames) != T:
            raise RuntimeError(f"{branch}: expected {T} sampled frames, got {len(frames)}")
        raw = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).float().div_(255.0)
        if branch == "depthir_ord3":
            raw = raw[:, list(ORD3_PLANES)]     # §96 (a): [ord, ord, ord, IR], exactly as the loader served it

        if branch != "thermal":
            # Trap 4: 40 ORIGINAL px rescaled to the cache width, depth planes only.
            cols = int(round(40 * raw.shape[-1] / ORIGINAL_WH["depthir"][0]))
            raw[:, 0:3, :, :cols] = 0.0

        # NORMALISATION IS PER MEMBER, NOT PER BRANCH (§J-DQ (c)). It was
        # keyed on the branch name until 2026-08-15, which is correct exactly
        # while one architecture is in play. J1 introduces a second pretraining
        # corpus and therefore a second set of statistics, and a member fed the
        # wrong ones does not raise -- it just gets quietly worse. Members are
        # grouped by their declared norm so the tensor is built once per distinct
        # normalisation, and the branch's own weighting is unchanged: the mean
        # below is over every member of the branch regardless of grouping.
        # m73: RESOLUTION IS A SECOND GROUPING AXIS, for the same reason
        # normalisation is one. Members trained at different sizes need
        # different tensors, and feeding one the other's scale is silent.
        members = self.by_branch[branch]
        groups: dict[tuple[str, tuple | None, int], list] = {}
        for mem in members:
            groups.setdefault((mem["norm"], mem["input_size"], mem["crops"]), []).append(mem["model"])

        out = []
        with torch.no_grad():
            for (key, size, crops), models in groups.items():
                mean = torch.tensor(NORM[key]["mean"]).view(1, -1, 1, 1)
                std = torch.tensor(NORM[key]["std"]).view(1, -1, 1, 1)
                # §96 (c) (D29): a member under a 5-statistic key takes the depth-validity plane as its fifth channel — the
                # SAME helper the loader used, on the same raw tensor after the strip, BEFORE normalisation (then the resize
                # below treats it as the loader did). Per clip, stateless (T-L7). A 4-statistic key takes this line untouched.
                xr = raw if mean.shape[1] == raw.shape[1] else torch.cat([raw, depth_validity(raw)], dim=1)
                xn = (xr - mean) / std
                # AFTER normalisation, bilinear, align_corners=False — the exact
                # order and interpolation dataset.py uses, so a packaged member
                # sees what it saw in training. A different interpolation here
                # would be a second silent degradation.
                if crops == 3:
                    # §105 (D34; P-A): the 3-crop spatial TTA — the normalised frames resized to 256² (the same bilinear call,
                    # align_corners=False), three deterministic 224² crops at x ∈ {0, 16, 32} (rows 16:240), each flipped: six
                    # views in one batch, averaged below exactly as the two views are. Per clip, no RNG (T-L7). Reached ONLY by a
                    # member stamped crops=3; a crops=1 member takes the branch below, untouched.
                    if tuple(size or ()) != (224, 224):
                        raise RuntimeError(f"{branch}: crops=3 needs a 224x224 member, got {size}")
                    x256 = torch.nn.functional.interpolate(xn, size=(256, 256), mode="bilinear", align_corners=False)
                    xs = torch.stack([x256[..., 16:240, x0:x0 + 224] for x0 in (0, 16, 32)]).to(self.device)
                    xb = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=0)
                else:
                    if size is not None and tuple(xn.shape[-2:]) != tuple(size):
                        xn = torch.nn.functional.interpolate(
                            xn, size=tuple(size), mode="bilinear", align_corners=False)
                    x = xn.unsqueeze(0).to(self.device)
                    # Upright and flipped in one batch of 2 -- measurement 51's TTA.
                    xb = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)
                for m in models:
                    logits = m(xb)
                    # FUSE_TEMP (m107). Softmax at a TEMPERATURE before averaging.
                    # T=1 is the historical behaviour and reproduces every prior
                    # submission byte-for-byte. T>1 flattens each member first, so a
                    # confident-but-wrong member cannot dominate the average — and at
                    # high T the whole stack effectively fuses at LOGIT level, which
                    # is where the gain comes from. Measured on the frozen folds by
                    # LEAVE-ONE-FOLD-OUT (not argmax-over-all): +1.112 pp, sign 3/3,
                    # on 008's k=4 shape. Costs zero bytes and zero GPU.
                    out.append(torch.softmax(logits.float() / FUSE_TEMP, dim=1)
                               .mean(0).cpu().numpy())
        return np.mean(out, axis=0)

    def predict(self, clip_dir: Path) -> int | None:
        """action_id, or None when no branch could be built for this clip."""
        import numpy as np

        got = []
        for branch in self.by_branch:
            frames = self._frames(clip_dir, branch)
            if frames is None or not len(frames):
                continue
            got.append(self._probs(frames, branch))
        if not got:
            return None
        # Renormalise over the branches actually PRESENT (dataset.py trap 3):
        # 4 test clips have no IR and 10 no thermal, and the intersection is
        # empty, so every clip can feed at least one branch. Averaging over what
        # is present beats feeding a zero plane the network never saw.
        #
        # §119 (D50; P-A): with no fusion stamp — the shipped 022 container — this is the WHOLE decode and the
        # line below is the one that has always been here. A stamped artefact, and only where BOTH branches spoke,
        # takes the logit-adjusted geometric pool instead: s = mean(log p) − λ·log π, argmax. For two branches
        # mean(log p) IS 0.5·z_A + 0.5·z_B bitwise (halving is exact in binary), which is the arithmetic
        # tools/p15_fusion_family.py measured; the widening to float64 matches that desk exactly, and the
        # single-branch clip below never reaches it (the registered pass-through convention).
        if self.fusion is None or len(got) < 2:
            return int(np.mean(got, axis=0).argmax())
        z = np.log(np.maximum(np.asarray(got, dtype=np.float64), FUSION_LOG_FLOOR))
        return int((z.mean(axis=0) - self.fusion_lam * self.fusion_logpi).argmax())


def predict_clip(clip_dir: Path | None, ensemble: "Ensemble | None" = None) -> int:
    """Predict one clip's action_id.

    THE CONSTANT IS THE PER-CLIP FLOOR, AND ONLY THAT. When this clip has no
    usable modality it returns the majority class rather than raising, because a
    traceback on the committee's machine costs the whole submission and a wrong
    class costs one clip (0.4975 pp of the public LB). That trade is only
    favourable per clip. `ensemble is None` no longer reaches here on the default
    path -- main() exits 1 first (§J-DQ (a)) -- because 405 fallbacks are not 405
    cheap losses, they are a disqualification wearing a valid CSV.
    """
    if ensemble is None or clip_dir is None:
        return FALLBACK_ACTION_ID
    out = ensemble.predict(clip_dir)
    return FALLBACK_ACTION_ID if out is None else out


def find_checkpoints(spec: list[str]) -> list[Path]:
    """Resolve --checkpoints entries to swa.pt files, deterministically ordered.

    SWA ONLY. A run directory holds `best.pt` AND `swa.pt`, and SWA is the
    recipe's checkpoint selector -- best-epoch is the control arm of a settled
    A/B, not a member. An earlier version globbed `*` inside a run directory and
    silently returned BOTH, so a two-checkpoint request loaded four and reported
    89.6 MB against an expected 44.8. It looked like a working ensemble.
    """
    found: list[Path] = []
    for s in spec:
        p = Path(s)
        if p.is_file():
            found.append(p)
            continue
        if p.is_dir():
            if (p / "swa.pt").is_file():
                found.append(p / "swa.pt")
            continue
        # Only now is it a glob: expand it and take each match's swa.pt.
        base = p.parent if not p.parent.is_absolute() else p.parent
        root = base if base.is_absolute() else Path.cwd() / base
        for d in sorted(root.glob(p.name)):
            if d.is_dir() and (d / "swa.pt").is_file():
                found.append(d / "swa.pt")
            elif d.is_file() and d.suffix == ".pt":
                found.append(d)
    return sorted(set(found))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path, help="directory holding the test clips")
    ap.add_argument("-o", "--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--test-csv", type=Path, default=None, help="explicit path to the organisers' test.csv")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="globs of run dirs or .pt files; default is every run "
                         "matching the shipping configuration under runs/")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--max-mb", type=float, default=95.0,
                    help="packaged fp16 budget, D6. 11.20 M params = 22.4 MB per checkpoint")
    ap.add_argument("--allow-oversize", action="store_true",
                    help="measure an ensemble that cannot ship — never for a submission")
    # §J-DQ (a). This was `--require-model`, OFF by default, with the comment
    # "for CI, never for the committee's run, where the constant floor must
    # survive anything." That reasoning is right about a CLIP and wrong about a
    # RUN, and the polarity is now inverted to match. A whole-run constant is not
    # a floor -- it scores 0.10945, fails reproduction by >10%, and exits 0 with
    # a well-formed CSV, so it is a disqualification that no downstream check can
    # see. The escape hatch stays, because measuring the floor deliberately is a
    # real thing to want; it just is not the default any more.
    ap.add_argument("--allow-constant-fallback", action="store_true",
                    help="emit the constant class instead of failing when NO model can be "
                         "loaded at all. Scores the 0.10945 sanity floor — never for a "
                         "submission; per-clip fallback is unaffected and always on")
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir is not a directory: {data_dir}")

    row_keys, provenance = find_row_keys(data_dir, args.test_csv)
    print(f"[predict] {provenance}", file=sys.stderr)
    print(f"[predict] {len(row_keys)} rows", file=sys.stderr)

    # MODEL LOADING IS FATAL BY DEFAULT (§J-DQ (a), 2026-08-15). It used to be
    # best-effort: no torch, no checkpoint or a state dict from another
    # architecture all degraded to the constant and exited 0. That is the
    # project's own method lesson 3 -- "a silent drop is worse than a crash" --
    # violated at the one level where the drop costs everything rather than one
    # clip. PER-CLIP failure handling below is untouched and still never raises.
    # THE SHIPPING SET, AND IT IS DELIBERATELY SMALL. `ab5_lrhi` is lr 0.005,
    # the settled value (measurement 53); the old `ab2_tsmon` runs are lr 0.0025
    # and are beaten by ~1.3 pp at the stack level. Measurement 56: at the settled
    # lr, k=2 scores 48.792% and k=4 49.452% -- a 0.66 pp gap on 3 folds, which
    # flipped sign between the two learning rates, i.e. noise. Two checkpoints is
    # chosen because it is 44.8 MB against D6's 95 MB and leaves half the budget
    # for a DIVERSE member, which measurement 55's flat curve says is worth more
    # than a third seed. For the 4-checkpoint variant pass:
    #   --checkpoints 'runs/depthir_f0_alien_ab5_lrhi_s[12]' 'runs/thermal_f0_alien_ab5_lrhi_s[12]'
    # hflip TTA doubles the votes either way and spends zero bytes. The budget
    # guard below refuses anything that could not ship.
    # TWO FOLDS, NOT ONE — 2026-08-15, AND IT IS A BOOKKEEPING FIX FIRST.
    # Submission 002 shipped fold 0 alone while the project quoted 0.48792, a
    # THREE-FOLD MEAN. Those are different objects and the difference is now
    # measured: per-fold k=2 at lr 0.005 is f0 46.568 · f1 49.068 · f2 50.740,
    # so fold 0 sits 2.22 pp below its own 3-fold mean. Of the 7.00 pp CV↔LB gap
    # (48.792 quoted vs 41.791 scored), 2.22 pp was never generalisation at all
    # — it was quoting a number for a configuration we did not ship.
    #
    # THE 0.711 TEAM SHIPS EXACTLY TWO FOLDS, equal weights, logit average
    # (BASELINE-0711.md §4), and quotes the mean of those same two folds. Their
    # CV tracked their LB to 0.6 pp. Ours did not, and this is the difference.
    #
    # The honest CV for THIS set is the 2-fold mean, 47.818 = (46.568+49.068)/2.
    # It is not a measurement OF the shipped object: fold f's model saw fold
    # g's val subjects, so a cross-fold ensemble cannot be scored on validation
    # at all (see the Ensemble docstring). It is the same construct the exemplar
    # uses, quoted the same way, which is the most that is available here.
    #
    # 89.82 MB fp16 MEASURED (tools/package.py), against D6's 95 — legal with
    # 5.18 MB to spare, and all four checkpoints already existed.
    default_globs = SHIPPING
    repo = Path(__file__).resolve().parent.parent
    # R4 (host ruling, on the record since 08-12): everything loaded at
    # inference ships as ONE checkpoint file. `checkpoints/model.pth` (the site's
    # deliverable name; the legacy `model.pt` is accepted, `.pth` wins — P0 a) is that
    # file -- built by `tools/package.py --shipping --out`, gitignored like every
    # other weight, and supplied ALONGSIDE a clean clone exactly as m106's
    # fresh-clone rehearsal already supplies the four run checkpoints.
    #
    # The four-run SHIPPING list stays the fallback so the dev box keeps working
    # with no container present. WHICH ONE RAN IS PRINTED, because "was the
    # artefact I graded the artefact I ship" is precisely the question a silent
    # default makes unanswerable.
    if args.checkpoints is not None:
        specs = args.checkpoints
    elif SHIPPED_CONTAINER.is_file():
        specs = [str(SHIPPED_CONTAINER)]
        print(f"[predict] R4 single-file artefact: {SHIPPED_CONTAINER}", file=sys.stderr)
        both = [n for n in SHIPPED_CONTAINER_NAMES if (_CHECKPOINTS / n).is_file()]
        if len(both) > 1:
            # Two slots is one too many; say which one votes rather than let a
            # stale twin pass for the artefact.
            print(f"[predict] ⚠ both {' and '.join(both)} exist under checkpoints/ — "
                  f"{SHIPPED_CONTAINER.name} is the one loaded; the other is IGNORED", file=sys.stderr)
    else:
        specs = [str(repo / g) for g in default_globs]
        print(f"[predict] ⚠ no {SHIPPED_CONTAINER.name} — falling back to the "
              f"{len(default_globs)}-run dev path. This is NOT the R4 artefact; build it with "
              f"`tools/package.py --shipping --out {SHIPPED_CONTAINER}`.", file=sys.stderr)
    ensemble = None
    try:
        ckpts = find_checkpoints(specs)
        if not ckpts:
            raise FileNotFoundError(f"no checkpoint matched {specs}")
        ensemble = Ensemble(ckpts, args.device)
        print(f"[predict] ensemble: {ensemble.n} checkpoints over "
              f"{sorted(ensemble.by_branch)} on {ensemble.device}, hflip TTA on "
              f"· {ensemble.fp16_mb:.1f} MB fp16", file=sys.stderr)
        # Every member names its own arch and norm. Printed rather than assumed,
        # because "which normalisation did that member actually use" is the one
        # question a wrong answer to costs accuracy without costing an exception.
        for branch, members in sorted(ensemble.by_branch.items()):
            for mem in members:
                tag = " (legacy: arch/norm not stamped, resolved by default)" if mem["legacy"] else ""
                print(f"[predict]   {branch:<8} arch={mem['arch']:<13} norm={mem['norm']:<9}"
                      f" {mem['run']}{tag}{(' · ' + mem['quant']) if mem.get('quant') else ''}", file=sys.stderr)
        # §119 (P-A): "which fusion actually voted" is the same unanswerable question as "which checkpoint actually
        # voted" if it is not printed. Absent ⇒ the arithmetic mean and this line does not appear.
        if ensemble.fusion is not None:
            print(f"[predict]   fusion: {ensemble.fusion[0]} λ={ensemble.fusion[1]} over a stamped {len(ensemble.fusion[2])}-class "
                  f"TRAIN prior (§119); branches present < 2 pass through unchanged", file=sys.stderr)
        # THE ARTEFACT CAP IS A SUBMISSION RULE, NOT A PREFERENCE. D6: "the
        # 100 MB file IS the budget. fp16 default, <=95 MB packaged." An 18-
        # checkpoint experimental ensemble is ~403 MB and would be rejected --
        # and nothing else in this path would have noticed. Measurement 55 shows
        # obeying it costs nothing anyway: 2 checkpoints at 44.8 MB scored
        # HIGHER than 6 at 134 MB, because every measurable gain is in the first
        # checkpoint of each branch and hflip TTA spends no bytes at all.
        size_mb = ensemble.packed_mb if ensemble.all_packed else ensemble.fp16_mb
        if ensemble.all_packed:
            print(f"[predict] container on disk: {ensemble.packed_mb:.2f} MB (the cap is weighed on the file, D28)", file=sys.stderr)
        if size_mb > args.max_mb and not args.allow_oversize:
            raise RuntimeError(
                f"ensemble is {size_mb:.1f} MB {'on disk' if ensemble.all_packed else 'fp16'}, over the {args.max_mb:.0f} MB "
                f"packaged budget (D6) — pass --checkpoints with fewer members, or "
                f"--allow-oversize to measure something that cannot ship"
            )
    except Exception as exc:
        if not args.allow_constant_fallback:
            print(f"[predict] 🔴 FATAL: no model could be loaded "
                  f"({type(exc).__name__}: {exc}).\n"
                  f"[predict]    Refusing to write a submission. Without a model this run can "
                  f"only emit\n"
                  f"[predict]    class {FALLBACK_ACTION_ID} for all {len(row_keys)} rows, which "
                  f"scores the 0.10945 sanity floor,\n"
                  f"[predict]    fails reproduction by >10%, and would exit 0 with a CSV that "
                  f"looks correct.\n"
                  f"[predict]    Pass --allow-constant-fallback only to measure that floor "
                  f"deliberately.", file=sys.stderr)
            return 1
        print(f"[predict] ⚠ NO MODEL ({type(exc).__name__}: {exc}) — --allow-constant-fallback "
              f"is set, so emitting the constant class {FALLBACK_ACTION_ID}. "
              f"This scores the 0.10945 sanity floor and must never be submitted.",
              file=sys.stderr)

    n_missing = 0
    n_failed = 0
    predictions = []
    for key in row_keys:
        clip_dir = clip_dir_for(data_dir, key)
        if clip_dir is None:
            n_missing += 1
        try:
            action_id = predict_clip(clip_dir, ensemble)
            if not isinstance(action_id, int) or not 0 <= action_id <= 39:
                raise ValueError(f"prediction out of range: {action_id!r}")
        except Exception as exc:  # never crash on one clip
            print(f"[predict] FALLBACK for {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
            action_id = FALLBACK_ACTION_ID
            n_failed += 1
        predictions.append((key, action_id))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="" so csv does not emit \r\r\n; LF endings for a stable sha256.
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "prediction"])
        w.writerows(predictions)

    print(f"[predict] wrote {args.out} ({len(predictions)} rows)", file=sys.stderr)
    if n_missing:
        print(f"[predict] WARNING: {n_missing} rows had no directory on disk", file=sys.stderr)
    if n_failed:
        print(f"[predict] WARNING: {n_failed} rows fell back to class {FALLBACK_ACTION_ID}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
