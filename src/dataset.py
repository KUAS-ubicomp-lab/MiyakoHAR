"""Memmap-backed clip Dataset with TSN segmental sampling.

Reads the uint8 shards built by preprocess.py. The cache stores PIXELS; every
decision -- normalisation, the depth dead strip, temporal sampling, augmentation
-- lives here, so all of it stays A/B-able without a 20 GB rebuild.

TRAPS THIS FILE EXISTS TO HANDLE. Each one has already cost us, or would have.

1. A NUMPY MEMMAP MUST NOT CROSS A FORK. Opening it in __init__ and then setting
   num_workers>0 gives every worker a copy of the parent's page mappings; it
   appears to work and returns wrong or empty frames under load. The shard is
   therefore opened LAZILY, on first access inside whichever process is asking.

2. SHORT CLIPS ARE THE COMMON CASE, NOT AN EDGE CASE. IR's median is 24 frames
   and 42 clips hold exactly ONE. Segment boundaries come from linspace over the
   real length, so n=1 yields T copies of frame 0 with no branch and no crash.

3. NEVER FEED AN ALL-ZERO PLANE. cache_manifest.json's `channel_absent` records
   channel groups with no good frame anywhere -- 4 test clips have no IR at all,
   122 byte-identical zero-filled PNGs at the RIGHT byte size, so no size check
   catches them. Those clips still have valid Depth. We report present=False and
   let fusion renormalise over what is present, rather than showing the network
   an input it never saw in training.
   MEASURED 2026-08-13: every clip has at least one usable branch. The 4 IR-less
   test clips all have thermal; the 10 thermal-less test clips all have depth_ir;
   the intersection is EMPTY. So skip_branch always leaves something to fuse.

4. THE DEAD STRIP IS IN ORIGINAL PIXELS. Finding 21 measured "leftmost 40 px"
   on the 640-wide Depth frame = 6.25% of width. baseline.yaml carries
   `dead_strip_cols: 40` under a 96-wide input, where taking it literally masks
   42% of the frame. We store the ORIGINAL width alongside it and rescale.

5. A CLIP IS NOT AT THE INDEX YOU THINK. Two train clips are excluded from
   depthir_raw_train (no Depth at all), so shard row i and manifest row i are
   NOT the same clip. Every lookup goes through the shard's own clip_ids.

6. AUGMENTATION IS DRAWN ONCE PER CLIP. Per-frame parameters would inject motion
   that is not in the action and destroy exactly the signal TSM reads. All
   geometric and photometric params are sampled once and applied to all T frames.

7. THE DEAD STRIP IS A CONSTANT ONLY ON A FULL FRAME. Trap 4 rescales 40 original
   px to a fixed 10 of 160 cache columns because every full-frame clip shows the
   same 640 columns. The CROPPED branch shows a different window per clip, so the
   same 40 px land somewhere different -- or nowhere -- in each. Measured over
   cache/crops.csv: 82.53% of crop windows start past column 40 and contain NO
   dead strip at all, while the 17.47% that do would be masked at median 8 and up
   to 27 of their 96 columns. Reusing the full-frame constant would zero real
   foreground on five clips in six. dead_cols therefore lives on the ClipRef.

Sanity floor: 0.10945 (majority class). A model under ~0.11 has a bug here or in
model.py, not a weak signal.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# PyYAML IS DELIBERATELY NOT IMPORTED HERE. It was, at module level, until
# 2026-08-15 -- and predict.py imports NORM, ORIGINAL_WH and _tsn_indices from
# this file, so every inference run pulled PyYAML in with them. NOTICE lists it
# as TRAINING ONLY and TECHNICAL.md declares the inference path stdlib-plus-torch;
# both were false. The single consumer is load_folds(), which reads
# splits/folds.yaml on the training path only, so the import moves in there.
#
# HOW THIS SURVIVED A TEST WRITTEN TO CATCH IT. test_inference_robustness.sh
# puts a poisoned yaml module on PYTHONPATH and runs the real entry point. It
# passed anyway: the ImportError fired inside the per-clip try/except, every clip
# fell back to the majority class, a 6-row CSV was written and the script exited
# 0 -- which is exactly what the test asserted. The whole-run fallback that
# §J-DQ (a) removes was ALSO what made this defect invisible. Two defects, one
# mechanism, and the green test was green for the wrong reason.

# Original capture geometry, measured 2026-08-13 from the corpus itself.
# The cache is 160x120, so these are exact integer downscales -- which is why
# preprocess.py's Image.BOX (area-average) was the correct resampler.
ORIGINAL_WH = {"thermal": (320, 240), "depthir": (640, 480)}

# Per-branch normalisation. ImageNet statistics, with the IR channel taking the
# mean of the three RGB values (baseline.yaml).
#
# ADD, NEVER REPLACE. The `thermal` and `depthir` entries are the statistics
# every measured number in the ledger was produced under; changing either makes
# the control arm of J1 not bit-identical to ab5_lrhi and silently invalidates
# every paired comparison the project has. The Kinetics entries are NEW KEYS.
NORM = {
    "thermal": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "depthir": {
        "mean": (0.485, 0.456, 0.406, 0.449),
        "std": (0.229, 0.224, 0.225, 0.226),
    },
    # Kinetics-400 statistics, read off S3D_Weights.KINETICS400_V1.transforms()
    # on 2026-08-15 -- NOT borrowed from the R3D family, which carries different
    # values. The 4th (IR) plane takes the mean of the three colour planes, the
    # same convention the ImageNet entries above already use, so the s3d and
    # ResNet-18 arms differ in the pretraining corpus and in nothing else.
    #   mean [0.43216, 0.394666, 0.37645]  -> mean of the three = 0.401092
    #   std  [0.22803, 0.22145, 0.216989]  -> mean of the three = 0.222156
    "thermal_kinetics": {
        "mean": (0.43216, 0.394666, 0.37645),
        "std": (0.22803, 0.22145, 0.216989),
    },
    "depthir_kinetics": {
        "mean": (0.43216, 0.394666, 0.37645, 0.401092),
        "std": (0.22803, 0.22145, 0.216989, 0.222156),
    },
    # §96 (c) (D29): the depth+IR branch with a FIFTH channel, the depth-validity plane (`depth != black`, computed from the
    # raw frames BEFORE augmentation by `depth_validity` below). The four shipped statistics verbatim + (0, 1) for the plane,
    # which is already in [0, 1]. A NEW KEY, as the rule above says; a member that declares it gets the plane from the loader
    # and from predict.py through the same helper.
    "depthir_valid": {
        "mean": (0.485, 0.456, 0.406, 0.449, 0.0),
        "std": (0.229, 0.224, 0.225, 0.226, 1.0),
    },
    # §96 (a)/(b) (D29): the ordinal branches under the depth+IR statistics verbatim — NEW KEYS so the stamp names the input, the
    # values the incumbent's (the only change is the input's content). ADDED, never replaced.
    "depthir_ord3": {
        "mean": (0.485, 0.456, 0.406, 0.449),
        "std": (0.229, 0.224, 0.225, 0.226),
    },
    "depthir_ordgrad": {
        "mean": (0.485, 0.456, 0.406, 0.449),
        "std": (0.229, 0.224, 0.225, 0.226),
    },
}

# The NORM keys whose members take the depth-validity plane as their last channel (§96 (c)).
VALID_MASK_NORMS = frozenset({"depthir_valid"})

# §95 (iii): the replayed-geometry generators are seeded (seed, row, GEO_SALT, k) — a fixed salt so they never coincide with the per-item
# (seed, row, spark) generators; the canonical draw length is GEO_T (the thermal student's T; the T = 16 consumers take every other frame).
GEO_SALT = 20260905
GEO_T = 32


def depth_validity(x: torch.Tensor) -> torch.Tensor:
    """§96 (c): the depth-validity plane of a (T, C, H, W) [0, 1] tensor whose first three planes are the JET depth rendering —
    1.0 where the depth rendering is NOT black (all three planes exactly 0 = "no return"), else 0.0; (T, 1, H, W), x's dtype.
    THE ONE DEFINITION: the loader calls it on the raw frames BEFORE `_augment` (post-augmentation the plane would be an
    artefact — `adjust_contrast` moves black off zero, `affine`'s fill manufactures fake zeros), and predict.py calls it on the
    same raw tensor after the dead-column strip, so training and inference agree pixel for pixel. Per-clip; nothing from any
    other clip (L7)."""
    return (x[:, 0:3].abs().sum(dim=1, keepdim=True) > 0).to(x.dtype)

# Channel groups get separate photometric treatment: hue IS the depth value and
# hue IS the temperature, so neither may be jittered.
PHOTOMETRIC = {
    "thermal": [(slice(0, 3), 0.30, 0.30)],
    "depthir": [(slice(0, 3), 0.15, 0.15), (slice(3, 4), 0.30, 0.30)],
}

# Which stream a branch reads, and which channel group must be present for the
# branch to be usable at all.
BRANCH_STREAM = {"thermal": "thermal", "depthir": "depthir_raw", "depthir_crop": "depthir_crop",
                 "depthir_cropsq": "depthir_cropsq",   # A3 (QUEUE §33): the at-source square crop
                 "depthir_ord3": "depthir_ord", "depthir_ordgrad": "depthir_ord"}   # §96 (a)/(b): the ordinal stream
BRANCH_REQUIRED_GROUP = {"thermal": "rgb", "depthir": "depth", "depthir_crop": "depth", "depthir_cropsq": "depth",
                         "depthir_ord3": "depth", "depthir_ordgrad": "depth"}
# §96 (a)/(b) (D29): the ordinal branches read the depthir_ord shard [ord, dx, dy, IR]; (a) ord3 replicates the ordinal into the three
# depth channels (ORD3_PLANES), (b) ordgrad takes the planes as stored. Both keep the depth+IR photometric group and dead strip.
ORD_BRANCHES = ("depthir_ord3", "depthir_ordgrad")
ORD3_PLANES = (0, 0, 0, 3)
# Which set of channel statistics a branch belongs to. `depthir_crop` is the same
# four channels as `depthir` -- only the framing differs -- so it must share the
# normalisation and the photometric policy exactly, or A/B #3 measures two
# changes at once. Previously this was an inline `if branch == "thermal" else
# "depthir"` repeated in three places, which happened to give the right answer
# for a third branch by accident; a table cannot be right by accident.
BRANCH_GROUP = {"thermal": "thermal", "depthir": "depthir", "depthir_crop": "depthir", "depthir_cropsq": "depthir",
                "depthir_ord3": "depthir", "depthir_ordgrad": "depthir"}


@dataclass(frozen=True)
class ClipRef:
    """One row of the dataset. `row` indexes the SHARD, never the manifest."""

    clip_id: str
    row: int
    start: int
    stop: int
    label: int
    user: int
    branch_present: bool
    # Dead-strip columns to mask, in CACHE pixels. Constant per branch on the
    # full-frame streams; PER CLIP on the cropped one -- see trap 7.
    dead_cols: int = 0


def load_folds(path: Path) -> dict:
    """Read the frozen split. TRAINING PATH ONLY — see the import note above."""
    import yaml  # noqa: PLC0415 — deliberately lazy; keeps PyYAML out of inference

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _tsn_indices(n: int, T: int, train: bool, rng: np.random.Generator | None) -> np.ndarray:
    """TSN segmental sampling. Correct from n=1 upward -- see trap 2.

    Segment k spans [k*n/T, (k+1)*n/T). Training draws uniformly inside it;
    evaluation takes its centre, which makes the eval path deterministic without
    a seed. When n < T the segments overlap and frames repeat, which is the
    repeat_pad policy falling out of the same expression rather than a branch.
    """
    edges = np.linspace(0.0, float(n), T + 1)
    lo, hi = edges[:-1], edges[1:]
    if train:
        assert rng is not None
        pos = lo + rng.random(T) * (hi - lo)
    else:
        pos = (lo + hi) * 0.5
    return np.clip(pos.astype(np.int64), 0, n - 1)


class ClipDataset(Dataset):
    """One branch, one split, one fold.

    branch : 'thermal' | 'depthir' | 'depthir_crop'
    split  : 'train' | 'test'          (the corpus split -- what is on disk)
    mode   : 'train' | 'val' | 'test'  (what we do with it)

    For split='train', mode selects the fold side: 'train' takes the fold's
    training subjects, 'val' its held-out six. Test clips carry label -1.
    """

    def __init__(
        self,
        cache_dir: Path,
        manifest_csv: Path,
        branch: str,
        split: str,
        mode: str,
        fold: int = 0,
        T: int = 8,
        folds_file: Path | None = None,
        augment: bool | None = None,
        dead_strip_orig_px: int = 40,
        seed: int = 20260813,
        norm: str | None = None,
        input_size: tuple[int, int] | None = None,
        train_subjects: int | None = None,
        all_subjects: bool = False,
        pseudo_labels: dict[str, int] | None = None,
        pseudo_val: bool = False,
        aug_wide: bool = False,
        aug_tcrop: bool = False,
        aug_trivial: bool = False,
        aug_transplant: bool = False,
        geo_draws: int | None = None,
        geo_k: int | None = None,
        geo_T: int | None = None,
        photometric: bool = True,
    ) -> None:
        if branch not in BRANCH_STREAM:
            raise ValueError(f"branch must be one of {sorted(BRANCH_STREAM)}, got {branch!r}")
        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be train|val|test, got {mode!r}")
        if (mode == "test") != (split == "test"):
            raise ValueError(f"mode={mode!r} is incoherent with split={split!r}")

        # DEFAULTS REPRODUCE THE LEGACY PATH EXACTLY. norm=None resolves to
        # BRANCH_GROUP[branch] and input_size=None skips the resize entirely, so
        # a ClipDataset built the way every existing run built one yields
        # bit-identical tensors. That is what keeps J1's control arm comparable
        # to ab5_lrhi rather than merely similar to it.
        if norm is not None and norm not in NORM:
            raise ValueError(f"norm must be one of {sorted(NORM)}, got {norm!r}")
        self.norm = norm
        self.input_size = tuple(input_size) if input_size else None
        self.train_subjects = train_subjects   # J5, m86 — train split only
        self.all_subjects = all_subjects       # J9, m87 — train split only
        # SELF-TRAINING, rules-class PSEUDO-LABEL(R7) — the ONE recorded legal
        # class of test-input-derived technique. These labels are OUR MODEL'S OWN
        # PREDICTIONS on test INPUTS; no test label is read, here or anywhere. The
        # guard below makes that structural: pseudo_labels is REFUSED on the train
        # split, so it can never silently overwrite a real corpus label.
        # §58 (D21): the FOLD INSTRUMENT for self-training — pseudo_val=True admits pseudo
        # labels on the fold's VAL clips (mode="val" of the train split): the clips' labels
        # then come from the dict ONLY, the corpus label is never read as a target, and the
        # train mode of the same fold is untouched. Off => the original guard, unchanged.
        if pseudo_labels is not None and split != "test" and not (pseudo_val and mode == "val"):
            raise ValueError(
                f"pseudo_labels apply to the TEST split only, got split={split!r}. "
                "On the train split they would overwrite real labels."
            )
        self.pseudo_labels = pseudo_labels
        self.pseudo_val = bool(pseudo_val and pseudo_labels is not None and mode == "val")
        # T3 (QUEUE §26): 2x widening of the clip-consistent affine ranges. Off => bit-identical.
        self.aug_wide = aug_wide
        # T6 (QUEUE §29): clip-consistent TEMPORAL crop, train mode only, p = 0.5 per clip. Off => bit-identical.
        self.aug_tcrop = aug_tcrop
        # §72 (D24): TrivialAugment-Wide — ONE extra op per clip at a uniform magnitude, clip-consistent, on top of
        # the recipe's block; drawn from the same per-clip rng, so off ⇒ the recipe's pixels bitwise.
        self.aug_trivial = aug_trivial
        # A5 (QUEUE §36): subject-crossing box transplant, train mode only, p = 0.5, depth+IR branches
        # with boxes only. The tight boxes come from cache/crops.csv (ORIGINAL 640x480 px) and are
        # scaled to the cache geometry at use. Off => bit-identical.
        self.aug_transplant = aug_transplant
        self._boxes: dict[str, tuple[int, int, int, int]] = {}
        if aug_transplant:
            if branch not in ("depthir", "depthir_crop", "depthir_cropsq"):
                raise ValueError("aug_transplant needs a depth+IR branch (thermal has no boxes)")
            if branch != "depthir":
                raise ValueError("aug_transplant is defined on the full-frame depthir stream (QUEUE §36)")
            with (Path(cache_dir) / "crops.csv").open(newline="", encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    if r["split"] == split and r["status"] == "ok":
                        self._boxes[r["clip_id"]] = (int(r["tight_x0"]), int(r["tight_y0"]),
                                                     int(r["tight_x1"]), int(r["tight_y1"]))

        self.cache_dir = Path(cache_dir)
        self.branch = branch
        self.stream = BRANCH_STREAM[branch]
        self.split = split
        self.mode = mode
        self.T = T
        self.seed = seed
        # Augment on the training side only. Val and test must be deterministic:
        # checkpoint selection and the submission both depend on it.
        self.augment = (mode == "train") if augment is None else augment
        # §95 (iii) (D29): CONSISTENT GEOMETRIC TEACHING — the replayed-geometry mode. With `geo_draws = K` (the student: the item's
        # draw index k = spark % K) or `geo_k = k` (the teacher's scoring pass k), the GEOMETRIC parameters of the augmentation — the
        # temporal-crop window, the TSN segment offsets (drawn at the canonical `geo_T` and subsampled to this dataset's T), the hflip
        # and the affine — come from a per-(clip, k) generator, default_rng((seed, row, GEO_SALT, k)), so a teacher and a student see the
        # SAME geometry for the same (clip, k); the PHOTOMETRIC parameters stay per item (`photometric=False` skips them — the teacher's
        # side). With the mode off every draw comes from the ONE per-item generator in the original order: the incumbent's tensors are
        # bitwise unchanged (tests/check_geo_replay.py §1). Eval never enters the mode (augment is False there).
        if geo_draws is not None and geo_k is not None:
            raise ValueError("geo_draws (the student) and geo_k (the teacher's pass) are exclusive")
        self.geo_draws, self.geo_k, self.photometric = geo_draws, geo_k, bool(photometric)
        self.geo_T = int(geo_T) if geo_T is not None else None
        if (geo_draws is not None or geo_k is not None):
            if self.geo_T is None:
                raise ValueError("the replayed-geometry mode needs geo_T (the canonical draw length, 32)")
            if self.geo_T < self.T or self.geo_T % self.T != 0:
                raise ValueError(f"geo_T {self.geo_T} must be a multiple of T {self.T} (the T = 16 consumer takes every (geo_T / T)-th frame)")

        cm = json.loads((self.cache_dir / "cache_manifest.json").read_text())
        shards = [s for s in cm["shards"] if s["stream"] == self.stream and s["split"] == split]
        if len(shards) != 1:
            raise RuntimeError(f"expected exactly one {self.stream}/{split} shard, found {len(shards)}")
        self.shard_meta = shards[0]
        self.shard_path = self.cache_dir / self.shard_meta["shard"]
        total, self.H, self.W, self.C = self.shard_meta["shape"]
        self._n_frames_total = total
        self._mm: np.memmap | None = None  # trap 1: opened lazily, per process

        # Trap 4: rescale the dead strip from original pixels to cache pixels.
        # TRAP 7 -- ON THE CROPPED BRANCH THIS IS NOT A CONSTANT. The strip is
        # the leftmost 40 ORIGINAL columns, and a crop window starting past column
        # 40 does not contain it at all while a narrow window starting at 0
        # contains it at a much larger FRACTION than the full frame does. Measured
        # over cache/crops.csv: 17.47% of clips have x0 < 40, and for those the
        # masked width is median 8 / max 27 of 96 columns against the full frame's
        # uniform 10 of 160. Carrying the full-frame 10 across would mask the wrong
        # pixels on 100% of clips -- foreground on most of them. None here means
        # "per clip"; the value used is on the ClipRef.
        self.dead_strip_orig_px = dead_strip_orig_px
        if branch in ("depthir_crop", "depthir_cropsq"):
            self.dead_strip_cols: int | None = None
        elif branch == "depthir" or branch in ORD_BRANCHES:
            ow = ORIGINAL_WH["depthir"][0]
            self.dead_strip_cols = int(round(dead_strip_orig_px * self.W / ow))
        else:
            self.dead_strip_cols = 0

        # Trap 3: which clips cannot feed this branch.
        absent = self.shard_meta.get("channel_absent", {})
        required = BRANCH_REQUIRED_GROUP[branch]
        self._unusable = set(absent.get(required, []))
        # An absent non-required group (IR gone, Depth intact) still yields a
        # usable clip -- it is only the required group that kills the branch.
        self._partial = {c for g, ids in absent.items() if g != required for c in ids}

        self._build_index(manifest_csv, fold, folds_file)

    # ── the dead strip, per clip (trap 7) ───────────────────────────────────
    def _dead_cols(self, shard_row: int) -> int:
        """Dead-strip columns to mask for shard row `shard_row`, in cache pixels.

        Rounds UP: a partially-dead cache column is dead, and the alternative
        leaves a sliver of the strip feeding the network on exactly the clips
        where it is closest to the subject.
        """
        if self.dead_strip_cols is not None:
            return self.dead_strip_cols
        x0, _, x1, _ = self.shard_meta["crop_boxes"][shard_row]
        inside = max(0, self.dead_strip_orig_px - x0)  # ORIGINAL px of strip in the window
        return min(self.W, -(-inside * self.W // (x1 - x0)))

    # ── index ───────────────────────────────────────────────────────────────
    def _build_index(self, manifest_csv: Path, fold: int, folds_file: Path | None) -> None:
        with Path(manifest_csv).open(newline="", encoding="utf-8-sig") as fh:
            rows = {r["clip_id"]: r for r in csv.DictReader(fh) if r["split"] == self.split}

        keep_users: set[int] | None = None
        if self.split == "train":
            ff = folds_file or (Path(__file__).resolve().parent.parent / "splits" / "folds.yaml")
            spec = load_folds(Path(ff))
            entry = next(f for f in spec["folds"] if f["fold"] == fold)
            keep_users = set(entry["train" if self.mode == "train" else "val"])

            # ── J9: train the SHIPPED member on all 18 subjects (m87) ───────
            # THIS IS THE ONE MODE THAT HAS NO VALIDATION SET BY CONSTRUCTION,
            # and that is the whole point: the fold's held-out six are moved into
            # training, so nothing is left to score on. D5 is NOT touched — it
            # freezes the folds as the measurement instrument, and this mode does
            # not re-roll, enlarge or edit them; it declines to use them to carve
            # the artefact. m87: "the folds are an instrument, not a constraint on
            # the artefact." train.py refuses to emit any val number in this mode.
            if self.all_subjects:
                if self.mode != "train":
                    raise ValueError(
                        "all_subjects builds a TRAIN dataset over every subject. Applying it to "
                        f"mode={self.mode!r} is incoherent — there is no held-out side left."
                    )
                if self.train_subjects is not None:
                    raise ValueError(
                        "all_subjects and train_subjects are mutually exclusive: one takes every "
                        "subject, the other takes N of the fold's 12."
                    )
                keep_users = None      # no user filter — every training subject
                print(f"[dataset] J9: training on ALL {len(set(entry['train']) | set(entry['val']))} "
                      f"subjects (fold {fold}'s val subjects {sorted(entry['val'])} are now TRAIN) "
                      f"-> NO VALIDATION SET EXISTS")

            # ── J5: subject-count ablation (m86) ────────────────────────────
            # IT TOUCHES THE TRAIN SPLIT ONLY, NEVER val. D5 freezes the folds
            # and forbids re-rolling or enlarging them; this neither re-rolls nor
            # enlarges — it trains on a SUBSET of a fold's own training subjects
            # and scores on that fold's untouched validation subjects, so the
            # metric and the comparison set are exactly what every other run uses.
            # The guard below makes that structural rather than a promise.
            if self.train_subjects is not None:
                if self.mode != "train":
                    raise ValueError(
                        "train_subjects may only subset the TRAIN split. Applying it to "
                        f"mode={self.mode!r} would change the validation set and void D5."
                    )
                # BALANCED ACROSS RECORDING BLOCKS, NOT SORTED. m6b measured
                # that users 1-9 and 16-24 are different blocks (radar presence is
                # a perfect block indicator, eta^2 0.973). Taking the first N in
                # sorted order would take Block I first, so the "subject count"
                # slope would silently be a "block coverage" slope instead —
                # exactly the confound D2 dropped an entire modality to avoid.
                lo = sorted(u for u in keep_users if u <= 9)
                hi = sorted(u for u in keep_users if u >= 16)
                n = self.train_subjects
                if n > len(lo) + len(hi):
                    raise ValueError(f"fold {fold} has {len(lo) + len(hi)} train subjects, asked for {n}")
                take_lo, take_hi = (n + 1) // 2, n // 2
                # If one block is short, spill into the other rather than silently
                # returning fewer subjects than were asked for.
                if take_lo > len(lo):
                    take_hi += take_lo - len(lo); take_lo = len(lo)
                if take_hi > len(hi):
                    take_lo += take_hi - len(hi); take_hi = len(hi)
                keep_users = set(lo[:take_lo]) | set(hi[:take_hi])
                print(f"[dataset] J5: training on {len(keep_users)} of "
                      f"{len(lo) + len(hi)} subjects -> {sorted(keep_users)} "
                      f"(block I {take_lo}, block II {take_hi})")

        clip_ids = self.shard_meta["clip_ids"]
        offsets = self.shard_meta["offsets"]

        self.items: list[ClipRef] = []
        self.n_skipped_unusable = 0
        for i, cid in enumerate(clip_ids):
            row = rows.get(cid)
            if row is None:
                raise KeyError(f"shard clip {cid!r} is absent from the manifest -- caches disagree")
            user = int(row["user"])
            if keep_users is not None and user not in keep_users:
                continue
            if self.pseudo_labels is not None and cid not in self.pseudo_labels:
                # Below the confidence threshold, or unusable. NOT kept with -1:
                # a -1 reaching CrossEntropyLoss is a silent garbage gradient.
                continue
            if cid in self._unusable:
                # Trap 3. Never silently drop: count it, and predict.py handles
                # the test-side clip through fusion renormalisation instead.
                self.n_skipped_unusable += 1
                continue
            self.items.append(
                ClipRef(
                    clip_id=cid,
                    row=i,
                    start=int(offsets[i]),
                    stop=int(offsets[i + 1]),
                    label=(self.pseudo_labels[cid] if self.pseudo_val            # §58: the teacher's label
                           else (int(row["action_id"]) if self.split == "train"
                                 else (self.pseudo_labels.get(cid, -1)
                                       if self.pseudo_labels is not None else -1))),
                    user=user,
                    branch_present=cid not in self._partial,
                    dead_cols=self._dead_cols(i),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    # ── pixels ──────────────────────────────────────────────────────────────
    def _shard(self) -> np.memmap:
        if self._mm is None:  # trap 1
            self._mm = np.memmap(
                self.shard_path,
                dtype=np.uint8,
                mode="r",
                shape=(self._n_frames_total, self.H, self.W, self.C),
            )
        return self._mm

    def __getitem__(self, i: int) -> dict:
        ref = self.items[i]
        n = ref.stop - ref.start
        if n <= 0:
            raise RuntimeError(f"{ref.clip_id} has {n} frames in the shard -- check_cache.py should have caught this")

        rng = None
        if self.augment:
            # THE SEED MUST ADVANCE. Seeding from (self.seed, ref.row,
            # torch.initial_seed()) looks per-clip and reproducible, and is
            # BROKEN: all three are constant within a process, so a given clip
            # draws the SAME augmentation in every epoch forever. With
            # persistent_workers=true the workers are never re-seeded either, so
            # nothing rescues it. The network would see 2,891 fixed images
            # instead of 2,891 varied ones, and the only symptom would be an A/B
            # concluding that augmentation does not help.
            # torch's per-worker RNG is seeded differently per worker PER EPOCH
            # by DataLoader and advances on every draw, so this varies.
            # Eval never reaches here, so determinism there is untouched.
            spark = int(torch.randint(0, 2**31 - 1, (1,)).item())
            rng = np.random.default_rng((self.seed, ref.row, spark))

        # §95 (iii): the geometric generator is the per-item one (the incumbent's path — the SAME object, so the draw order is
        # unchanged) unless the replayed-geometry mode is on, when it is the per-(clip, k) one and the draw length is geo_T.
        rng_geo, draw_k, T_draw = rng, -1, self.T
        if self.augment and (self.geo_draws is not None or self.geo_k is not None):
            draw_k = int(self.geo_k) if self.geo_k is not None else int(spark % self.geo_draws)
            rng_geo = np.random.default_rng((self.seed, ref.row, GEO_SALT, draw_k))
            T_draw = self.geo_T

        if self.augment and self.aug_tcrop and rng_geo.random() < 0.5:
            # T6: lay the T segments over a random contiguous window of the clip (>= half of it);
            # a window shorter than T repeats frames by the existing repeat_pad rule. Eval never
            # reaches here (self.augment is False), so the deterministic path is untouched.
            win = max(1, int(round(n * rng_geo.uniform(0.5, 1.0))))
            off = int(rng_geo.integers(0, n - win + 1))
            idx = off + _tsn_indices(win, T_draw, train=True, rng=rng_geo)
        else:
            idx = _tsn_indices(n, T_draw, train=self.augment, rng=rng_geo)
        if T_draw != self.T:
            idx = idx[:: T_draw // self.T]      # the T = 16 consumer takes the even frames of the canonical T = 32 draw
        frames = np.asarray(self._shard()[ref.start + idx])  # (T,H,W,C) uint8
        if self.branch == "depthir_ord3":
            frames = frames[..., list(ORD3_PLANES)]           # §96 (a): [ord, ord, ord, IR]

        if self.augment and self.aug_transplant and ref.clip_id in self._boxes and rng.random() < 0.5:
            # A5 (QUEUE §36): paste this clip's tight-box region (its subject) onto another training
            # clip's frames, sampled at the same relative positions; the label stays this clip's.
            other = self.items[int(rng.integers(0, len(self.items)))]
            n_o = other.stop - other.start
            idx_o = np.clip((idx.astype(np.float64) / max(1, n - 1) * (n_o - 1)).round().astype(np.int64), 0, n_o - 1)
            bg = np.array(self._shard()[other.start + idx_o])          # (T,H,W,C) copy
            ox0, oy0, ox1, oy1 = self._boxes[ref.clip_id]
            sx, sy = self.W / ORIGINAL_WH["depthir"][0], self.H / ORIGINAL_WH["depthir"][1]
            x0, x1 = int(ox0 * sx), max(int(ox0 * sx) + 1, int(round(ox1 * sx)))
            y0, y1 = int(oy0 * sy), max(int(oy0 * sy) + 1, int(round(oy1 * sy)))
            bg[:, y0:y1, x0:x1, :] = frames[:, y0:y1, x0:x1, :]
            frames = bg

        x = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float().div_(255.0)

        if ref.dead_cols > 0:
            # Depth only. IR is intact there (finding 21) and keeps its signal.
            x[:, 0:3, :, : ref.dead_cols] = 0.0

        if self.norm in VALID_MASK_NORMS:
            # §96 (c) (D29): the depth-validity plane from the RAW frames, BEFORE augmentation (G-methods-critic F13), after the
            # dead-column strip (those columns carry no return and read as black on both the train and the inference path).
            # The geometric ops below carry it (nearest fill → it stays binary); the photometric ops touch slices 0:3 / 3:4 only.
            x = torch.cat([x, depth_validity(x)], dim=1)

        if self.augment:
            x = self._augment(x, rng, rng_geo, self.photometric)

        # The arch's statistics when it declared one, the branch's otherwise.
        # NOT BRANCH_GROUP unconditionally: a Kinetics-pretrained backbone fed
        # ImageNet statistics does not raise, it just gets quietly worse.
        key = self.norm or BRANCH_GROUP[self.branch]
        mean = torch.tensor(NORM[key]["mean"], dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(NORM[key]["std"], dtype=x.dtype).view(1, -1, 1, 1)
        x = (x - mean) / std

        # Resolution is a J1 factor: s3d's weights were trained at 224x224 and
        # the cache is 120x160. Interpolating here needs no cache rebuild and no
        # new shard -- and it happens AFTER normalisation and augmentation so
        # that neither is affected by the target size.
        if self.input_size is not None and tuple(x.shape[-2:]) != self.input_size:
            x = torch.nn.functional.interpolate(
                x, size=self.input_size, mode="bilinear", align_corners=False)

        return {
            "x": x,  # (T, C, H, W)
            "y": ref.label,
            "clip_id": ref.clip_id,
            "user": ref.user,
            "n_frames": n,
            "present": ref.branch_present,
            "draw_k": draw_k,           # §95 (iii): the replayed draw index; −1 outside the mode
        }

    # ── augmentation, drawn ONCE per clip (trap 6) ──────────────────────────
    def _augment(self, x: torch.Tensor, rng: np.random.Generator, rng_geo: np.random.Generator | None = None,
                 photometric: bool = True) -> torch.Tensor:
        from torchvision.transforms.v2 import functional as F

        # §95 (iii): `rng_geo` draws the hflip and the affine; it IS `rng` outside the replayed-geometry mode, so the incumbent's
        # draw order (photometric → hflip → affine → trivial) and values are unchanged. `photometric=False` is the teacher's side.
        if rng_geo is None:
            rng_geo = rng
        key = BRANCH_GROUP[self.branch]
        if photometric:
            for sl, b, c in PHOTOMETRIC[key]:
                if b > 0:
                    x[:, sl] = F.adjust_brightness(x[:, sl], float(rng.uniform(1 - b, 1 + b)))
                if c > 0:
                    x[:, sl] = F.adjust_contrast(x[:, sl], float(rng.uniform(1 - c, 1 + c)))

        # hflip is safe here: none of the 40 classes is chirality-defined
        # (no left/right variants) -- verified against the class list 2026-08-13.
        if rng_geo.random() < 0.5:
            x = F.horizontal_flip(x)

        # Whole-clip affine. Same parameters for all T frames, so the induced
        # motion is zero and TSM still sees only the action's own motion.
        # T3 (QUEUE §26): --aug-wide widens the ranges 2x (rot +-15, translate +-15 %, scale 0.70-1.30);
        # the draw stays ONCE per clip. Default = the ranges every ledger number was produced under.
        rot, tr, (s_lo, s_hi) = (15.0, 0.15, (0.70, 1.30)) if self.aug_wide else (10.0, 0.08, (0.85, 1.15))
        angle = float(rng_geo.uniform(-rot, rot))
        tx = float(rng_geo.uniform(-tr, tr)) * self.W
        ty = float(rng_geo.uniform(-tr, tr)) * self.H
        scale = float(rng_geo.uniform(s_lo, s_hi))
        x = F.affine(x, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0])
        if self.aug_trivial:
            x = self._trivial(x, rng, F)
        return x

    # §72 (D24): TrivialAugment-Wide's rule — one op, one uniform magnitude — clip-consistent (the draws come from
    # the clip's rng, so every frame gets the same op and parameters). Hue is never touched (hue IS temperature);
    # no mixing op (VideoMix is closed). Photometric ops act on the first three channels only (torchvision's
    # colour ops take 1 or 3 channels); geometric ops act on every channel.
    def _trivial(self, x: torch.Tensor, rng: np.random.Generator, F) -> torch.Tensor:
        op = int(rng.integers(0, 8))
        m = float(rng.uniform(0.0, 1.0))
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        sgn2 = 1.0 if rng.random() < 0.5 else -1.0
        c3 = slice(0, min(3, x.shape[1]))
        if op == 0:
            x[:, c3] = F.adjust_brightness(x[:, c3], 1.0 + sgn * 0.6 * m)
        elif op == 1:
            x[:, c3] = F.adjust_contrast(x[:, c3], 1.0 + sgn * 0.6 * m)
        elif op == 2:
            x[:, c3] = F.adjust_gamma(x[:, c3].clamp(0.0, 1.0), 0.5 + 1.5 * m)
        elif op == 3:
            x[:, c3] = F.adjust_sharpness(x[:, c3], 1.0 + sgn * 0.9 * m)
        elif op == 4:
            x = F.affine(x, angle=sgn * 20.0 * m, translate=[0, 0], scale=1.0, shear=[0.0, 0.0])
        elif op == 5:
            x = F.affine(x, angle=0.0, translate=[sgn * 0.12 * m * self.W, sgn2 * 0.12 * m * self.H], scale=1.0, shear=[0.0, 0.0])
        elif op == 6:
            x = F.affine(x, angle=0.0, translate=[0, 0], scale=1.0 + sgn * 0.2 * m, shear=[0.0, 0.0])
        else:
            x = F.affine(x, angle=0.0, translate=[0, 0], scale=1.0, shear=[sgn * 10.0 * m, 0.0])
        return x


def collate(batch: list[dict]) -> dict:
    """Stack to (B,T,C,H,W) and keep the per-clip bookkeeping as plain lists."""
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
        "clip_id": [b["clip_id"] for b in batch],
        "user": torch.tensor([b["user"] for b in batch], dtype=torch.long),
        "n_frames": torch.tensor([b["n_frames"] for b in batch], dtype=torch.long),
        "present": torch.tensor([b["present"] for b in batch], dtype=torch.bool),
        "draw_k": torch.tensor([b.get("draw_k", -1) for b in batch], dtype=torch.long),   # §95 (iii)
    }
