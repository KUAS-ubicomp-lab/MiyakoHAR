"""Decode the corpus once into memmap shards that training reads at random.

WHY THIS EXISTS. Training touches every clip 40 times per fold, three folds, then
once per A/B. Decoding a 640x480 PNG costs 2.3 ms (IR) or 4.8 ms (Depth); a batch
of 16 clips at T=8 is 128 frames, so the raw path spends ~0.9 s of CPU per step
and reads 61 MB of disk. The GPU step is ~50-100 ms. Decoding is therefore the
whole run. Decoded once into uint8 memmaps, the same batch is a 9 MB read.

WHAT IS BAKED IN, AND WHAT DELIBERATELY IS NOT.
The cache stores pixels; the model stores decisions. Baked in: decode, optional
person crop, resize, uint8. Everything reversible stays at training time --
normalisation, the depth dead strip (finding 21), augmentation, temporal
sampling. Two irreversible things are accepted because they are what makes the
cache small: the resize, and the crop on the cropped shard. That is why the
uncropped shard exists alongside it.

WHOLE CLIPS, NOT SAMPLED FRAMES. `baseline.yaml` carried max_frames_per_clip: 32
against its own comment. Measured on this corpus, a 32-cap discards 23.3% of IR
frames and 58.5% of thermal frames -- 74.7% of thermal clips are longer than 32,
and thermal is the strongest modality. Whole clips cost 9 GB more on a disk with
776 GB free. Ruled 2026-08-13; the config line is corrected in place.

Repeat-padding a short clip up to T is a SAMPLER concern, not a storage one. This
module stores what exists, including the 42 clips holding exactly one frame and
the clips holding zero. The dataset pads at read time, where it has to anyway --
IR's median is 24 frames, so T=8 with segmental sampling meets short clips
constantly, not only at n=1.

THE TRAPS THIS MODULE HANDLES, each measured rather than anticipated:

1. IR and Depth pair on a SHARED TOKEN, never on filename. IR ships as
   `IR_<date>_<time>_<globalidx>.png` and Depth as
   `Depth_<date>_<time>_<globalidx>_Color.png`. A naive stem match fails on 100%
   of clips; the shared `<date>_<time>_<globalidx>` matches on 300/300 sampled.
   This is the same class of error as the IMU filename trap in manifest.py.

2. Frames order by GLOBAL FRAME INDEX, not by filename sort. The index is the
   documented shared clock across IR, Depth and Skeleton. A natural sort over the
   whole filename lets the date digits dominate the key.

3. Bad frames are handled PER CHANNEL, and this corrects the record. The known
   figure was "122 of 9,190 test IR PNGs unreadable, confined to 4 clips — a
   per-file try/except is enough." Re-measured 2026-08-13, that reading is wrong
   in the way that matters:

     · The 122 are not scattered bad frames. They are 100% of the IR in exactly
       four clips (SM_test_0012, 0014, 0154, 0194). Those clips have NO IR at all.
     · All 122 are byte-identical: 308,272 bytes of pure zero, one md5 across the
       set, no PNG signature. Nothing is recoverable — not truncation, and
       LOAD_TRUNCATED_IMAGES recovers 0 of 122.
     · They are the RIGHT SIZE. A good IR PNG is 308,262 bytes, so every
       size-based sanity check passes. This is the third recurrence of the
       project's own lesson — a file existing is not the modality being present —
       and the most deceptive instance of it so far.
     · Their Depth_Color is 100% intact, 122/122.

   So a whole-frame try/except is exactly the bug: it throws away good depth
   alongside absent IR and forfeits four TEST clips (~2 of the 201 public clips,
   about 1 pp). Each channel group is therefore decoded and repaired
   independently — reuse-previous-good, back-filling the head from the first good
   frame when a clip opens with bad ones. A channel with no good frame anywhere
   stays zero and the clip is named in cache_manifest.json under
   `channel_absent`, so the dataset skips that branch and fusion renormalises
   over the branches that are present, rather than feeding the model an all-zero
   plane it has never seen in training.

   Scope is bounded and measured, not assumed: all 413,418 image files in the
   corpus were magic-byte scanned in 85 s. Train is 100% clean across Depth, IR
   and Thermal; test Depth and Thermal are 100% clean. The 122 above are the
   entire defect.

4. Ragged modality coverage, which nothing recorded before finding 24. Of 3,036
   train clips, 103 have no IR at all and 145 no thermal. Two more (7_Eat_food/
   user22/3-1-3 and 37_Take_medicine/user23/7-1-2) have IR but NO Depth. Those
   two are EXCLUDED from the depth_ir shards rather than zero-filled, because a
   zero depth channel is a lie the model would learn. They are named in
   cache_manifest.json.

5. Downsampling uses PIL's BOX filter (area average), not BILINEAR. Our reductions
   are 4x (640x480 -> 160x120) and 2x (320x240 -> 160x120); bilinear samples a 2x2
   neighbourhood and aliases at those ratios, throwing away exactly the high-
   frequency edge detail that IR and colormapped depth carry. BOX is what
   torchvision's antialias=True approximates.

SHARD GEOMETRY, and why the uncropped one is not 96x192.

    thermal        160x120x3   native 320x240, no crop (its FOV is already 59% of
                               IR linearly -- a transferred box gives IoU 0.35)
    depthir_raw    160x120x4   FULL FRAME, 4:3 aspect preserved
    depthir_crop    96x192x4   person crop, 1:2, margin x1.20, aspect held

depthir_raw is the control arm of A/B #3 (crop on/off, expected +3 to +8 pp), so
it has to be a fair control. Letterboxing a 4:3 frame into 96x192 would leave
62.5% padding and squeeze the person -- median 5.9% of frame by silhouette -- into
about 8 pixels of height. That is a strawman that would "prove" the crop helps by
construction. 160x120 preserves the native aspect at 19,200 px against the crop's
18,432, a 1.04x compute ratio, so only what the pixels SHOW differs. The backbone
is identical either way; ResNet-18 global-average-pools.

Usage:
    python src/preprocess.py --manifest /tmp/m.csv --out cache/ \
        --streams thermal,depthir_raw --workers 6
    # A/B #3's treatment arm. Needs cache/crops.csv, which src/person_crop.py
    # writes and .gitignore keeps out of the repo (CUHK-X License v2.0 §5.2).
    # m_FRESH, not m_derived: select_clips reads depth_n / ir_n, and the
    # derived manifest carries only clip_id, split, user, action_id.
    python src/preprocess.py --manifest /tmp/m_fresh.csv --out cache/ \
        --streams depthir_crop --workers 6
    python src/preprocess.py --out cache/ --verify-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import natural_key  # noqa: E402  the same ordering rule, imported not copied

TRAIN_ROOT = Path.home() / "cuhk-x/extracted/HAR/data"
TEST_ROOT = Path.home() / "cuhk-x/test_extracted/small_model_track_test"

# (W, H, C). Rationale for each in the module docstring.
STREAMS = {
    "thermal": {"size_wh": (160, 120), "channels": 3, "mods": ("Thermal",)},
    "depthir_raw": {"size_wh": (160, 120), "channels": 4, "mods": ("Depth_Color", "IR")},
    "depthir_crop": {"size_wh": (96, 192), "channels": 4, "mods": ("Depth_Color", "IR"), "crop": True},
    # A3 (QUEUE §33): the AT-SOURCE square crop — the TIGHT detection box, margin 1.15, aspect 1:1,
    # cut from the 640x480 originals straight to the model's 224x224 (no resample in the dataset).
    "depthir_cropsq": {"size_wh": (224, 224), "channels": 4, "mods": ("Depth_Color", "IR"),
                       "crop": True, "crop_from": "tight", "crop_aspect": 1.0, "crop_margin": 1.15},
    # §96 (a)/(b) (D29): the ORDINAL depth stream — the JET-family rendering decoded to its 254-level ordinal from the 640x480
    # ORIGINALS (before the 4x BOX reduction; `ordinal_planes`), with ∂x/∂y at full resolution, into [ord, ∂x, ∂y, IR]; the
    # reduction averages VALID pixels only (`resize_ordinal`). Full-frame only; needs the depth LUT (`--lut`, a TRAIN statistic).
    "depthir_ord": {"size_wh": (160, 120), "channels": 4, "mods": ("Depth_Color", "IR"), "ordinal": True},
}

# §96 (a)/(b): the ordinal decode's constants. ORD_HOP = person_crop.to_ordinal's snap tolerance (a colour within one hop of the ramp is
# that entry; farther is not depth); ORD_GRAD_GAIN = ordinal levels per pixel per byte step around 128 (±32 levels/px span the byte).
ORD_HOP = 8.0
ORD_GRAD_GAIN = 4.0


def load_depth_lut(path) -> np.ndarray:
    """The (K, 3) uint8 inverse table, far -> near, as {"ramp": [[r, g, b], ...]} (cache/crop_palette.json's shape; the
    tools/build_depth_lut.py files)."""
    d = json.loads(Path(path).read_text())
    ramp = np.asarray(d["ramp"], dtype=np.uint8)
    if ramp.ndim != 2 or ramp.shape[1] != 3 or ramp.shape[0] < 2:
        raise ValueError(f"{path}: not a (K, 3) ramp")
    return ramp


def lut_sha16(lut: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(lut, dtype=np.uint8).tobytes()).hexdigest()[:16]


def to_ordinal_np(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) uint8 -> int16 ordinal, -1 where invalid (black or off-ramp). THE SAME RULE as person_crop.to_ordinal — the
    nearest ramp entry within ORD_HOP, black never valid — re-implemented here in numpy so the inference path never imports
    person_crop (scipy); tests/check_ord_stream.py asserts the two agree."""
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2].astype(np.uint32)
    u, inv = np.unique(packed, return_inverse=True)
    cols = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], axis=1).astype(np.float64)
    d = np.linalg.norm(cols[:, None, :] - lut[None, :, :].astype(np.float64), axis=2)
    idx = d.argmin(axis=1).astype(np.int16)
    idx[d.min(axis=1) > ORD_HOP] = -1
    idx[u == 0] = -1
    return idx[inv].reshape(packed.shape)


def ordinal_planes(rgb: np.ndarray, lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) uint8 RGB -> ([ord, dx, dy] uint8 (H, W, 3), validity masks bool (H, W, 3)) at the SOURCE resolution.
    ord: 0 = no return / off-ramp, 1..K = far -> near. dx/dy: central differences of the ordinal where BOTH neighbours are
    valid, 128 + ORD_GRAD_GAIN * g clipped to [0, 255]; 128 (neutral) and masked elsewhere."""
    o = to_ordinal_np(rgb, lut)
    valid = o >= 0
    of = o.astype(np.float32)
    ord8 = np.where(valid, o.astype(np.int32) + 1, 0).astype(np.uint8)
    H, W = o.shape
    gx = np.zeros((H, W), np.float32); gy = np.zeros((H, W), np.float32)
    vx = np.zeros((H, W), bool); vy = np.zeros((H, W), bool)
    vx[:, 1:-1] = valid[:, :-2] & valid[:, 2:]
    gx[:, 1:-1] = (of[:, 2:] - of[:, :-2]) / 2.0
    vy[1:-1, :] = valid[:-2, :] & valid[2:, :]
    gy[1:-1, :] = (of[2:, :] - of[:-2, :]) / 2.0
    gx8 = np.where(vx, np.clip(np.round(128.0 + ORD_GRAD_GAIN * gx), 0, 255), 128).astype(np.uint8)
    gy8 = np.where(vy, np.clip(np.round(128.0 + ORD_GRAD_GAIN * gy), 0, 255), 128).astype(np.uint8)
    return np.stack([ord8, gx8, gy8], axis=-1), np.stack([valid, vx, vy], axis=-1)


def resize_ordinal(planes: np.ndarray, masks: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """The validity-aware BOX reduction: each output pixel is the mean of the VALID source pixels of its block (an all-invalid
    block -> 0 for ord, 128 for the gradients). The source must be an integer multiple of the target (640x480 -> 160x120 = 4x)."""
    Wo, Ho = size_wh
    H, W, _ = planes.shape
    if H % Ho or W % Wo:
        raise ValueError(f"the ordinal reduction needs an integer factor: {W}x{H} -> {Wo}x{Ho}")
    fy, fx = H // Ho, W // Wo
    out = np.empty((Ho, Wo, 3), np.uint8)
    for c, fill in ((0, 0.0), (1, 128.0), (2, 128.0)):
        v = planes[..., c].astype(np.float64).reshape(Ho, fy, Wo, fx)
        m = masks[..., c].astype(np.float64).reshape(Ho, fy, Wo, fx)
        num = (v * m).sum(axis=(1, 3)); den = m.sum(axis=(1, 3))
        mean = np.where(den > 0, num / np.maximum(den, 1.0), fill)
        out[..., c] = np.clip(np.round(mean), 0, 255).astype(np.uint8)
    return out
# The centred square for a clip the detector could not box (A3's analogue of CROP_FALLBACK_BOX).
CROP_FALLBACK_SQUARE = (80, 0, 560, 480)

# ── the person crop (A/B #3) ────────────────────────────────────────────────
# THE CROP IS TAKEN FROM THE ORIGINAL 640x480 FRAME, NEVER FROM A SHARD.
# The uncropped shard is 160x120, and M-05/finding 19 put the median person at
# 155x291 ORIGINAL px = 39x73 shard px. Cropping a shard to the 96x192 target
# would therefore UPSAMPLE by ~2.5x, and A/B #3 would compare a sharp full frame
# against a blurred crop -- a null would say nothing about framing. Cropping here,
# before the resize, is the only place the comparison is fair.
CROP_BOX_CSV = Path(__file__).resolve().parent.parent / "cache/crops.csv"
# person_crop.py measures and clamps in 640x480 and this module re-crops there. A
# frame of any other size means the boxes were measured against different pixels,
# which is a silent misalignment rather than a bad frame -- so it RAISES.
CROP_SOURCE_WH = (640, 480)
# The largest 1:2 region centred in a 640x480 frame. The stated policy for a clip
# the detector could not box (status != "ok"): 0.72% of train and 5.93% of test.
# Centring neither invents a person nor drops the clip, and it is the same window
# for every such clip, so nothing about it is learnable.
CROP_FALLBACK_BOX = (200, 0, 440, 480)


class CropGeometryError(RuntimeError):
    """The source frame is not the geometry the crop boxes were measured in."""


def fit_square(box, W=640, H=480, margin=1.15):
    """A3: a 1:1, margin-expanded, frame-clamped square around the TIGHT box (person not stretched)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(1.0, max(x1 - x0, y1 - y0) * margin)
    side = min(side, H)
    cx = min(max(cx, side / 2), W - side / 2)
    cy = min(max(cy, side / 2), H - side / 2)
    return (int(round(cx - side / 2)), int(round(cy - side / 2)),
            int(round(cx + side / 2)), int(round(cy + side / 2)))


def load_crop_boxes(path: Path = CROP_BOX_CSV, source: str = "fitted", aspect: float = 0.5,
                    margin: float = 1.2) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    """(split, clip_id) -> (x0, y0, x1, y1) in ORIGINAL pixels, `status == "ok"` only.

    `source="fitted"` returns the CSV's 1:2 envelope box as written (D7's stream);
    `source="tight"` refits a square around the CSV's tight detection box (A3, QUEUE §33).

    Rows the detector failed are deliberately ABSENT rather than carried with a
    sentinel: the caller substitutes CROP_FALLBACK_BOX and records which clips got
    it, so the fallback rate is a number in cache_manifest.json instead of a
    property nobody measured.
    """
    if not path.exists():
        raise SystemExit(
            f"no {path} -- regenerate it with `python -m src.person_crop --manifest /tmp/m_fresh.csv`"
            " (it is corpus-derived, so it is gitignored and does not survive a clone)"
        )
    out: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            if source == "tight":
                tb = (int(r["tight_x0"]), int(r["tight_y0"]), int(r["tight_x1"]), int(r["tight_y1"]))
                out[(r["split"], r["clip_id"])] = fit_square(tb, margin=margin)
            else:
                out[(r["split"], r["clip_id"])] = (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"]))
    return out

# Trap 2. The global frame index is the shared clock; take it, not the whole name.
#
# TWO SCHEMAS, and the second one is undocumented. Most of the corpus ships
#    IR_2025-06-10_10-43-49.016_00000151.png / Depth_..._00000151_Color.png
# but 23 train clips ship a SHORT form carrying no wall clock at all:
#    IR_00000074.png / Depth_00000074_Color.png
# 1,012 IR files and 973 Depth files. A pattern that requires the date token
# matches none of them, and pairing on token+index then yields an EMPTY
# intersection -- so those clips store zero frames and nothing raises. That is
# how this was found: check_cache.py compared stored lengths against the
# manifest and 21 clips came back 0.
#
# The date is therefore optional, and pairing keys on the INDEX ALONE, which is
# the shared clock the schemas actually agree on. This is the same family as the
# IMU traps in manifest.py: the corpus has more than one naming convention, and
# only content-level keys survive all of them.
_IR_RE = re.compile(r"^IR_(?:.+_)?(?P<idx>\d+)\.png$", re.I)
_DEPTH_RE = re.compile(r"^Depth_(?:.+_)?(?P<idx>\d+)_Color\.png$", re.I)
_THERMAL_RE = re.compile(r"^frame_(?P<idx>\d+)\.jpg$", re.I)


def _listdir(d: Path) -> list[str]:
    try:
        with os.scandir(d) as it:
            return [e.name for e in it if e.is_file() and not e.name.startswith(".")]
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []


def thermal_frames(d: Path):
    """Sorted by counter. Bare counter filenames -- no wall clock (finding 7)."""
    keyed, unmatched = [], []
    for n in _listdir(d):
        if not n.lower().endswith(".jpg"):
            continue
        m = _THERMAL_RE.match(n)
        (keyed if m else unmatched).append((int(m.group("idx")), n) if m else n)
    keyed.sort()
    # Never drop a frame because its name is exotic; order the rest naturally.
    paths = [d / n for _, n in keyed] + [d / n for n in sorted(unmatched, key=natural_key)]
    return paths, {"unparsed": unmatched, "duplicate_index": 0, "unpaired": 0}


def _by_index(d: Path, rx: re.Pattern, suffix: str) -> tuple[dict[int, Path], list[str], int]:
    """Map global frame index -> path. Also returns names that did not parse."""
    out: dict[int, Path] = {}
    unparsed: list[str] = []
    dupes = 0
    for n in _listdir(d):
        if not n.lower().endswith(suffix):
            continue
        m = rx.match(n)
        if not m:
            unparsed.append(n)
            continue
        i = int(m.group("idx"))
        if i in out:
            dupes += 1
        out[i] = d / n
    return out, unparsed, dupes


def depthir_frames(ir_dir: Path, depth_dir: Path):
    """Pair IR and Depth on the GLOBAL FRAME INDEX (trap 1 + trap 2).

    Returns (ir_paths, depth_paths, diag) with the two lists equal-length and
    ordered by index. Keying on the index alone rather than on the filename
    survives both naming schemas and would survive a clip that mixed them.

    An index present in only one stream is dropped from both, so the pair is
    always aligned -- but the count is reported rather than swallowed, because a
    silent drop here is invisible downstream and cost this module one rebuild.
    """
    ir_by_idx, ir_unparsed, ir_dupes = _by_index(ir_dir, _IR_RE, ".png")
    dp_by_idx, dp_unparsed, dp_dupes = _by_index(depth_dir, _DEPTH_RE, ".png")

    shared = sorted(set(ir_by_idx) & set(dp_by_idx))
    diag = {
        "unparsed": ir_unparsed + dp_unparsed,
        "duplicate_index": ir_dupes + dp_dupes,
        "unpaired": len(set(ir_by_idx) ^ set(dp_by_idx)),
    }
    return [ir_by_idx[i] for i in shared], [dp_by_idx[i] for i in shared], diag


def clip_dirs(clip_id: str, split: str, mod: str) -> Path:
    if split == "train":
        action, user, trial = clip_id.split("/")
        return TRAIN_ROOT / mod / action / user / trial
    return TEST_ROOT / clip_id / mod


def frame_paths(clip_id: str, split: str, stream: str):
    """One tuple per output frame: (thermal,) or (ir, depth). Plus diagnostics."""
    if stream == "thermal":
        paths, diag = thermal_frames(clip_dirs(clip_id, split, "Thermal"))
        return [(p,) for p in paths], diag
    ir, depth, diag = depthir_frames(
        clip_dirs(clip_id, split, "IR"), clip_dirs(clip_id, split, "Depth_Color")
    )
    return list(zip(ir, depth)), diag


def _open_resized(
    path: Path, size_wh: tuple[int, int], mode: str, box: tuple[int, int, int, int] | None = None,
    lut: np.ndarray | None = None,
) -> np.ndarray | None:
    """Decode -> convert -> optional ORIGINAL-pixel crop -> BOX-resize.

    None means unreadable (trap 3). A wrong SOURCE SIZE under a crop is not that:
    it would silently box the wrong pixels, so it raises past the None path. The
    ordering of the two excepts is what keeps it from being swallowed as a bad
    frame and repaired by reuse.
    """
    if mode == "ORD":
        # §96 (a)/(b): decode the ordinal from the ORIGINAL frame, then reduce with validity — never resize the colours first
        if lut is None:
            raise ValueError("the ORD plane needs the depth LUT (a TRAIN statistic)")
        if box is not None:
            raise CropGeometryError("the ordinal stream is full-frame only")
        try:
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
        if rgb.shape[1] != CROP_SOURCE_WH[0] or rgb.shape[0] != CROP_SOURCE_WH[1]:
            raise CropGeometryError(f"{path} is {rgb.shape[1]}x{rgb.shape[0]}, the ordinal decode expects {CROP_SOURCE_WH}")
        planes, masks = ordinal_planes(rgb, lut)
        return resize_ordinal(planes, masks, size_wh)
    try:
        with Image.open(path) as im:
            im = im.convert(mode)
            if box is not None:
                if im.size != CROP_SOURCE_WH:
                    raise CropGeometryError(
                        f"{path} is {im.size}, but crop boxes are measured in {CROP_SOURCE_WH}"
                    )
                im = im.crop(box)
            if im.size != size_wh:
                im = im.resize(size_wh, Image.BOX)
            return np.asarray(im, dtype=np.uint8)
    except CropGeometryError:
        raise
    except Exception:
        return None


# (name, destination channel slice, PIL mode, index into the per-frame path tuple)
PLANES = {
    "thermal": [("rgb", slice(0, 3), "RGB", 0)],
    "depthir_raw": [("depth", slice(0, 3), "RGB", 1), ("ir", slice(3, 4), "L", 0)],
    "depthir_crop": [("depth", slice(0, 3), "RGB", 1), ("ir", slice(3, 4), "L", 0)],
    "depthir_cropsq": [("depth", slice(0, 3), "RGB", 1), ("ir", slice(3, 4), "L", 0)],
    "depthir_ord": [("depth", slice(0, 3), "ORD", 1), ("ir", slice(3, 4), "L", 0)],   # §96 (a)/(b): [ord, dx, dy] + IR
}


def build_clip(
    paths: list[tuple[Path, ...]], stream: str, box: tuple[int, int, int, int] | None = None,
    lut: np.ndarray | None = None, size_wh: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Decode one clip, repairing each channel group independently (trap 3).

    Returns (frames NHWC uint8, stats). `stats['absent']` names channel groups
    with no good frame anywhere in the clip -- those stay zero and the caller
    records the clip so the dataset can skip that branch rather than feed the
    model a plane it never saw in training.

    `box` is the per-clip person crop in ORIGINAL pixels, applied to EVERY plane.
    M-07 puts Depth_Color and IR at exactly (0,0) registration and both ship at
    640x480, so one box is correct for both; cropping only depth would shear the
    two channel groups apart and nothing downstream would notice.
    """
    spec = STREAMS[stream]
    # §124 (P-A): `size_wh` overrides the stream's cache resolution FOR THIS CALL ONLY. The CLI's --size-wh mutates
    # STREAMS (main(), below) because one build has one resolution; the INFERENCE path cannot, because one container
    # may hold a thermal member decoded at 320x240 beside a depth+IR member decoded at 160x120 in the same process.
    # None is the default and reads the table exactly as before, so every existing caller is bitwise unchanged.
    W, H = spec["size_wh"] if size_wh is None else (int(size_wh[0]), int(size_wh[1]))
    n = len(paths)
    out = np.zeros((n, H, W, spec["channels"]), dtype=np.uint8)
    stats = {"bad": 0, "reused": 0, "backfilled": 0, "absent": []}

    for name, dst, mode, src in PLANES[stream]:
        good = [False] * n
        for i, tup in enumerate(paths):
            a = _open_resized(tup[src], (W, H), mode, box, lut=lut)
            if a is None:
                stats["bad"] += 1
                continue
            out[i, :, :, dst] = a if a.ndim == 3 else a[..., None]
            good[i] = True

        first_good = next((i for i, g in enumerate(good) if g), None)
        if first_good is None:
            stats["absent"].append(name)
            continue
        # Reuse the previous good frame; the head back-fills from the first one.
        last = None
        for i in range(n):
            if good[i]:
                last = i
            elif last is not None:
                out[i, :, :, dst] = out[last, :, :, dst]
                stats["reused"] += 1
            else:
                out[i, :, :, dst] = out[first_good, :, :, dst]
                stats["backfilled"] += 1

    return out, stats


_MM: np.memmap | None = None
_LUT: np.ndarray | None = None


def _init_worker(shard_path: str, total: int, H: int, W: int, C: int, lut: np.ndarray | None = None) -> None:
    """Map the shard once per worker process, not once per clip.

    Mapping an 11 GB file 2,891 times is pure syscall overhead, and msync on a
    mapping that size is not free either. Opened here, it lives for the pool.
    """
    global _MM, _LUT
    _MM = np.memmap(shard_path, dtype=np.uint8, mode="r+", shape=(total, H, W, C))
    _LUT = lut


def _worker(job) -> tuple[str, dict]:
    """Decode one clip straight into its slice of the shared memmap.

    Workers write disjoint byte ranges of a MAP_SHARED mapping, which is coherent
    across processes -- so no pixel data crosses the IPC boundary. munmap at
    process exit flushes them.
    """
    clip_id, stream, offset, paths, box = job
    frames, stats = build_clip(paths, stream, box, lut=_LUT)
    if len(frames):
        _MM[offset : offset + len(frames)] = frames
    stats["all_zero"] = bool(len(frames)) and not frames.any()
    return clip_id, stats


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_shard(stream: str, split: str, clips: list[str], out_dir: Path, workers: int, lut: np.ndarray | None = None) -> dict:
    spec = STREAMS[stream]
    W, H = spec["size_wh"]
    C = spec["channels"]
    shard = out_dir / f"{stream}_{split}.mm"

    t0 = time.perf_counter()
    print(f"[preprocess] {stream}/{split}: listing {len(clips)} clips…", file=sys.stderr)
    listed = [frame_paths(c, split, stream) for c in clips]
    per_clip = [p for p, _ in listed]
    counts = [len(p) for p in per_clip]

    # A frame dropped at listing time is invisible downstream. Surface it here.
    unparsed = {clips[i]: d["unparsed"] for i, (_, d) in enumerate(listed) if d["unparsed"]}
    unpaired = {clips[i]: d["unpaired"] for i, (_, d) in enumerate(listed) if d["unpaired"]}
    dupes = {clips[i]: d["duplicate_index"] for i, (_, d) in enumerate(listed) if d["duplicate_index"]}
    for label, dd in (("UNPARSED FILENAME", unparsed), ("UNPAIRED INDEX", unpaired), ("DUPLICATE INDEX", dupes)):
        if dd:
            n = sum(len(v) if isinstance(v, list) else v for v in dd.values())
            print(
                f"[preprocess] 🔴 {label}: {n} in {len(dd)} clips, e.g. "
                f"{list(dd.items())[:2]} — these frames are NOT in the shard",
                file=sys.stderr,
            )
    # Per-clip crop box, index-aligned with `clips` exactly as `offsets` is.
    # A clip the detector failed takes CROP_FALLBACK_BOX and is NAMED, so the
    # fallback rate is a recorded number rather than an unmeasured property.
    crop_boxes: list[tuple[int, int, int, int]] | None = None
    crop_fallback: list[str] = []
    if spec.get("crop"):
        tight = spec.get("crop_from") == "tight"
        known = (load_crop_boxes(source="tight", margin=spec.get("crop_margin", 1.15)) if tight
                 else load_crop_boxes())
        fallback = CROP_FALLBACK_SQUARE if tight else CROP_FALLBACK_BOX
        crop_boxes = []
        for c in clips:
            b = known.get((split, c))
            if b is None:
                crop_fallback.append(c)
                b = fallback
            crop_boxes.append(b)
        print(
            f"[preprocess] {stream}/{split}: crop boxes for {len(clips)-len(crop_fallback)}/{len(clips)} "
            f"clips · {len(crop_fallback)} on the centre fallback "
            f"({100*len(crop_fallback)/max(1,len(clips)):.2f}%)",
            file=sys.stderr,
        )

    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    nbytes = total * H * W * C

    print(
        f"[preprocess] {stream}/{split}: {total:,} frames, {nbytes/1e9:.2f} GB "
        f"({time.perf_counter()-t0:.1f}s to list)",
        file=sys.stderr,
    )

    # Allocate, then let workers write disjoint slices of it.
    np.memmap(shard, dtype=np.uint8, mode="w+", shape=(max(total, 1), H, W, C)).flush()

    jobs = [
        (clips[i], stream, int(offsets[i]), per_clip[i], crop_boxes[i] if crop_boxes else None)
        for i in range(len(clips))
        if counts[i] > 0
    ]

    bad = reused = backfilled = 0
    all_zero: list[str] = []
    channel_absent: dict[str, list[str]] = {}
    t1 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(shard), max(total, 1), H, W, C, lut),
    ) as ex:
        for clip_id, st in ex.map(_worker, jobs, chunksize=8):
            bad += st["bad"]
            reused += st["reused"]
            backfilled += st["backfilled"]
            for name in st["absent"]:
                channel_absent.setdefault(name, []).append(clip_id)
            if st["all_zero"]:
                all_zero.append(clip_id)
            done += 1
            if done % 400 == 0:
                rate = done / (time.perf_counter() - t1)
                print(
                    f"[preprocess]   {done}/{len(jobs)} clips  {rate:.0f} clip/s  "
                    f"eta {(len(jobs)-done)/rate:.0f}s",
                    file=sys.stderr,
                )

    dt = time.perf_counter() - t1
    print(f"[preprocess] {stream}/{split}: decoded in {dt:.0f}s, hashing…", file=sys.stderr)
    digest = sha256_file(shard)

    crop_meta = {}
    if crop_boxes is not None:
        crop_meta = {
            # IRREVERSIBLE, unlike everything in `not_baked_in`. Kept here so the
            # dataset can recover what was framed -- the dead strip is 40 ORIGINAL
            # px and a crop window starting past column 40 does not contain it.
            "crop_boxes": [list(b) for b in crop_boxes],
            "crop_source_wh": list(CROP_SOURCE_WH),
            "crop_fallback_box": list(CROP_FALLBACK_SQUARE if spec.get("crop_from") == "tight" else CROP_FALLBACK_BOX),
            "crop_from": spec.get("crop_from", "fitted"),
            "crop_fallback_clips": crop_fallback,
            "crop_source_csv": str(CROP_BOX_CSV),
        }

    return {
        "shard": shard.name,
        "stream": stream,
        "split": split,
        "shape": [total, H, W, C],
        **crop_meta,
        "dtype": "uint8",
        "layout": "NHWC",
        "channel_order": "RGB" if C == 3 else "depth_R,depth_G,depth_B,IR",
        "resample": "PIL.Image.BOX",
        "bytes": nbytes,
        "sha256": digest,
        "n_clips": len(clips),
        "clip_ids": clips,
        "offsets": offsets.tolist(),
        "n_frames_total": total,
        "n_clips_zero_frames": int(sum(1 for c in counts if c == 0)),
        "n_clips_one_frame": int(sum(1 for c in counts if c == 1)),
        "bad_frames": bad,
        "frames_repaired_by_reuse": reused,
        "frames_repaired_by_backfill": backfilled,
        # A channel group with no good frame anywhere. The dataset must SKIP that
        # branch for these clips -- never feed the model an all-zero plane.
        "channel_absent": channel_absent,
        "clips_all_zero_pixels": all_zero,
        # Frames the listing step could not place. Must be empty; a non-empty
        # value means a third naming schema exists and the shard is incomplete.
        "clips_with_unparsed_filenames": {k: len(v) for k, v in unparsed.items()},
        "clips_with_unpaired_index": unpaired,
        "clips_with_duplicate_index": dupes,
        "decode_seconds": round(dt, 1),
        "depth_lut_sha16": lut_sha16(lut) if lut is not None else None,   # §96 (a)/(b): the decode's LUT, for the stamp check
    }


def select_clips(rows: list[dict], stream: str, split: str) -> tuple[list[str], list[str]]:
    """Clips this shard covers, and the ones deliberately excluded (trap 4)."""
    rs = [r for r in rows if r["split"] == split]
    if stream == "thermal":
        return [r["clip_id"] for r in rs if int(r["thermal_n"]) > 0], []

    # Gate on DEPTH, which is 3 of the 4 channels and is what the person crop
    # thresholds. A clip with depth but no readable IR is still worth storing --
    # build_clip discovers that by CONTENT and flags it, which is the distinction
    # the manifest's file-count columns cannot make. An IR-only clip is excluded:
    # 3 zeroed channels is not a training example, it is a fabrication.
    keep, excluded = [], []
    for r in rs:
        if int(r["depth_n"]) > 0:
            keep.append(r["clip_id"])
        elif int(r["ir_n"]) > 0:
            excluded.append(r["clip_id"])
    return keep, excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m.csv"))
    ap.add_argument("--out", type=Path, default=Path("cache"))
    ap.add_argument("--streams", default="thermal,depthir_raw")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--verify-only", action="store_true", help="re-hash existing shards, build nothing")
    # T2 (QUEUE §25): the SOURCE-resolution cache. Overrides the raw streams' size_wh
    # (thermal 320x240 = its native size, no resample; depth/IR = a 2x BOX reduction of
    # 640x480). Refused for the cropped stream: its 96x192 geometry is a crop decision,
    # not a resolution one, and the dead-strip arithmetic on the ClipRef assumes it.
    ap.add_argument("--size-wh", default=None, metavar="WxH",
                    help="override the raw streams' cache resolution, e.g. 320x240 (QUEUE §25)")
    ap.add_argument("--lut", type=Path, default=None, help="§96 (a)/(b): the depth LUT json for the ordinal stream (a TRAIN statistic — tools/build_depth_lut.py)")
    args = ap.parse_args()
    if args.size_wh:
        W, H = (int(v) for v in args.size_wh.lower().split("x"))
        for stream in args.streams.split(","):
            stream = stream.strip()
            if STREAMS.get(stream, {}).get("crop") or STREAMS.get(stream, {}).get("ordinal"):
                raise SystemExit(f"--size-wh is refused for the cropped / ordinal stream {stream!r}")
            if stream in STREAMS:
                STREAMS[stream]["size_wh"] = (W, H)
        print(f"[preprocess] size override: {args.streams} at {W}x{H} (BOX; native sizes are skipped)",
              file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    cm_path = args.out / "cache_manifest.json"

    if args.verify_only:
        if not cm_path.exists():
            raise SystemExit(f"no {cm_path}")
        cm = json.loads(cm_path.read_text())
        bad = 0
        for s in cm["shards"]:
            p = args.out / s["shard"]
            if not p.exists():
                print(f"  MISSING  {s['shard']}", file=sys.stderr)
                bad += 1
                continue
            size_ok = p.stat().st_size == s["bytes"]
            digest = sha256_file(p)
            ok = size_ok and digest == s["sha256"]
            print(f"  {'ok  ' if ok else 'FAIL'}  {s['shard']}  {s['bytes']/1e9:.2f} GB", file=sys.stderr)
            bad += 0 if ok else 1
        print(f"\n{len(cm['shards'])-bad} ok, {bad} failed", file=sys.stderr)
        return 1 if bad else 0

    with args.manifest.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[preprocess] manifest: {len(rows)} rows", file=sys.stderr)

    shards, excluded_all = [], {}
    for stream in args.streams.split(","):
        stream = stream.strip()
        if stream not in STREAMS:
            raise SystemExit(f"unknown stream {stream!r}; known: {list(STREAMS)}")
        for split in args.splits.split(","):
            split = split.strip()
            clips, excluded = select_clips(rows, stream, split)
            if excluded:
                excluded_all[f"{stream}_{split}"] = excluded
            lut = None
            if STREAMS[stream].get("ordinal"):
                if args.lut is None:
                    raise SystemExit(f"{stream} needs --lut (the depth LUT, a TRAIN statistic — tools/build_depth_lut.py)")
                lut = load_depth_lut(args.lut)
                print(f"[preprocess] {stream}: the ordinal decode uses {args.lut} (sha16 {lut_sha16(lut)}, {lut.shape[0]} levels)", file=sys.stderr)
            shards.append(build_shard(stream, split, clips, args.out, args.workers, lut=lut))

    cm = json.loads(cm_path.read_text()) if cm_path.exists() else {"shards": []}
    by_name = {s["shard"]: s for s in cm["shards"]}
    for s in shards:
        by_name[s["shard"]] = s
    cm["shards"] = [by_name[k] for k in sorted(by_name)]
    cm["excluded_clips"] = {**cm.get("excluded_clips", {}), **excluded_all}
    cm["not_baked_in"] = [
        "normalisation",
        "depth dead strip (finding 21) — applied at train time, so it stays A/B-able",
        "temporal sampling and repeat-padding",
        "augmentation",
    ]
    cm["manifest_source"] = str(args.manifest)
    cm_path.write_text(json.dumps(cm, indent=2) + "\n")

    print(f"\n[preprocess] wrote {cm_path}", file=sys.stderr)
    for s in shards:
        print(
            f"  {s['shard']:26} {str(s['shape']):24} {s['bytes']/1e9:7.2f} GB  "
            f"{s['n_clips']:>5} clips  {s['bad_frames']:>4} bad  "
            f"{sum(len(v) for v in s['channel_absent'].values()):>2} chan-absent",
            file=sys.stderr,
        )
    total_gb = sum(s["bytes"] for s in shards) / 1e9
    print(f"  {'TOTAL':26} {'':24} {total_gb:7.2f} GB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
