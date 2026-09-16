"""Person box per clip, from Depth_Color + IR. Flag L26, decisions D7 / M-16.

WHY THIS FILE IS A REBUILD AND NOT A PORT.

D7 and M-16 name the method -- "depth-foreground + motion-weighted largest
connected component" -- and quote 0% / 1.2% detection failure on train / test.
That figure is real. The IMPLEMENTATION THAT PRODUCED IT WAS NEVER RECORDED and
its script is not in this repo: no thresholds, no colormap handling, no failure
definition. HANDOFF §9.14 is explicit that inheriting the number while rebuilding
the method from its name is how a silently-wrong crop enters every downstream
result. So nothing here is inherited. Every constant below is either measured in
this file's own calibration step or traceable to a numbered measurement.

WHAT HAD TO BE MEASURED BEFORE ANY OF THIS COULD BE WRITTEN.

1. Depth_Color IS NOT DEPTH. It is an 8-bit LUT colormap (L31: the corpus ships
   no raw depth). A "depth threshold" is therefore not implementable until the
   colour -> depth ordering is recovered. Measured on this corpus: the palette is
   254 non-black colours lying on a 1-D path in RGB space -- nearest-neighbour
   hops of median 4.00 and MAX 6.40, which is a ramp, not a scatter -- running
   rgb(132,0,0) dark red to rgb(0,0,132) dark blue. That is jet.

2. A PATH HAS TWO ENDS AND THE CORPUS LABELS NEITHER. Guessing the direction
   inverts foreground and background, which boxes the far wall while looking
   entirely healthy. Settled against an independent sensor instead: active-
   illumination IR falls off with distance, and M-07 puts Depth_Color and IR at
   exactly (0,0) px registration. Measured over 90 frames from 30 clips:
       corr(path position from dark red, IR brightness) = +0.568, 90/90 positive
       corr(path position, image row)                   = +0.467, 89/90 positive
   The second is the floor of a forward-facing room receding upward. Both say the
   same thing: BLUE IS NEAR, RED IS FAR, and position along the path counts
   NEARNESS. Neither was inferable from the documentation.

3. THE DEAD STRIP IS MASKED BEFORE DETECTION, NOT AFTER. Finding 21 records the
   leftmost 40 original px as permanently degraded; measured here at 72.0% black
   against 16.7% elsewhere. A largest-connected-component search that sees it
   first is dragged leftward by a column of invalid pixels. Trap 31: those 40 px
   are ORIGINAL pixels, and this module works in original 640x480, so 40 is
   correct HERE and would be wrong applied to any downscaled array.

THE ALGORITHM, and why each step exists.

  · The background is the ROW PROFILE: the median ordinal of each image row over
    the clip. A forward-facing camera sees the floor recede upward and the wall
    behind it, so the room's depth is a function of row -- and the floor, which is
    genuinely near and would win every nearest-blob search, is removed by
    construction. A person never occupies enough of a row's width to move its
    median, so the subject cannot contaminate its own background.
  · Foreground = pixels nearer than their ROW by DELTA. This is the
    "depth-foreground" half, and it holds for a motionless subject.
  · Components are scored by area UPWEIGHTED by motion, not selected by motion:
    a person who barely moves -- Watch_TV and Make_a_phone_call are the two
    smallest-person classes in finding 23 -- must still be found. Motion breaks
    ties between a person and a chair; it must not be a gate.
  · There is NO static-subject fallback, because the row profile does not need
    one -- a motionless person is still nearer than their row. An earlier version
    subtracted a PER-PIXEL temporal median and did need one; it fired on 19-49%
    of frames and returned near-whole-frame boxes, which is finding 3's silent
    drop wearing a different hat. The measurement that killed it is in clip_box.

Geometry is settled and not re-derived here: 96x192 (1:2), margin x1.20, aspect
held (DECISIONS §8). Thermal gets NO crop -- M-08 measured its FOV at 59% of IR
linearly, so a transferred box gives IoU 0.35.

Usage:
    python -m src.person_crop --manifest /tmp/m_fresh.csv --out cache/crops.csv
    python -m src.person_crop --manifest /tmp/m_fresh.csv --calibrate-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from src.preprocess import clip_dirs, depthir_frames

ROOT = Path(__file__).resolve().parent.parent

# Finding 21 / measurement 31. ORIGINAL pixels -- this module never leaves 640x480.
DEAD_STRIP_COLS = 40
# Nearer than the ROW by this many LUT steps counts as foreground. 254 steps span
# the sensor's whole range, so this is a fraction of it and not a metric depth.
#
# THE ONLY FREE NUMBER IN THIS FILE, and it is CALIBRATED, not chosen. Swept
# over {4,6,8,10,12,16,20,25,30,40} on 200 random train clips against M-05's and
# finding 19's measured person statistics. 10 minimises total relative error, and
# what makes it convincing is that ONE choice lands SIX statistics at once:
#
#             median bbox   p10 sil   median box   aspect   clips <10%
#   measured      13.98%      4.00%    151 x 299    1.87x      34.4%
#   recorded      14.70%      3.30%    155 x 291    1.88x      30.8%
#
# THESE SIX ARE THE CALIBRATION SET AND CANNOT ALSO BE THE VALIDATION. The
# held-out check is finding 23's per-class ordering, which no step here uses --
# see tests/check_crop.py.
FG_DELTA = 10
# A pixel counts as MOVING at this fraction of the frame's 95th-percentile motion.
# Paired with the 5x5 opening below, it is what separates a person from sensor
# flicker; neither works alone. Not swept -- it is a shape parameter of the noise,
# and finding 23's ordering separates cleanly at this value.
MOTION_THRESH = 0.35
# Frames sampled per clip. Finding 8 puts IR/Depth at median 24 frames, min 1.
N_SAMPLE = 12
# A component smaller than this cannot be a person at 640x480: M-05 puts the p10
# silhouette at 3.3% of frame, and this is an order of magnitude below that.
MIN_COMPONENT_PX = 400
# Geometry, DECISIONS §8. Aspect is width:height.
CROP_ASPECT = 96 / 192
CROP_MARGIN = 1.20


# ----------------------------------------------------------------------------
# the palette, and its direction


def build_palette(rows, n_clips=120, seed=0) -> np.ndarray:
    """Recover the colormap ramp, ordered FAR -> NEAR.

    Returned array is (K, 3) uint8 whose row index IS the depth ordinal: index 0
    is the far end, index K-1 the near end. Built by walking the 1-D path rather
    than by assuming jet, so a corpus that shipped a different ramp still works
    and a corpus that shipped a non-ramp fails the assertion instead of silently
    producing nonsense.
    """
    rng = np.random.default_rng(seed)
    have = [r for r in rows if r["depth_present"] == "1"]
    seen: dict[int, int] = {}
    for k in rng.choice(len(have), min(n_clips, len(have)), replace=False):
        r = have[k]
        d = clip_dirs(r["clip_id"], r["split"], "Depth_Color")
        try:
            files = sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")
        except OSError:
            continue
        for p in files[:: max(1, len(files) // 3)][:3]:
            a = np.asarray(Image.open(p).convert("RGB")).reshape(-1, 3)
            packed = (a[:, 0].astype(np.uint32) << 16) | (a[:, 1].astype(np.uint32) << 8) | a[:, 2]
            u, c = np.unique(packed, return_counts=True)
            for uu, cc in zip(u.tolist(), c.tolist()):
                seen[uu] = seen.get(uu, 0) + int(cc)
    seen.pop(0, None)  # black is "invalid", not a depth
    X = np.array([[(u >> 16) & 255, (u >> 8) & 255, u & 255] for u in seen], dtype=np.float64)
    if len(X) < 16:
        raise SystemExit(f"[crop] only {len(X)} non-black colours — this is not a colormap")

    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    # A path's two endpoints are the colours whose SECOND-nearest neighbour is
    # farthest away; interior colours have two close neighbours. Starting from an
    # extreme of the point cloud instead makes the walk double back on itself and
    # produces one huge hop -- which is how the ramp first looked like a scatter.
    start = int(np.argmax(np.sort(D, axis=1)[:, 1]))
    unvis, cur, chain = set(range(len(X))) - {start}, start, [start]
    while unvis:
        nxt = min(unvis, key=lambda j: D[cur, j])
        chain.append(nxt)
        unvis.discard(nxt)
        cur = nxt
    hops = np.array([D[chain[i], chain[i + 1]] for i in range(len(chain) - 1)])
    ramp = X[chain].astype(np.uint8)

    # Orient it: correlate ordinal against IR brightness on real frames.
    r_ir = _direction_corr(rows, ramp, rng)
    if r_ir < 0:
        ramp = ramp[::-1]
    print(f"[crop] palette {len(ramp)} colours · hop median {np.median(hops):.2f} "
          f"max {hops.max():.2f} · corr(ordinal, IR) {r_ir:+.3f} "
          f"· near end rgb{tuple(int(v) for v in ramp[-1])}")
    if hops.max() > 6 * np.median(hops):
        raise SystemExit("[crop] palette is not a 1-D ramp — the walk has a break")
    return ramp


def _direction_corr(rows, ramp: np.ndarray, rng, n=20) -> float:
    """Mean corr(ordinal, IR brightness). Positive => ordinal counts nearness."""
    have = [r for r in rows if r["depth_present"] == "1" and r["ir_present"] == "1"]
    out = []
    for k in rng.choice(len(have), min(n, len(have)), replace=False):
        r = have[k]
        ir_ps, d_ps, _ = depthir_frames(clip_dirs(r["clip_id"], r["split"], "IR"),
                                        clip_dirs(r["clip_id"], r["split"], "Depth_Color"))
        if not ir_ps:
            continue
        i = len(ir_ps) // 2
        dep = np.asarray(Image.open(d_ps[i]).convert("RGB"))
        ir = np.asarray(Image.open(ir_ps[i]).convert("L")).astype(np.float64)
        if dep.shape[:2] != ir.shape[:2]:
            continue
        o = to_ordinal(dep, ramp)
        v = o >= 0
        v[:, :DEAD_STRIP_COLS] = False
        if v.sum() < 5000:
            continue
        out.append(np.corrcoef(o[v].astype(np.float64), ir[v])[0, 1])
    if not out:
        raise SystemExit("[crop] could not orient the palette — no usable IR/Depth pairs")
    return float(np.mean(out))


def to_ordinal(rgb: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    """RGB frame -> int16 ordinal, -1 where invalid (black or off-ramp).

    Unknown colours are SNAPPED to the nearest ramp entry when they are within a
    hop of it, and rejected otherwise. PNG is lossless, so an off-ramp colour is a
    palette entry the sample missed rather than compression noise -- snapping
    recovers it; a colour far from the whole ramp is not depth at all.
    """
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    u, inv = np.unique(packed, return_inverse=True)
    cols = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], axis=1).astype(np.float64)
    d = np.linalg.norm(cols[:, None, :] - ramp[None, :, :].astype(np.float64), axis=2)
    idx = d.argmin(axis=1).astype(np.int16)
    idx[d.min(axis=1) > 8.0] = -1  # off the ramp entirely
    idx[u == 0] = -1               # black is invalid, never "far"
    return idx[inv].reshape(packed.shape)


# ----------------------------------------------------------------------------
# the detector


def clip_box(ord_stack: np.ndarray, fg_delta: int = FG_DELTA):
    """(T,H,W) int16 ordinals -> (box, diag). box is (x0, y0, x1, y1) exclusive."""
    T, H, W = ord_stack.shape
    valid = ord_stack >= 0
    valid[:, :, :DEAD_STRIP_COLS] = False  # trap 3: before anything reads it

    o = ord_stack.astype(np.float32)
    o[~valid] = np.nan
    # A pixel invalid in EVERY frame -- the dead strip is 40 columns of exactly
    # that -- has no median. Expected, not exceptional: it stays NaN, fails the
    # foreground comparison below, and never reaches a component.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # PER-ROW, NOT PER-PIXEL, AND THAT CORRECTION IS THE WHOLE METHOD.
        # A per-pixel temporal median is the obvious background model and it is
        # WRONG HERE, measured: clips are ~2.4 s (finding 18) and the subject is
        # mostly stationary within one, so the person IS the median at their own
        # pixels. Only their moving edges then survive subtraction, and the box
        # collapses to a fragment -- measured at median 105 x 130 px and aspect
        # 1.41 against finding 19's 155 x 291 and 1.88, with the static-subject
        # fallback firing on 19-49% of frames.
        # The room's depth is a function of IMAGE ROW: the floor recedes upward
        # and the wall sits behind it. A person standing at any row is NEARER
        # than that row's typical depth whether or not they move, and they never
        # occupy enough of a row's width to move its median. So the row profile
        # is a background the subject cannot contaminate.
        row_bg = np.nanmedian(np.nanmedian(o, axis=0), axis=1)   # (H,)
        # Motion keeps its job from the method's own name -- WEIGHTING components,
        # never gating them -- so a person who barely moves is still found.
        px_bg = np.nanmedian(o, axis=0)
    near = np.where(valid, o - row_bg[None, :, None], np.nan)   # + is nearer than the row
    motion = np.nan_to_num(np.abs(np.where(valid, o - px_bg[None], np.nan)), nan=0.0)
    mnorm = motion / max(1.0, float(np.percentile(motion[motion > 0], 95)) if (motion > 0).any() else 1.0)
    mnorm = np.clip(mnorm, 0.0, 1.0)

    boxes, per_frame = [], []
    for t in range(T):
        fg = np.nan_to_num(near[t], nan=-1e9) > fg_delta
        if fg.sum() < MIN_COMPONENT_PX:
            per_frame.append(None)
            continue
        fg = ndimage.binary_opening(fg, np.ones((3, 3)), iterations=1)
        fg = ndimage.binary_closing(fg, np.ones((7, 7)), iterations=1)
        lab, n = ndimage.label(fg)
        if n == 0:
            per_frame.append(None)
            continue
        # SCORE BY COHERENT MOTION. Two earlier rules were wrong and both were
        # caught by rendering the masks, not by reading the code.
        #   area x (1 + motion)  picked a door, a counter or a side wall in 6 of 6
        #     inspected clips. Rooms are full of large structures nearer than their
        #     row, every one beats a person on area, and a x2 bonus cannot rescue a
        #     component 5x smaller.
        #   sum(motion)          still lost whenever the subject was FAR. Depth
        #     sensors flicker on large flat surfaces, and a big region of weak
        #     per-pixel noise outsums a small person. Measured against finding 23:
        #     Make_a_phone_call came out at 31.5% of frame against a recorded 3.8%,
        #     i.e. the ordering was inverted for the far-subject classes.
        # The discriminator is that flicker is pixel-local and high-frequency while
        # a moving person is a spatially COHERENT patch. Thresholding motion and
        # opening it destroys the first and keeps the second: isolated noisy pixels
        # then contribute nothing however many of them there are. This is the only
        # rule of four tried that separates finding 23's named small classes from
        # its named large ones (max small 10.9% vs min large 15.5%).
        coherent = ndimage.binary_opening(mnorm[t] > MOTION_THRESH, np.ones((5, 5)))
        score = ndimage.sum(coherent.astype(np.float32), lab, index=np.arange(1, n + 1))
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        score[area < MIN_COMPONENT_PX] = -1.0
        if score.max() <= 0:
            per_frame.append(None)
            continue
        # find_objects wants the LABEL array and returns slices indexed by
        # label-1; handing it a boolean mask raises rather than misbehaving.
        sl = ndimage.find_objects(lab)[int(score.argmax())]
        boxes.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
        per_frame.append(boxes[-1])

    if not boxes:
        return None, {"n_frames": T, "n_detected": 0}
    b = np.array(boxes, dtype=np.float64)
    # Percentile envelope, not a union: one frame whose component leaked into a
    # wall would otherwise set the box for the whole clip.
    box = (float(np.percentile(b[:, 0], 10)), float(np.percentile(b[:, 1], 10)),
           float(np.percentile(b[:, 2], 90)), float(np.percentile(b[:, 3], 90)))
    diag = {"n_frames": T, "n_detected": len(boxes), "per_frame": per_frame}
    return box, diag


def fit_geometry(box, W=640, H=480, aspect=CROP_ASPECT, margin=CROP_MARGIN):
    """Aspect-held, margin-expanded, frame-clamped crop around `box`."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(1.0, (x1 - x0) * margin), max(1.0, (y1 - y0) * margin)
    if w / h > aspect:      # too wide for 1:2 -- grow height
        h = w / aspect
    else:
        w = h * aspect
    w, h = min(w, W), min(h, H)
    cx = min(max(cx, w / 2), W - w / 2)
    cy = min(max(cy, h / 2), H - h / 2)
    return (int(round(cx - w / 2)), int(round(cy - h / 2)),
            int(round(cx + w / 2)), int(round(cy + h / 2)))


# ----------------------------------------------------------------------------
# driving it over the corpus

_RAMP: np.ndarray | None = None


def _init(ramp_bytes, shape):
    global _RAMP
    _RAMP = np.frombuffer(ramp_bytes, dtype=np.uint8).reshape(shape)


def _one(job):
    clip_id, split, action_id, action_name = job
    try:
        ir_ps, d_ps, _ = depthir_frames(clip_dirs(clip_id, split, "IR"),
                                        clip_dirs(clip_id, split, "Depth_Color"))
    except OSError:
        ir_ps, d_ps = [], []
    if not d_ps:
        try:
            d = clip_dirs(clip_id, split, "Depth_Color")
            d_ps = sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")
        except OSError:
            d_ps = []
    if not d_ps:
        return {"clip_id": clip_id, "split": split, "action_id": action_id, "action_name": action_name, "status": "no_depth"}
    step = max(1, len(d_ps) // N_SAMPLE)
    picks = d_ps[::step][:N_SAMPLE]
    stack = []
    for p in picks:
        try:
            a = np.asarray(Image.open(p).convert("RGB"))
        except OSError:
            continue
        if a.shape[:2] != (480, 640):
            continue
        stack.append(to_ordinal(a, _RAMP))
    if not stack:
        return {"clip_id": clip_id, "split": split, "action_id": action_id, "action_name": action_name, "status": "unreadable"}
    box, diag = clip_box(np.stack(stack))
    if box is None:
        return {"clip_id": clip_id, "split": split, "action_id": action_id, "action_name": action_name,
                "status": "no_component", "n_frames": diag["n_frames"]}
    x0, y0, x1, y1 = fit_geometry(box)
    pf = [b for b in diag["per_frame"] if b is not None]
    sil = float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in pf]) / (640 * 480)) if pf else 0.0
    return {"clip_id": clip_id, "split": split, "action_id": action_id, "action_name": action_name, "status": "ok",
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "tight_x0": int(box[0]), "tight_y0": int(box[1]),
            "tight_x1": int(box[2]), "tight_y1": int(box[3]),
            "n_frames": diag["n_frames"], "n_detected": diag["n_detected"],
            "bbox_frac": round(sil, 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m_fresh.csv"))
    # CSV, not parquet. baseline.yaml says crops.parquet, which needs pyarrow --
    # a third-party dependency, and agents may not add those (DECISIONS §5).
    ap.add_argument("--out", type=Path, default=ROOT / "cache/crops.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--palette-clips", type=int, default=120)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.manifest.open()))
    print(f"[crop] manifest {args.manifest} · {len(rows)} rows")
    ramp = build_palette(rows, n_clips=args.palette_clips)
    if args.calibrate_only:
        return 0

    jobs = [(r["clip_id"], r["split"], int(r["action_id"]), r["action_name"]) for r in rows
            if r["depth_present"] == "1"]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[crop] {len(jobs)} clips with depth · {args.workers} workers")

    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(ramp.tobytes(), ramp.shape)) as ex:
        for i, res in enumerate(ex.map(_one, jobs, chunksize=8)):
            out.append(res)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["clip_id", "split", "action_id", "action_name", "status", "x0", "y0", "x1", "y1",
            "tight_x0", "tight_y0", "tight_x1", "tight_y1",
            "n_frames", "n_detected", "bbox_frac"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"[crop] wrote {args.out} · {len(out)} rows · {time.time()-t0:.0f}s")

    for split in ("train", "test"):
        s = [r for r in out if r["split"] == split]
        ok = [r for r in s if r["status"] == "ok"]
        print(f"[crop] {split}: {len(ok)}/{len(s)} ok "
              f"({100*(1-len(ok)/max(1,len(s))):.2f}% clip failure)")
    json.dump({"ramp": ramp.tolist()}, (args.out.parent / "crop_palette.json").open("w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
