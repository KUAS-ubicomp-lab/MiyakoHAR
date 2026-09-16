"""Verify the memmap cache against the corpus it came from.

WHY THIS IS NOT OPTIONAL. A truncated memmap does not raise. numpy maps whatever
length the file has and returns zeros past the end, so a cache that lost 3 GB in
transfer trains silently on black frames and reports a real-looking number for it.
The same is true of an off-by-one in the offset table: every clip reads its
neighbour's frames, accuracy drops a few points, and it looks like a bad
hyperparameter. Neither failure announces itself.

So this re-derives the cache's own claims from the corpus rather than trusting
cache_manifest.json, and the ordering check deliberately extracts the global frame
index with a DIFFERENT expression from the one preprocess.py uses -- a test that
imports the code it is testing only proves the code is self-consistent.

The project's standing rule applies to any failure here: diagnose, never adjust
the expected value. Four of check_manifest.py's 38 checks failed on first run;
two were denominator differences and two were real findings that editing the
expectations would have destroyed.

Usage:
    python tests/check_cache.py --cache cache/ --manifest /tmp/m.csv
    python tests/check_cache.py --cache cache/ --manifest /tmp/m.csv --quick   # skip sha256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN_ROOT = Path.home() / "cuhk-x/extracted/HAR/data"
TEST_ROOT = Path.home() / "cuhk-x/test_extracted/small_model_track_test"

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
    return cond


# ── independent re-derivation of frame order ────────────────────────────────
# NOT preprocess.py's regexes. The last digit-group in the name is the global
# frame index in every one of these schemas, so this agrees for a different reason.
def _idx(name: str) -> int:
    groups = re.findall(r"\d+", name)
    return int(groups[-1]) if groups else -1


def corpus_frames(clip_id: str, split: str, stream: str) -> list[tuple[Path, ...]]:
    def d(mod: str) -> Path:
        if split == "train":
            a, u, t = clip_id.split("/")
            return TRAIN_ROOT / mod / a / u / t
        return TEST_ROOT / clip_id / mod

    def names(p: Path, suffix: str) -> list[str]:
        try:
            return sorted((n for n in os.listdir(p) if n.lower().endswith(suffix)), key=_idx)
        except OSError:
            return []

    if stream == "thermal":
        p = d("Thermal")
        return [(p / n,) for n in names(p, ".jpg")]

    ip, dp = d("IR"), d("Depth_Color")
    # Depth carries a _Color suffix, so its last digit group is still the index.
    ir_n, depth_n = names(ip, ".png"), names(dp, ".png")
    by_idx_ir = {_idx(n): n for n in ir_n}
    by_idx_dp = {_idx(n): n for n in depth_n}
    shared = sorted(set(by_idx_ir) & set(by_idx_dp))
    return [(ip / by_idx_ir[i], dp / by_idx_dp[i]) for i in shared]


def decode_expected(
    tup: tuple[Path, ...], stream: str, W: int, H: int, box: tuple[int, ...] | None = None
) -> np.ndarray | None:
    """Re-decode one frame from the corpus, independently of preprocess.py.

    `box` is the per-clip crop in ORIGINAL pixels, read back out of
    cache_manifest.json. This is the check that a cropped shard was cropped with
    the box it CLAIMS -- an off-by-one, a transposed (x, y), or a box applied to
    the wrong clip all survive every other check in this file, and would show up
    only as an A/B result nobody could interpret.
    """
    def one(p: Path, mode: str):
        try:
            with Image.open(p) as im:
                im = im.convert(mode)
                if box is not None:
                    im = im.crop(tuple(box))
                if im.size != (W, H):
                    im = im.resize((W, H), Image.BOX)
                return np.asarray(im, dtype=np.uint8)
        except Exception:
            return None

    if stream == "thermal":
        return one(tup[0], "RGB")
    ir, dp = one(tup[0], "L"), one(tup[1], "RGB")
    if dp is None:
        return None
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[:, :, 0:3] = dp
    if ir is not None:
        out[:, :, 3] = ir
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m.csv"))
    ap.add_argument("--spot-clips", type=int, default=40)
    ap.add_argument("--quick", action="store_true", help="skip sha256 (it reads every byte)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    random.seed(args.seed)

    cm_path = args.cache / "cache_manifest.json"
    if not cm_path.exists():
        print(f"no {cm_path}", file=sys.stderr)
        return 2
    cm = json.loads(cm_path.read_text())

    with args.manifest.open(newline="", encoding="utf-8-sig") as fh:
        rows = {(r["split"], r["clip_id"]): r for r in csv.DictReader(fh)}

    # Scoped PER SHARD, not globally: a clip excluded from depth_ir because it has
    # no Depth is still a perfectly good thermal clip. Checking a flattened set
    # against every shard was this file's own bug on first run.
    excluded_by_shard = cm.get("excluded_clips", {})

    for s in cm["shards"]:
        stream, split = s["stream"], s["split"]
        total, H, W, C = s["shape"]
        path = args.cache / s["shard"]
        print(f"\n{s['shard']}  ({stream}/{split})")

        # ── integrity ───────────────────────────────────────────────────────
        if not check_true(f"{s['shard']} exists", path.exists()):
            continue
        check(f"{s['shard']} byte size", path.stat().st_size, s["bytes"])
        check(f"{s['shard']} bytes = N*H*W*C", total * H * W * C, s["bytes"])
        if not args.quick:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                while block := fh.read(1 << 24):
                    h.update(block)
            check_true(f"{s['shard']} sha256 matches", h.hexdigest() == s["sha256"])

        # ── the offset table ────────────────────────────────────────────────
        offs = np.asarray(s["offsets"], dtype=np.int64)
        clips = s["clip_ids"]
        check(f"{s['shard']} offsets length = n_clips+1", len(offs), len(clips) + 1)
        check_true(f"{s['shard']} offsets non-decreasing", bool(np.all(np.diff(offs) >= 0)))
        check(f"{s['shard']} last offset = frame total", int(offs[-1]), total)
        check(f"{s['shard']} clip_ids unique", len(set(clips)), len(clips))
        excluded = set(excluded_by_shard.get(f"{stream}_{split}", []))
        check_true(
            f"{s['shard']} carries none of ITS OWN excluded clips",
            not (set(clips) & excluded),
            f"overlap {sorted(set(clips) & excluded)[:3]}",
        )
        check_true(
            f"{s['shard']} listing dropped no filename",
            not s.get("clips_with_unparsed_filenames"),
            f"{list(s.get('clips_with_unparsed_filenames', {}).items())[:2]}",
        )
        check_true(
            f"{s['shard']} no unpaired frame index",
            not s.get("clips_with_unpaired_index"),
            f"{list(s.get('clips_with_unpaired_index', {}).items())[:2]}",
        )

        # ── the crop box table, if this shard is a cropped one ──────────────
        # Structural only. The PIXELS are checked at the bottom of the loop, by
        # re-cropping from the corpus with the box recorded here.
        if "crop_boxes" in s:
            boxes = np.asarray(s["crop_boxes"], dtype=np.int64)
            sw, sh = s["crop_source_wh"]
            check(f"{s['shard']} one crop box per clip", len(boxes), len(clips))
            check_true(f"{s['shard']} every crop box lies inside {sw}x{sh}",
                       bool((boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
                            and (boxes[:, 2] <= sw).all() and (boxes[:, 3] <= sh).all()))
            bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            check_true(f"{s['shard']} no degenerate crop box", bool((bw > 0).all() and (bh > 0).all()))
            # 1:2 is the settled geometry (DECISIONS §8). fit_geometry clamps to
            # the frame AFTER holding the aspect, so a box pinned to the full
            # height keeps a little extra width -- that is the only way to miss.
            asp = bw / bh
            if s.get("crop_from") == "tight":
                # A3 (QUEUE §33): the square stream — 1:1 up to frame clamping (a box pinned to the
                # full height keeps its width, so the only way to miss 1.0 is the clamp).
                check_true(f"{s['shard']} crop aspect is 1:1 up to frame clamping",
                           bool((asp >= 0.95).all() and (asp <= 1.05).all()),
                           f"min {asp.min():.3f} max {asp.max():.3f}")
            else:
                check_true(f"{s['shard']} crop aspect is 1:2 up to frame clamping",
                           bool((asp >= 0.48).all() and (asp <= (sw / sh) + 1e-9).all()),
                           f"min {asp.min():.3f} max {asp.max():.3f}")
            # The fallback is a POLICY, so it must be the declared box on
            # exactly the declared clips -- not a sentinel that leaked wider.
            fb = list(s["crop_fallback_box"])
            got_fb = {clips[i] for i in range(len(clips)) if list(boxes[i]) == fb}
            declared = set(s["crop_fallback_clips"])
            check_true(f"{s['shard']} every declared fallback clip carries the fallback box",
                       declared <= got_fb, f"{len(declared - got_fb)} declared clips carry another box")
            # RE-DERIVED 2026-08-26 (A3, m175): a fitted box CAN equal the fallback by construction —
            # the geometry is clamped to the frame, so a tall, centred subject's fitted square (or 1:2
            # box) lands exactly on the centred fallback. "Exactly the declared clips" was therefore the
            # wrong expectation; the property that matters is that every UNDECLARED clip carrying the
            # fallback box is a genuine coincidence: crops.csv has it `ok` and refitting ITS tight/fitted
            # box reproduces the fallback. Anything else is a leaked sentinel and still fails.
            extra = sorted(got_fb - declared)
            leaked = []
            if extra:
                import csv as _csv
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
                from preprocess import fit_square  # noqa: PLC0415
                crops = {(r["split"], r["clip_id"]): r
                         for r in _csv.DictReader(open(Path(__file__).resolve().parent.parent / "cache/crops.csv",
                                                       newline="", encoding="utf-8-sig"))}
                for c in extra:
                    r = crops.get((split, c))
                    if r is None or r["status"] != "ok":
                        leaked.append(c); continue
                    if s.get("crop_from") == "tight":
                        refit = list(fit_square((int(r["tight_x0"]), int(r["tight_y0"]), int(r["tight_x1"]), int(r["tight_y1"]))))
                    else:
                        refit = [int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])]
                    if refit != fb:
                        leaked.append(c)
            check_true(f"{s['shard']} no undeclared clip carries the fallback box except verified coincidences",
                       not leaked, f"leaked: {leaked[:5]}" if leaked else f"{len(extra)} coincidence(s) verified against crops.csv")
            print(f"    crop: {len(clips)-len(s['crop_fallback_clips'])}/{len(clips)} detected, "
                  f"{len(s['crop_fallback_clips'])} on centre fallback "
                  f"({100*len(s['crop_fallback_clips'])/len(clips):.2f}%)")

        # ── counts re-derived from the manifest, not from the cache ─────────
        col = "thermal_n" if stream == "thermal" else "ir_n"
        mism = [
            (c, int(offs[i + 1] - offs[i]), int(rows[(split, c)][col]))
            for i, c in enumerate(clips)
            if int(offs[i + 1] - offs[i]) != int(rows[(split, c)][col])
        ]
        check_true(
            f"{s['shard']} every clip length matches manifest {col}",
            not mism,
            f"{len(mism)} differ, e.g. {mism[:2]}",
        )
        expect_total = sum(int(rows[(split, c)][col]) for c in clips)
        check(f"{s['shard']} frame total matches manifest sum", total, expect_total)
        check_true(f"{s['shard']} no zero-length clip stored", min(np.diff(offs), default=1) > 0)

        mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(total, H, W, C))

        # ── absent channels are absent, and only where declared ─────────────
        absent = s.get("channel_absent", {})
        for name, ids in absent.items():
            sl = slice(0, 3) if name in ("rgb", "depth") else slice(3, 4)
            other = slice(3, 4) if name in ("rgb", "depth") else slice(0, 3)
            for cid in ids:
                i = clips.index(cid)
                blk = mm[offs[i] : offs[i + 1]]
                check_true(f"{cid}: '{name}' plane is zero as declared", not blk[:, :, :, sl].any())
                if C > 1 and name == "ir":
                    check_true(
                        f"{cid}: depth planes survived — the clip is NOT discarded",
                        bool(blk[:, :, :, other].any()),
                    )

        flagged = {c for v in absent.values() for c in v}
        # ── nothing silently blank ──────────────────────────────────────────
        blank = []
        for i in random.sample(range(len(clips)), min(200, len(clips))):
            if clips[i] in flagged:
                continue
            if not mm[offs[i] : offs[i + 1]].any():
                blank.append(clips[i])
        check_true(
            f"{s['shard']} no unflagged all-zero clip in a 200-clip sample",
            not blank,
            f"{blank[:3]}",
        )

        # ── one-frame clips survive, since 42 clips hit this on day one ─────
        ones = [c for i, c in enumerate(clips) if offs[i + 1] - offs[i] == 1]
        check(f"{s['shard']} one-frame clip count", len(ones), s["n_clips_one_frame"])
        if ones:
            bad = [c for c in ones if not mm[offs[clips.index(c)]].any() and c not in flagged]
            check_true(f"{s['shard']} every one-frame clip holds real pixels", not bad, f"{bad[:3]}")

        # ── ground truth: re-decode from the corpus and compare bytes ─────
        # This is the check that catches wrong order, wrong resample, wrong
        # channel order, and an off-by-one offset. Everything above would pass
        # with the frames in reverse.
        sample = random.sample(range(len(clips)), min(args.spot_clips, len(clips)))
        mismatch, compared = [], 0
        for i in sample:
            cid = clips[i]
            n = int(offs[i + 1] - offs[i])
            paths = corpus_frames(cid, split, stream)
            if len(paths) != n:
                mismatch.append((cid, f"listed {len(paths)} vs stored {n}"))
                continue
            box = s["crop_boxes"][i] if "crop_boxes" in s else None
            for j in {0, n // 2, n - 1}:
                exp = decode_expected(paths[j], stream, W, H, box)
                if exp is None:
                    continue
                got = np.asarray(mm[offs[i] + j])
                compared += 1
                if not np.array_equal(got, exp):
                    d = int(np.abs(got.astype(int) - exp.astype(int)).max())
                    mismatch.append((cid, f"frame {j} differs, max|Δ|={d}"))
                    break
        check_true(
            f"{s['shard']} ★ {compared} frames byte-match an independent re-decode",
            not mismatch,
            f"{len(mismatch)} mismatched, e.g. {mismatch[:2]}",
        )
        del mm

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
