"""Verify dataset.py against the cache and the frozen folds.

The failure modes here are all SILENT. A subject leaking from train into val
inflates every number we will ever report and looks like a good result. A
per-frame augmentation injects motion that is not in the action and looks like a
hard dataset. An off-by-one in the offset lookup feeds a neighbour's frames and
looks like a bad hyperparameter. None of them raise.

So, as in check_cache.py, expectations are re-derived INDEPENDENTLY of the code
under test wherever that is possible: labels from the clip_id string rather than
the manifest column, fold membership from splits/folds.yaml rather than the
manifest's val_fold, and pixels from a direct memmap read rather than the
Dataset's own indexing.

Standing rule: diagnose any failure. Never adjust the expected value.

Usage:
    python tests/check_dataset.py --cache cache/ --manifest /tmp/m.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import ClipDataset, _tsn_indices, collate  # noqa: E402

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
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m.csv"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    folds_file = root / "splits" / "folds.yaml"
    spec = yaml.safe_load(folds_file.read_text(encoding="utf-8"))
    cm = json.loads((args.cache / "cache_manifest.json").read_text())
    with args.manifest.open(newline="", encoding="utf-8-sig") as fh:
        mrows = list(csv.DictReader(fh))

    # ── 1. TSN sampling, in isolation ───────────────────────────────────────
    print("\nTSN segmental sampling")
    for n in (1, 2, 3, 7, 8, 9, 24, 201):
        ev = _tsn_indices(n, 8, train=False, rng=None)
        check_true(f"n={n}: eval indices in range", bool(ev.min() >= 0 and ev.max() < n), f"{ev.tolist()}")
        check_true(f"n={n}: eval indices non-decreasing", bool(np.all(np.diff(ev) >= 0)))
        check(f"n={n}: eval returns exactly T", len(ev), 8)
        ev2 = _tsn_indices(n, 8, train=False, rng=None)
        check_true(f"n={n}: eval is deterministic", bool(np.array_equal(ev, ev2)))
        rng = np.random.default_rng(0)
        tr = _tsn_indices(n, 8, train=True, rng=rng)
        check_true(f"n={n}: train indices in range", bool(tr.min() >= 0 and tr.max() < n))
    # n=1 is the one that used to be an n=1 edge case rather than a policy.
    check_true("n=1 repeat-pads to T copies of frame 0", bool(np.array_equal(_tsn_indices(1, 8, False, None), np.zeros(8, dtype=np.int64))))
    # A long clip must actually SPREAD, or we are training on the first second.
    spread = _tsn_indices(201, 8, train=False, rng=None)
    check_true("n=201 spans the clip, not just its head", bool(spread[-1] > 0.8 * 201), f"last index {spread[-1]}")

    # ── 2. The frozen folds, and the leak that would not raise ──────────────
    print("\nFold integrity (the failure that inflates every future number)")
    for fold in (0, 1, 2):
        tr_ds = ClipDataset(args.cache, args.manifest, "thermal", "train", "train", fold=fold, folds_file=folds_file)
        va_ds = ClipDataset(args.cache, args.manifest, "thermal", "train", "val", fold=fold, folds_file=folds_file)
        tr_u = {it.user for it in tr_ds.items}
        va_u = {it.user for it in va_ds.items}
        entry = next(f for f in spec["folds"] if f["fold"] == fold)
        check(f"fold {fold}: train subjects match folds.yaml", sorted(tr_u), sorted(entry["train"]))
        check(f"fold {fold}: val subjects match folds.yaml", sorted(va_u), sorted(entry["val"]))
        check_true(f"fold {fold}: NO subject in both sides", not (tr_u & va_u), f"leak {sorted(tr_u & va_u)}")
        tr_c = {it.clip_id for it in tr_ds.items}
        va_c = {it.clip_id for it in va_ds.items}
        check_true(f"fold {fold}: no clip in both sides", not (tr_c & va_c))
        # Nothing may vanish between the two sides.
        shard = next(s for s in cm["shards"] if s["stream"] == "thermal" and s["split"] == "train")
        check(f"fold {fold}: train+val covers the whole shard", len(tr_c | va_c), len(shard["clip_ids"]))

    # ── 3. Labels, re-derived from the clip_id string ───────────────────────
    print("\nLabels (re-derived from the path, not read from the manifest column)")
    ds = ClipDataset(args.cache, args.manifest, "thermal", "train", "train", fold=0, folds_file=folds_file)
    bad = [(it.clip_id, it.label) for it in ds.items if int(it.clip_id.split("_", 1)[0]) != it.label]
    check_true("every label equals its clip_id integer prefix", not bad, f"{bad[:3]}")
    check_true("labels span 0..39", set(it.label for it in ds.items) <= set(range(40)))
    baduser = [it.clip_id for it in ds.items if f"user{it.user}/" not in it.clip_id]
    check_true("every user id matches the clip_id path", not baduser, f"{baduser[:3]}")

    # ── 4. Absent channel groups ────────────────────────────────────────────
    print("\nchannel_absent handling (trap 3)")
    di_test = ClipDataset(args.cache, args.manifest, "depthir", "test", "test")
    th_test = ClipDataset(args.cache, args.manifest, "thermal", "test", "test")
    shard_di = next(s for s in cm["shards"] if s["stream"] == "depthir_raw" and s["split"] == "test")
    ir_absent = set(shard_di.get("channel_absent", {}).get("ir", []))
    ids_di = {it.clip_id for it in di_test.items}
    ids_th = {it.clip_id for it in th_test.items}
    # Depth survives on those clips, so depthir keeps them -- flagged, not dropped.
    check_true("IR-absent clips are KEPT (their Depth is intact)", ir_absent <= ids_di)
    flagged = {it.clip_id for it in di_test.items if not it.branch_present}
    check("IR-absent clips are flagged present=False", sorted(flagged), sorted(ir_absent))
    check("no depthir clip is dropped for an absent required group", di_test.n_skipped_unusable, 0)

    # The guarantee that makes skip_branch safe: if a clip could feed NEITHER
    # branch, fusion has nothing to renormalise over and predict.py falls back
    # to a constant — a silent accuracy hole on the leaderboard.
    #
    # Two earlier versions of this check were TAUTOLOGIES. `(A|B) - (A|B)` is
    # empty whatever the data, and iterating the union looking for members of
    # neither set can never find one. Both "passed" on data that had not been
    # consulted. The denominator must come from the MANIFEST, not from the sets
    # under test.
    all_test = {r["clip_id"] for r in mrows if r["split"] == "test"}
    check("manifest lists 405 test clips", len(all_test), 405)
    covered = ids_di | ids_th
    missing = sorted(all_test - covered)
    check_true("EVERY manifest test clip can feed at least one branch", not missing, f"{len(missing)} orphaned: {missing[:5]}")
    check_true("neither branch invents a clip the manifest lacks", not (covered - all_test), f"{sorted(covered - all_test)[:3]}")
    # Same property on the training side, where an orphan would be a silent
    # shrink of the training set rather than a leaderboard hole.
    all_train = {r["clip_id"] for r in mrows if r["split"] == "train"}
    tr_cov = set()
    for br, st in (("thermal", "thermal"), ("depthir", "depthir_raw")):
        s = next(x for x in cm["shards"] if x["stream"] == st and x["split"] == "train")
        tr_cov |= set(s["clip_ids"])
    check("manifest lists 3036 train clips", len(all_train), 3036)
    check_true("EVERY manifest train clip is in at least one shard", not (all_train - tr_cov), f"{sorted(all_train - tr_cov)[:5]}")

    # ── 5. Pixels: independent memmap read ──────────────────────────────────
    print("\nPixels byte-match an independent memmap read")
    total, H, W, C = shard_di["shape"]
    mm = np.memmap(args.cache / shard_di["shard"], dtype=np.uint8, mode="r", shape=(total, H, W, C))
    mismatch = []
    for i in (0, 1, 7, 40, 199, len(di_test) - 1):
        it = di_test.items[i]
        n = it.stop - it.start
        idx = _tsn_indices(n, 8, train=False, rng=None)
        want = np.asarray(mm[it.start + idx])  # (T,H,W,C)
        got = di_test[i]["x"]
        # Undo normalisation + dead strip to recover the stored bytes.
        from src.dataset import NORM

        mean = torch.tensor(NORM["depthir"]["mean"]).view(1, -1, 1, 1)
        std = torch.tensor(NORM["depthir"]["std"]).view(1, -1, 1, 1)
        rec = (got * std + mean).mul(255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        d = di_test.dead_strip_cols
        if not np.array_equal(rec[:, :, d:, :], want[:, :, d:, :]):
            mismatch.append((it.clip_id, int(np.abs(rec[:, :, d:, :].astype(int) - want[:, :, d:, :].astype(int)).max())))
    check_true(f"6 clips round-trip to the stored bytes", not mismatch, f"{mismatch[:2]}")
    del mm

    # ── 6. The dead strip, in the right units (trap 4) ──────────────────────
    print("\nDead strip")
    check("depthir dead strip is 40 orig px rescaled to cache width", di_test.dead_strip_cols, 10)
    check("thermal has no dead strip", th_test.dead_strip_cols, 0)
    check_true("dead strip is NOT the config's literal 40", di_test.dead_strip_cols != 40)
    s = di_test[0]["x"]
    from src.dataset import NORM as _N

    zero_lvl = -torch.tensor(_N["depthir"]["mean"][:3]) / torch.tensor(_N["depthir"]["std"][:3])
    strip = s[:, 0:3, :, : di_test.dead_strip_cols]
    check_true("depth channels are masked inside the strip", bool(torch.allclose(strip, zero_lvl.view(1, 3, 1, 1).expand_as(strip), atol=1e-5)))
    check_true("IR is NOT masked inside the strip", bool(s[:, 3:4, :, : di_test.dead_strip_cols].std() > 0))

    # ── 7. Clip-consistent augmentation (trap 6) ────────────────────────────
    print("\nAugmentation is drawn ONCE per clip")
    one = [i for i, it in enumerate(ds.items) if it.stop - it.start == 1]
    check_true("there are one-frame clips to test with", bool(one), f"{len(one)} found")
    if one:
        # A one-frame clip repeat-pads to T IDENTICAL frames. If augmentation
        # were per-frame they would diverge; clip-consistent keeps them equal.
        # This is the only way to observe the property from the outside.
        x = ds[one[0]]["x"]
        check_true("one-frame clip stays identical across T after augment", bool(torch.allclose(x, x[0:1].expand_as(x), atol=1e-6)))
    aug_off = ClipDataset(args.cache, args.manifest, "thermal", "train", "val", fold=0, folds_file=folds_file)
    a, b = aug_off[3]["x"], aug_off[3]["x"]
    check_true("val mode is deterministic across two reads", bool(torch.equal(a, b)))
    check_true("val mode does not augment", aug_off.augment is False)
    check_true("train mode does augment", ds.augment is True)

    # THE ONE THAT CAUGHT A REAL BUG. Seeding the per-clip RNG from
    # (seed, row, torch.initial_seed()) is constant within a process, so every
    # epoch re-drew the SAME augmentation and the effective training set was
    # frozen at one view per clip. Nothing raised; the only symptom would have
    # been a later A/B concluding augmentation does not help.
    # Verified red-green: this check FAILS on the constant-seed version.
    r1, r2 = ds[5]["x"], ds[5]["x"]
    check_true("train augmentation VARIES between two reads of the same clip", not torch.equal(r1, r2))
    raw = ClipDataset(args.cache, args.manifest, "thermal", "train", "train", fold=0, folds_file=folds_file, augment=False)
    check_true("augment=False is deterministic", bool(torch.equal(raw[5]["x"], raw[5]["x"])))
    check_true("augmentation actually changes the pixels", not torch.equal(ds[5]["x"], raw[5]["x"]))
    # ...while still being drawn once per clip, not once per frame.
    if one:
        x2 = ds[one[0]]["x"]
        check_true("still clip-consistent after the seeding fix", bool(torch.allclose(x2, x2[0:1].expand_as(x2), atol=1e-6)))

    # ── 8. Shapes, dtypes, and the collate contract ─────────────────────────
    print("\nShapes and collation")
    b0 = ds[0]
    check("thermal item shape", tuple(b0["x"].shape), (8, 3, 120, 160))
    check("depthir item shape", tuple(di_test[0]["x"].shape), (8, 4, 120, 160))
    check("dtype is float32", b0["x"].dtype, torch.float32)
    check_true("normalised values are finite", bool(torch.isfinite(b0["x"]).all()))
    batch = collate([ds[i] for i in range(4)])
    check("collated batch shape", tuple(batch["x"].shape), (4, 8, 3, 120, 160))
    check("collated labels shape", tuple(batch["y"].shape), (4,))
    check_true("collated labels are valid classes", bool((batch["y"] >= 0).all() and (batch["y"] < 40).all()))

    # ── 9. Mode/split coherence is refused, not silently accepted ───────────
    print("\nGuards")
    for kw, why in (
        (dict(branch="thermal", split="test", mode="val"), "test split with val mode"),
        (dict(branch="thermal", split="train", mode="test"), "train split with test mode"),
        (dict(branch="nope", split="train", mode="train"), "unknown branch"),
    ):
        try:
            ClipDataset(args.cache, args.manifest, folds_file=folds_file, **kw)
            check_true(f"refuses {why}", False, "constructed anyway")
        except ValueError:
            check_true(f"refuses {why}", True)

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
