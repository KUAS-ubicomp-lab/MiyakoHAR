"""Cross-check the manifest against measurements taken independently of it.

The manifest is the index everything downstream reads, so a silent error in it
propagates everywhere. Rather than trust it, this script re-derives figures that
were measured earlier by separate one-off scripts and compares.

Agreement means two independent paths reached the same number. Disagreement is a
finding either way: either the manifest is wrong, or the earlier measurement was.
Neither outcome is a reason to adjust a number until the cause is known.

Usage:  python tests/check_manifest.py manifest.csv
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PASS, FAIL = [], []


def check(name: str, got, want, note: str = "") -> None:
    ok = got == want
    (PASS if ok else FAIL).append(name)
    mark = "ok  " if ok else "FAIL"
    detail = f"got {got!r}, expected {want!r}"
    print(f"  [{mark}] {name}: {detail}{('  -- ' + note) if note else ''}")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "manifest.csv")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    i = lambda r, k: int(r[k])  # noqa: E731

    print(f"\n{path}: {len(rows)} rows\n")

    print("clip counts")
    check("train clips", len(train), 3036)
    check("test clips", len(test), 405)

    print("\nradar — the block signal (finding 6b)")
    b1 = [r for r in train if r["block"] == "I"]
    b2 = [r for r in train if r["block"] == "II"]
    check("Block I radar files", sum(i(r, "radar_n_files") for r in b1), 1430)
    check("Block I radar non-empty", sum(i(r, "radar_present") for r in b1), 1409)
    check("Block II radar files", sum(i(r, "radar_n_files") for r in b2), 1484)
    check("Block II radar non-empty", sum(i(r, "radar_present") for r in b2), 0,
          "the sensor was off for every user 16-24")
    check("test radar non-empty", sum(i(r, "radar_present") for r in test), 198)
    check("test clips with no radar file", sum(1 for r in test if i(r, "radar_n_files") == 0), 1)

    print("\nclass imbalance (finding 3)")
    per_class = Counter(r["action_name"] for r in train)
    check("Walk clips", per_class["36_Walk"], 365)
    check("Watch_TV clips", per_class["25_Watch_TV"], 12)
    # The record says "median 59". The true median of 40 values is the mean of
    # the 20th and 21st, which are 57 and 59 -> 58.0. Assert the pair instead,
    # which is unambiguous under either convention.
    ranked = sorted(per_class.values())
    check("clips/class, 20th and 21st ranked", (ranked[19], ranked[20]), (57, 59),
          "median 58.0; the recorded 59 is median_high")
    check("imbalance ratio (1dp)", round(365 / 12, 1), 30.4)

    print("\nfactorial structure (finding 2)")
    pairs = {(r["action_name"], r["user"]) for r in train}
    check("(action,user) pairs present", len(pairs), 537, "of a possible 40x18=720")
    by_user = defaultdict(set)
    by_action = defaultdict(set)
    for a, u in pairs:
        by_user[u].add(a)
        by_action[a].add(u)
    check("actions per user, min-max",
          (min(len(v) for v in by_user.values()), max(len(v) for v in by_user.values())), (23, 35))
    check("users per action, min-max",
          (min(len(v) for v in by_action.values()), max(len(v) for v in by_action.values())), (3, 18))

    print("\nframes per clip (finding 8)")
    # Finding 8's figures are over clips THAT HAVE the modality. The manifest
    # rows are all clips, so a clip missing a stream contributes 0. Both are
    # right; only the denominator differs. Asserted present-only, to compare
    # like with like.
    ir = [i(r, "ir_n") for r in train if i(r, "ir_n") > 0]
    th = [i(r, "thermal_n") for r in train if i(r, "thermal_n") > 0]
    check("train IR min/median/max, present only",
          (min(ir), int(statistics.median(ir)), max(ir)), (1, 24, 236))
    check("train thermal min/median/max, present only",
          (min(th), int(statistics.median(th)), max(th)), (1, 55, 595))
    check("train clips with exactly 1 thermal frame", sum(1 for n in th if n == 1), 40,
          "the sampler must repeat-pad from n=1")
    check("train clips with exactly 1 IR frame", sum(1 for n in ir if n == 1), 2)

    print("\ntrain modality coverage is RAGGED (measured here first)")
    # Not in the prior record, which characterised test completeness only.
    # A clip with zero IR frames cannot feed the depth+IR branch at all.
    for key, label, want in (("ir_n", "IR", 103), ("depth_n", "Depth", 105),
                             ("skel_n", "Skeleton", 105), ("thermal_n", "Thermal", 145)):
        check(f"train clips with no {label}", sum(1 for r in train if i(r, key) == 0), want)
    check("train clips carrying IR", sum(1 for r in train if i(r, "ir_n") > 0), 2933,
          "matches the 2,933-clip denominator of the duplicate-frame finding")
    check("train clips carrying a radar file",
          sum(1 for r in train if i(r, "radar_n_files") > 0), 2914,
          "matches the 2,914 radar CSVs parsed for D2")

    print("\ntest structure (findings 18, 20)")
    tir = [i(r, "ir_n") for r in test]
    check("test IR median", int(statistics.median(tir)), 20, "shorter than train's 24")
    mods = ["depth", "ir", "thermal", "imu", "radar", "skel"]
    # The recorded "394/405 have all six" is a FILE-EXISTENCE figure. By
    # content it is 191, because only 198 test clips have a non-empty radar CSV
    # -- the rest ship the file with no detections. Both are asserted so the
    # distinction cannot quietly collapse again.
    by_content = sum(1 for r in test if all(i(r, f"{m}_present") for m in mods))
    by_file = sum(1 for r in test
                  if all((i(r, f"{m}_n") > 0 if m != "radar" else i(r, "radar_n_files") > 0)
                         for m in mods))
    check("test clips with all six by CONTENT", by_content, 191)
    check("test clips with all six by FILE EXISTENCE", by_file, 394,
          "this is the figure the record quotes")
    no_thermal = sorted(r["clip_id"] for r in test if i(r, "thermal_n") == 0)
    check("test clips with no thermal", len(no_thermal), 10)
    check("named no-thermal clips", no_thermal,
          ["SM_test_0013", "SM_test_0072", "SM_test_0104", "SM_test_0207", "SM_test_0271",
           "SM_test_0286", "SM_test_0297", "SM_test_0310", "SM_test_0354", "SM_test_0403"])

    print("\nIMU schema trap (finding 12)")
    en = [r for r in test if "en" in r["imu_schema"]]
    check("test clips with the English 23-column header", len(en), 50,
          "this schema appears zero times in training")
    check("train clips with the English header", sum(1 for r in train if "en" in r["imu_schema"]), 0)
    check("all 5 IMU sites recovered somewhere",
          sorted({s for r in rows for s in r["imu_sites"].split("|") if s}),
          ["C", "LA", "LL", "RA", "RL"])

    print("\nfrozen folds (D5, finding 4 / M-11)")
    for k in range(3):
        val = [r for r in train if i(r, "val_fold") == k]
        trn = [r for r in train if i(r, "val_fold") != k]
        classes_in_train = {r["action_name"] for r in trn}
        check(f"fold {k} train covers all 40 classes", len(classes_in_train), 40)
        if k == 1:
            check("fold 1 validates zero clips of class 25",
                  sum(1 for r in val if r["action_name"] == "25_Watch_TV"), 0,
                  "known unmeasurable cell")
    check("every train clip assigned a fold", sum(1 for r in train if i(r, "val_fold") < 0), 0)

    print(f"\n{'=' * 62}")
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  failures:")
        for f in FAIL:
            print(f"    - {f}")
        print("\n  A mismatch is a finding, not a number to adjust. Establish")
        print("  which of the two measurements is wrong before changing either.")
    print(f"{'=' * 62}\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
