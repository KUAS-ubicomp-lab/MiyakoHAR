"""Build the clip manifest — one row per clip, train and test.

This is the index everything downstream reads, so it is also where the corpus's
loader traps get handled once instead of in every consumer. Four of them, each
measured on the real data rather than anticipated:

1. BOM. CSVs are opened `utf-8-sig`. A BOM renames the first header to "﻿path"
   or "﻿时间" and every lookup by name then misses.

2. IMU columns are read POSITIONALLY, never by name. Training files carry a
   21-column Chinese header; 50 of 405 test clips carry a 23-column English one
   that appears nowhere in training. Columns 1-19 are semantically identical in
   both (time, device, accel xyz, gyro xyz, angle xyz, mag xyz, quat 0-3, temp);
   they diverge only at column 20, where Chinese has Version/Battery and English
   has Height/Pressure/Version/Battery. A name-based parse such as
   df['加速度X(g)'] raises KeyError on 12.3% of the test set on submission day.

3. IMU files are matched by CONTENT, never by filename. 61 training filenames
   and 14 test filenames are non-canonical -- CJK 上/下, doubled parens, stray
   spaces, and an ordering (上(LA+C+RA)) that never occurs in training. Worse,
   6 clips have the up and down file CONTENTS swapped. So this module never asks
   what a file is called: it reads the device-name column and assigns each row
   to a body site. Swapped files and exotic names both stop mattering.

4. Natural sort on frame filenames, so a change in zero-padding cannot silently
   reorder a clip.

And one definition that is not a trap but caused a month of wrong figures:
`{mod}_present` means CONTENT EXISTS, not that a file or directory exists. Radar
is the cautionary case -- every clip in Block II has a radar CSV, and every one
of them is empty.

Usage:
    python src/manifest.py --out manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

MODALITIES = ["Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton"]

# Short column prefixes, so the manifest header stays readable.
MOD_PREFIX = {
    "Depth_Color": "depth",
    "IR": "ir",
    "Thermal": "thermal",
    "IMU": "imu",
    "Radar": "radar",
    "Skeleton": "skel",
}

# Recovered by measurement M-06; documented nowhere in the corpus.
IMU_SITE_BY_DEVICE = {
    "WTLA": "LA",  # left arm
    "WTRA": "RA",  # right arm
    "WTC": "C",    # chest
    "WTLL": "LL",  # left leg
    "WTRL": "RL",  # right leg
}

# Blocks are a property of the recording setup, and radar presence is a PERFECT
# indicator of them (Block I 98.5% non-empty, Block II 0/1484). Recorded here
# for the leave-one-block-out DIAGNOSTIC only.
# Block, and anything derived from it -- including a has-radar or
# missing-modality flag -- must never become a model input. It is a subject-group
# label, so it gains under subject-mixed validation and cannot generalise.
BLOCK_I = set(range(1, 10))
BLOCK_II = set(range(16, 25))

# A loose grouping of users, NOT a date partition: users 1, 2 and 3 each
# recorded in both May and June.
CAMPAIGN_BY_USER = {
    **{u: "may_a" for u in (1, 2, 3, 4, 5)},
    **{u: "may_b" for u in (6, 7, 8, 9)},
    **{u: "june_a" for u in (16, 17, 18, 19, 20)},
    **{u: "june_b" for u in (21, 22, 23, 24)},
}

VAL_FOLD_BY_USER = {
    **{u: 0 for u in (4, 5, 9, 21, 22, 24)},
    **{u: 1 for u in (2, 6, 8, 17, 18, 19)},
    **{u: 2 for u in (1, 3, 7, 16, 20, 23)},
}

FRAME_SUFFIX = {"Depth_Color": ".png", "IR": ".png", "Thermal": ".jpg"}

_num_re = re.compile(r"(\d+)")


def natural_key(name: str):
    """Split digits out so frame_9 sorts before frame_10 under any padding."""
    return [int(p) if p.isdigit() else p for p in _num_re.split(name)]


def iter_files(d: Path):
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file() and not e.name.startswith("."):
                    yield e.name
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return


def count_frames(clip_dir: Path, modality: str) -> int:
    suffix = FRAME_SUFFIX[modality]
    return sum(1 for n in iter_files(clip_dir) if n.lower().endswith(suffix))


def count_skeleton(clip_dir: Path) -> int:
    """Skeleton ships as predictions/*.json, with a sibling visualizations/."""
    n = sum(1 for name in iter_files(clip_dir / "predictions") if name.endswith(".json"))
    if n == 0:  # tolerate a flat layout rather than assume the nested one
        n = sum(1 for name in iter_files(clip_dir) if name.endswith(".json"))
    return n


def read_rows(path: Path) -> list[list[str]]:
    """Read a CSV positionally. utf-8-sig strips a BOM if one is present."""
    try:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
            return [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    except OSError:
        return []


def scan_radar(clip_dir: Path) -> tuple[int, int]:
    """Return (n_files, n_data_rows). Empty-but-present is the whole point."""
    n_files = n_rows = 0
    for name in iter_files(clip_dir):
        if not name.lower().endswith(".csv"):
            continue
        n_files += 1
        rows = read_rows(clip_dir / name)
        n_rows += max(0, len(rows) - 1)  # minus header
    return n_files, n_rows


def scan_imu(clip_dir: Path) -> tuple[int, int, list[str], str]:
    """Return (n_files, n_data_rows, sorted sites present, schema tag).

    Filenames are ignored entirely. Sites come from the device-name column,
    which is column index 1 in both the Chinese and English schemas.
    """
    n_files = n_rows = 0
    sites: set[str] = set()
    widths: set[int] = set()
    schemas: set[str] = set()

    for name in iter_files(clip_dir):
        if not name.lower().endswith(".csv"):
            continue
        n_files += 1
        rows = read_rows(clip_dir / name)
        if not rows:
            continue

        header = rows[0]
        widths.add(len(header))
        joined = ",".join(header)
        if "DeviceName" in joined:
            schemas.add("en")
        elif "设备名称" in joined:
            schemas.add("cn")
        else:
            schemas.add("?")

        for row in rows[1:]:
            if len(row) < 19:  # the 19 aligned columns are the contract
                continue
            n_rows += 1
            device = row[1].strip()
            prefix = device.split("(", 1)[0].strip().upper()
            site = IMU_SITE_BY_DEVICE.get(prefix)
            if site:
                sites.add(site)

    schema = "|".join(sorted(schemas)) if schemas else ""
    if len(widths) > 1:
        schema += f" (widths {sorted(widths)})"
    return n_files, n_rows, sorted(sites), schema


def scan_clip(clip_dirs: dict[str, Path]) -> dict:
    """clip_dirs maps modality -> that clip's directory (absent keys allowed)."""
    out: dict[str, object] = {}
    for mod in MODALITIES:
        p = MOD_PREFIX[mod]
        d = clip_dirs.get(mod)
        if d is None:
            out[f"{p}_n"] = 0
            out[f"{p}_present"] = 0
            if mod == "IMU":
                out["imu_n_files"] = 0
                out["imu_sites"] = ""
                out["imu_schema"] = ""
            if mod == "Radar":
                out["radar_n_files"] = 0
            continue

        if mod in FRAME_SUFFIX:
            n = count_frames(d, mod)
        elif mod == "Skeleton":
            n = count_skeleton(d)
        elif mod == "Radar":
            n_files, n = scan_radar(d)
            out["radar_n_files"] = n_files
        elif mod == "IMU":
            n_files, n, sites, schema = scan_imu(d)
            out["imu_n_files"] = n_files
            out["imu_sites"] = "|".join(sites)
            out["imu_schema"] = schema
        else:  # unreachable, but never guess a count
            n = 0

        out[f"{p}_n"] = n
        # Content, not existence.
        out[f"{p}_present"] = int(n > 0)
    return out


def load_class_map(path: Path) -> dict[str, int]:
    mapping = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            mapping[row["action_name"]] = int(row["action_id"])
    return mapping


def scan_train(root: Path, class_map: dict[str, int]) -> list[dict]:
    """root is HAR/data, holding <modality>/<action>/<user>/<a-b-c>/."""
    # Union across modalities: a clip missing one stream is still a clip.
    clips: dict[tuple[str, str, str], dict[str, Path]] = {}
    for mod in MODALITIES:
        mod_root = root / mod
        if not mod_root.is_dir():
            print(f"[manifest] WARNING: no {mod_root}", file=sys.stderr)
            continue
        for action in sorted(os.listdir(mod_root)):
            action_dir = mod_root / action
            if not action_dir.is_dir() or action.startswith("."):
                continue
            for user in sorted(os.listdir(action_dir)):
                user_dir = action_dir / user
                if not user_dir.is_dir() or user.startswith("."):
                    continue
                for trial in sorted(os.listdir(user_dir), key=natural_key):
                    trial_dir = user_dir / trial
                    if not trial_dir.is_dir() or trial.startswith("."):
                        continue
                    clips.setdefault((action, user, trial), {})[mod] = trial_dir

    rows = []
    for (action, user, trial) in sorted(clips, key=lambda k: (natural_key(k[0]), natural_key(k[1]), natural_key(k[2]))):
        user_n = int(re.sub(r"\D", "", user) or -1)
        parts = trial.split("-")
        a, b, c = (parts + ["", "", ""])[:3]
        row = {
            "split": "train",
            "clip_id": f"{action}/{user}/{trial}",
            "path": "",
            "action_name": action,
            "action_id": class_map.get(action, -1),
            "user": user_n,
            "block": "I" if user_n in BLOCK_I else ("II" if user_n in BLOCK_II else "?"),
            "campaign": CAMPAIGN_BY_USER.get(user_n, "?"),
            "val_fold": VAL_FOLD_BY_USER.get(user_n, -1),
            "session_a": a,
            "session_b": b,
            "take": c,
        }
        row.update(scan_clip(clips[(action, user, trial)]))
        rows.append(row)
    return rows


def scan_test(root: Path) -> list[dict]:
    """root holds SM_test_NNNN/<modality>/."""
    clip_names = sorted(
        (n for n in os.listdir(root) if re.match(r"^SM_test_\d+$", n) and (root / n).is_dir()),
        key=natural_key,
    )
    rows = []
    for name in clip_names:
        clip_dirs = {m: root / name / m for m in MODALITIES if (root / name / m).is_dir()}
        row = {
            "split": "test",
            "clip_id": name,
            "path": f"{root.name}/{name}/",
            "action_name": "",
            "action_id": -1,
            "user": -1,
            "block": "",
            "campaign": "",
            "val_fold": -1,
            "session_a": "",
            "session_b": "",
            "take": "",
        }
        row.update(scan_clip(clip_dirs))
        rows.append(row)
    return rows


def main() -> int:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", type=Path, default=home / "cuhk-x/extracted/HAR/data")
    ap.add_argument("--test-root", type=Path, default=home / "cuhk-x/test_extracted/small_model_track_test")
    ap.add_argument("--class-map", type=Path, default=home / "cuhk-x/Small-Model-Track/class_mapping.csv")
    ap.add_argument("--out", type=Path, default=Path("manifest.csv"))
    args = ap.parse_args()

    class_map = load_class_map(args.class_map)
    print(f"[manifest] {len(class_map)} classes", file=sys.stderr)

    rows: list[dict] = []
    if args.train_root.is_dir():
        rows += scan_train(args.train_root, class_map)
        print(f"[manifest] train clips: {sum(r['split'] == 'train' for r in rows)}", file=sys.stderr)
    else:
        print(f"[manifest] WARNING: no train root at {args.train_root}", file=sys.stderr)

    if args.test_root.is_dir():
        rows += scan_test(args.test_root)
        print(f"[manifest] test clips:  {sum(r['split'] == 'test' for r in rows)}", file=sys.stderr)
    else:
        print(f"[manifest] WARNING: no test root at {args.test_root}", file=sys.stderr)

    if not rows:
        raise SystemExit("no clips found")

    fields = list(rows[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"[manifest] wrote {args.out} ({len(rows)} rows, {len(fields)} columns)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
