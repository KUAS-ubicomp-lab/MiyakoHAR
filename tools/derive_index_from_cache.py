"""Derive the four manifest columns dataset.py reads, from the cache alone.

WHY THIS EXISTS: ALIEN holds a sha256-verified cache but its corpus is still
downloading, so `src/manifest.py` cannot run -- it walks HAR/data. dataset.py
reads exactly four fields (clip_id, split, user, action_id), and for a train
clip all three derived ones are recoverable from the clip_id, because
manifest.py BUILT that id as f"{action}/{user}/{trial}" from those same fields.
So this is a string split of a join, not an inference.

THIS IS NOT THE MANIFEST. It carries none of manifest.py's measured columns
(modality presence, frame counts, IMU site assignment) and no check_manifest.py
gate has passed on it. It exists to unblock Run 0 while the corpus lands, and it
is superseded the moment src/manifest.py can run.

Usage:
    python tools/derive_index_from_cache.py --cache cache/ --out /tmp/m_derived.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from manifest import load_class_map  # noqa: E402  the same loader, imported not copied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--class-map", type=Path,
                    default=Path.home() / "cuhk-x/Small-Model-Track/class_mapping.csv")
    ap.add_argument("--out", type=Path, default=Path("/tmp/m_derived.csv"))
    args = ap.parse_args()

    class_map = load_class_map(args.class_map)
    cm = json.loads((args.cache / "cache_manifest.json").read_text())

    rows: dict[str, dict] = {}
    for shard in cm["shards"]:
        split = shard["split"]
        for cid in shard["clip_ids"]:
            if cid in rows:
                continue
            if split == "train":
                action, user, _trial = cid.split("/")
                if action not in class_map:
                    raise SystemExit(f"clip {cid!r}: action {action!r} not in class map")
                user_n = int(re.sub(r"\D", "", user) or -1)
                if user_n < 0:
                    raise SystemExit(f"clip {cid!r}: no user number in {user!r}")
                rows[cid] = {"clip_id": cid, "split": "train",
                             "action_id": class_map[action], "user": user_n}
            else:
                rows[cid] = {"clip_id": cid, "split": "test", "action_id": -1, "user": -1}

    ordered = sorted(rows.values(), key=lambda r: (r["split"], r["clip_id"]))
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "split", "user", "action_id"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(ordered)

    n_train = sum(r["split"] == "train" for r in ordered)
    n_test = len(ordered) - n_train
    users = sorted({r["user"] for r in ordered if r["split"] == "train"})
    actions = {r["action_id"] for r in ordered if r["split"] == "train"}
    print(f"[derive] {n_train} train + {n_test} test -> {args.out}", file=sys.stderr)
    print(f"[derive] {len(users)} users {users}", file=sys.stderr)
    print(f"[derive] {len(actions)} distinct action_ids, "
          f"range {min(actions)}..{max(actions)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
