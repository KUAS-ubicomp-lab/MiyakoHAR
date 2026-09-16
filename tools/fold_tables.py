"""Aggregate tables from the held-out per-clip predictions.

Reads evaluation/fold_predictions_022_026.csv (written by tools/fold_predictions.py)
and writes, in the same directory:

    per_activity_accuracy.csv           accuracy per activity: each recipe fused, each network alone
    per_subject_accuracy.csv            the same per training subject
    confusion_fused_026.csv             40 x 40 confusion matrix of the 026 recipe, both seeds pooled
    confusion_fused_022.csv             the same for the 022 recipe
    clips_by_activity_and_subject.csv   the training split's clip count per (activity, subject)

With --markdown it also prints the Markdown tables that docs/held-out-predictions.md
and docs/data.md carry, so the documents can be regenerated from the data.

Accuracy is the share of correct predictions over clips and seeds: each clip is
predicted once per seed, so a clip counts twice. A network alone is scored on the
clips that carry its stream; the fused model on every clip. Needs only the CSV
and numpy, no dataset access.

Usage:
    python tools/fold_tables.py [--predictions evaluation/fold_predictions_022_026.csv] [--markdown]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (1, 2)
N_CLASSES = 40
# Column family -> label in the tables. Order is the column order of the accuracy CSVs.
MEMBERS = [
    ("fused_026", "Fused 026"),
    ("fused_022", "Fused 022"),
    ("depth_ir", "Depth + infrared alone"),
    ("thermal_026", "Thermal alone (026)"),
    ("thermal_022", "Thermal alone (022)"),
]


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key in ("label", "subject", "fold", "has_depth_ir", "has_thermal"):
            r[key] = int(r[key])
    return rows


def rate(rows: list[dict], member: str) -> float:
    """Percent correct over clips and seeds, on the predictions that exist for `member`."""
    hits = total = 0
    for r in rows:
        for seed in SEEDS:
            pred = r[f"pred_{member}_seed{seed}"]
            if pred != "":
                total += 1
                hits += int(pred) == r["label"]
    return 100.0 * hits / total if total else float("nan")


def fmt(x: float, digits: int) -> str:
    return "n/a" if np.isnan(x) else f"{x:.{digits}f}"


def per_activity(rows: list[dict]) -> list[dict]:
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    out = []
    for label in sorted(by_label):
        group = by_label[label]
        entry = {"label": label, "activity": group[0]["activity"], "clips": len(group),
                 "subjects": len({r["subject"] for r in group})}
        for member, _ in MEMBERS:
            entry[f"acc_{member}"] = rate(group, member)
        out.append(entry)
    return out


def per_subject(rows: list[dict]) -> list[dict]:
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    out = []
    for subject in sorted(by_subject):
        group = by_subject[subject]
        entry = {"subject": subject, "fold": group[0]["fold"], "clips": len(group)}
        for member, _ in MEMBERS:
            entry[f"acc_{member}"] = rate(group, member)
        out.append(entry)
    return out


def confusion(rows: list[dict], member: str) -> np.ndarray:
    C = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for r in rows:
        for seed in SEEDS:
            pred = r[f"pred_{member}_seed{seed}"]
            if pred != "":
                C[r["label"], int(pred)] += 1
    return C


def clip_counts(rows: list[dict]) -> tuple[list[int], list[int], dict, np.ndarray]:
    labels = sorted({r["label"] for r in rows})
    subjects = sorted({r["subject"] for r in rows})
    names = {r["label"]: r["activity"] for r in rows}
    counts = np.zeros((len(labels), len(subjects)), dtype=np.int64)
    for r in rows:
        counts[labels.index(r["label"]), subjects.index(r["subject"])] += 1
    return labels, subjects, names, counts


def write_csv(path: Path, header: list[str], body: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictions", type=Path, default=ROOT / "evaluation" / "fold_predictions_022_026.csv")
    ap.add_argument("--markdown", action="store_true", help="also print the Markdown tables for the docs")
    args = ap.parse_args()
    out_dir = args.predictions.parent
    rows = read_predictions(args.predictions)
    names = {r["label"]: r["activity"] for r in rows}
    acc_cols = [f"acc_{m}" for m, _ in MEMBERS]

    activities = per_activity(rows)
    write_csv(out_dir / "per_activity_accuracy.csv", ["label", "activity", "clips", "subjects"] + acc_cols,
              [[a["label"], a["activity"], a["clips"], a["subjects"]] + [fmt(a[c], 2) for c in acc_cols] for a in activities])

    subjects_tbl = per_subject(rows)
    write_csv(out_dir / "per_subject_accuracy.csv", ["subject", "fold", "clips"] + acc_cols,
              [[s["subject"], s["fold"], s["clips"]] + [fmt(s[c], 2) for c in acc_cols] for s in subjects_tbl])

    matrices = {}
    for recipe in ("026", "022"):
        C = confusion(rows, f"fused_{recipe}")
        matrices[recipe] = C
        head = [f"{i} {names[i]}" for i in range(N_CLASSES)]
        write_csv(out_dir / f"confusion_fused_{recipe}.csv", ["true \\ predicted"] + head,
                  [[head[i]] + C[i].tolist() for i in range(N_CLASSES)])

    labels, subjects, _, counts = clip_counts(rows)
    write_csv(out_dir / "clips_by_activity_and_subject.csv",
              ["label", "activity"] + [f"user{u}" for u in subjects] + ["clips", "subjects_with_clips"],
              [[lab, names[lab]] + counts[i].tolist() + [int(counts[i].sum()), int((counts[i] > 0).sum())]
               for i, lab in enumerate(labels)]
              + [["", "all activities"] + counts.sum(axis=0).tolist() + [int(counts.sum()), ""]])
    print(f"wrote five tables to {out_dir}/ from {len(rows)} clips")

    if not args.markdown:
        return 0

    print("\n### Accuracy per activity\n")
    print("| Id | Activity | Clips | Subjects | " + " | ".join(label for _, label in MEMBERS) + " |")
    print("|---:|---|---:|---:|" + "---:|" * len(MEMBERS))
    for a in activities:
        print(f"| {a['label']} | {a['activity']} | {a['clips']} | {a['subjects']} | "
              + " | ".join(fmt(a[c], 1) for c in acc_cols) + " |")
    print(f"| | All clips, clip-weighted | {len(rows)} | {len(subjects)} | "
          + " | ".join(f"**{rate(rows, m):.1f}**" for m, _ in MEMBERS) + " |")

    print("\n### Accuracy per subject\n")
    print("| Subject | Fold | Clips | " + " | ".join(label for _, label in MEMBERS) + " |")
    print("|---:|---:|---:|" + "---:|" * len(MEMBERS))
    for s in subjects_tbl:
        print(f"| {s['subject']} | {s['fold']} | {s['clips']} | " + " | ".join(fmt(s[c], 1) for c in acc_cols) + " |")

    C = matrices["026"]
    errors = int(C.sum() - np.trace(C))
    pairs = sorted(((int(C[i, j] + C[j, i]), i, j) for i in range(N_CLASSES) for j in range(i + 1, N_CLASSES)),
                   reverse=True)[:10]
    print(f"\n### Most confused pairs, 026 recipe ({errors} errors over {int(C.sum())} clip-seed predictions)\n")
    print("| Pair | Confusions, both directions | Share of all errors |")
    print("|---|---:|---:|")
    for n, i, j in pairs:
        print(f"| {names[i]} and {names[j]} | {n} | {100.0 * n / errors:.1f}% |")

    print("\n### Training clips per activity and subject\n")
    print("| Id | Activity | " + " | ".join(f"u{u}" for u in subjects) + " | Clips | Subj. |")
    print("|---:|---|" + "---:|" * len(subjects) + "---:|---:|")
    for i, lab in enumerate(labels):
        cells = " | ".join(str(c) if c else "." for c in counts[i].tolist())
        print(f"| {lab} | {names[lab]} | {cells} | **{int(counts[i].sum())}** | {int((counts[i] > 0).sum())} |")
    print("| | **All activities** | " + " | ".join(f"**{int(c)}**" for c in counts.sum(axis=0))
          + f" | **{int(counts.sum())}** | |")
    print("| | Activities with clips | " + " | ".join(str(int(c)) for c in (counts > 0).sum(axis=0)) + " | | |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
