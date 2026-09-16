"""Held-out predictions of the 022 and 026 recipes for every training clip.

WHAT IT PRODUCES. One CSV row per training clip (evaluation/fold_predictions_022_026.csv):
the clip's directory path, its label and subject, the fold in which the clip was
held out, which streams the clip carries, and the predicted class of each network
and of the fused model, for both final recipes and both training seeds. Every
training subject is held out exactly once, so the file is an out-of-fold
prediction of the whole training split.

WHERE THE NUMBERS COME FROM. The recipe-selection instrument of docs/training.md.
For each of the three folds of splits/folds.yaml and each of two seeds, the
depth + infrared member (ir-CSN-R50, T = 16) and the thermal member (ir-CSN-152,
T = 32, SWAD weights) were trained on the fold's 12 subjects, and their logits
on the 6 held-out subjects were cached per run as `val_logits*.npz` with the
arrays `logits` (N x 40), `labels` (N) and `clip_ids` (N); the `_hflip` file
holds the horizontally flipped pass. The 026 recipe shares the depth + infrared
member and differs in its thermal member, trained on frames decoded at 320 x 240
with a 320 x 320 input. Those caches are development outputs in the training
tree's `runs/` directory and are not part of this repository; this tool records
the derivation and re-runs it where the caches exist.

THE ESTIMATOR, as src/predict.py fuses at inference: per network, the mean of the
softmax of the upright pass and of the flipped pass; per clip, the mean over the
networks present; the prediction is the class with the largest probability.

CONTROL. The fused accuracy of each of the 12 (recipe, fold, seed) cells is
compared with the cell published in docs/training.md. A difference above 0.005
points stops the run and nothing is written.

Usage:
    python tools/fold_predictions.py --runs-dir /path/to/runs \
        --out evaluation/fold_predictions_022_026.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

FOLDS = (0, 1, 2)
SEEDS = (1, 2)
RECIPES = ("022", "026")

# Run names of the 18 fold cells in the training tree. The depth + infrared member
# is shared by both recipes; the recipes differ in the thermal member.
DEPTH_IR_RUN = {1: "labpc_s3_depthir_f{fold}_t16", 2: "labpc_seed2_depthir_f{fold}"}
THERMAL_RUN = {
    "022": "labpc_r152stack_thermal_f{fold}_s{seed}",
    "026": "labpc_hr320_r152stack_thermal_f{fold}_s{seed}",
}
# The thermal caches were written from the SWAD weights, the form that ships.
THERMAL_TAG = "_swadlofo"

# Fused accuracy per cell, in percent, as published in docs/training.md
# ("How the recipe was selected"), indexed [recipe][seed][fold].
PUBLISHED = {
    "022": {1: (73.975, 72.693, 71.834), 2: (74.898, 72.227, 70.719)},
    "026": {1: (76.025, 73.253, 75.076), 2: (76.332, 72.600, 73.354)},
}
TOLERANCE = 0.005

COLUMNS = ["clip_id", "label", "activity", "subject", "fold", "has_depth_ir", "has_thermal"] + [
    f"pred_{member}_seed{seed}"
    for seed in SEEDS
    for member in ("depth_ir", "thermal_022", "fused_022", "thermal_026", "fused_026")
]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def branch_probs(run_dir: Path, tag: str = "") -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """{clip_id: mean softmax over the upright and flipped passes}, {clip_id: label}."""
    upright = np.load(run_dir / f"val_logits{tag}.npz", allow_pickle=False)
    flipped = np.load(run_dir / f"val_logits{tag}_hflip.npz", allow_pickle=False)
    clips = [str(c) for c in upright["clip_ids"]]
    if clips != [str(c) for c in flipped["clip_ids"]]:
        raise SystemExit(f"{run_dir.name}: the upright and flipped caches list different clips")
    probs = 0.5 * (softmax(upright["logits"]) + softmax(flipped["logits"]))
    labels = upright["labels"].astype(int)
    return dict(zip(clips, probs)), dict(zip(clips, labels.tolist()))


def fuse(*branches: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean over the networks present for each clip, as predict.py does at inference."""
    fused = {}
    for clip in set().union(*branches):
        fused[clip] = np.mean([b[clip] for b in branches if clip in b], axis=0)
    return fused


def accuracy(probs: dict[str, np.ndarray], labels: dict[str, int]) -> float:
    return 100.0 * float(np.mean([int(probs[c].argmax()) == labels[c] for c in probs]))


def activity_name(clip_id: str) -> str:
    """'21_Read_documents/user4/1-1-1' -> 'Read documents'."""
    return clip_id.split("/")[0].split("_", 1)[1].replace("_", " ")


def subject_of(clip_id: str) -> int:
    return int(clip_id.split("/")[1].removeprefix("user"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs",
                    help="directory holding the 18 fold runs and their val_logits caches")
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation" / "fold_predictions_022_026.csv")
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            depth_ir, labels = branch_probs(args.runs_dir / DEPTH_IR_RUN[seed].format(fold=fold))
            thermal = {}
            for recipe in RECIPES:
                run = args.runs_dir / THERMAL_RUN[recipe].format(fold=fold, seed=seed)
                thermal[recipe], thermal_labels = branch_probs(run, THERMAL_TAG)
                labels.update(thermal_labels)
            fused = {recipe: fuse(depth_ir, thermal[recipe]) for recipe in RECIPES}

            for recipe in RECIPES:
                got = accuracy(fused[recipe], labels)
                want = PUBLISHED[recipe][seed][fold]
                status = "agrees" if abs(got - want) <= TOLERANCE else "DISAGREES"
                print(f"fold {fold} seed {seed} recipe {recipe}: fused {got:.3f}, published {want:.3f}, {status}")
                if status != "agrees":
                    raise SystemExit("a fused cell does not reproduce the published value; nothing written")

            for clip in sorted(fused["022"]):
                label = labels[clip]
                if label != int(clip.split("_", 1)[0]):
                    raise SystemExit(f"{clip}: the cached label disagrees with the clip's directory")
                row = rows.setdefault(clip, {
                    "clip_id": clip, "label": label, "activity": activity_name(clip),
                    "subject": subject_of(clip), "fold": fold,
                    "has_depth_ir": int(clip in depth_ir), "has_thermal": int(clip in thermal["022"]),
                })
                row[f"pred_depth_ir_seed{seed}"] = int(depth_ir[clip].argmax()) if clip in depth_ir else ""
                for recipe in RECIPES:
                    t = thermal[recipe]
                    row[f"pred_thermal_{recipe}_seed{seed}"] = int(t[clip].argmax()) if clip in t else ""
                    row[f"pred_fused_{recipe}_seed{seed}"] = int(fused[recipe][clip].argmax())

    ordered = sorted(rows.values(), key=lambda r: (r["label"], r["subject"], r["clip_id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"wrote {args.out} ({len(ordered)} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
