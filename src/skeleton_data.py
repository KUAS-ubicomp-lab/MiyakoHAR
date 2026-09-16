"""Skeleton branch data chain — parse → person-select → features → cache (D8 / P2).

WHY THIS FILE EXISTS. D8 ruled BUILD (E20-gated) and m146 ④ recorded that no
skeleton code was ever executed. This is the data half: it turns the corpus's
per-frame pose JSONs into fixed-size per-clip tensors the training path can
memmap, exactly to configs/baseline.yaml's skeleton block (T=24, H36M-17,
xyz + first difference = 102 dims/timestep).

RULES THAT SHAPE THIS FILE.
- D14(d) leak hygiene: pose CONTENT only. The trailing frame index in each
  filename is used for ONE thing — ordering frames within a clip (temporal
  sync, the class D3 keeps legal). No feature, split, or label may derive from
  filenames, timestamps, or frame IDs, and none does.
- D8 person selection: the person with MINIMUM left/right limb-length
  asymmetry, default index 0. Up to 4 persons per frame exist (931 frame
  files) — `len(d) <= 2` is unsafe (M-10).
- Scores are all 1.0 across the corpus (D8) and carry nothing; ignored.

VERIFICATION BUILT IN, NOT BOLTED ON. M-10 recorded three corpus facts this
chain must reproduce or stop being trusted: multi-person frame rate train
~4.1% vs test ~14.8%, and the selection rule cutting mean asymmetry
0.300 → 0.242. build_cache() prints all three next to M-10's values.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# H36M-17 joint order (VideoPose3D convention; topology confirmed by D8):
#  0 Hip · 1 RHip · 2 RKnee · 3 RAnkle · 4 LHip · 5 LKnee · 6 LAnkle
#  7 Spine · 8 Thorax · 9 Neck/Nose · 10 Head
#  11 LShoulder · 12 LElbow · 13 LWrist · 14 RShoulder · 15 RElbow · 16 RWrist
# Left/right bone pairs for the asymmetry rule: (L endpoints), (R endpoints).
LIMB_PAIRS = (
    ((4, 5), (1, 2)),     # upper leg
    ((5, 6), (2, 3)),     # lower leg
    ((11, 12), (14, 15)), # upper arm
    ((12, 13), (15, 16)), # forearm
)

T_OUT = 24
N_JOINTS = 17
N_FEATS = 102  # 17 joints × (xyz + Δxyz)

_FRAME_IDX = re.compile(r"_(\d+)\.json$")


def frame_order_key(name: str) -> int:
    """The trailing frame index — used ONLY to order frames (D14(d))."""
    m = _FRAME_IDX.search(name)
    return int(m.group(1)) if m else 0


def limb_asymmetry(kp: np.ndarray) -> float:
    """Mean |L−R| / mean(L,R) bone-length difference over the four limb pairs.

    Denominator note, measured not assumed: with |L−R|/(L+R) the corpus mean
    came out 0.117 — almost exactly half of M-10's recorded 0.242 — so M-10's
    convention was the mean-denominator form. Selection is argmin over persons,
    which the ×2 cannot change; only the reported unit did.
    """
    total = 0.0
    for (la, lb), (ra, rb) in LIMB_PAIRS:
        left = float(np.linalg.norm(kp[la] - kp[lb]))
        right = float(np.linalg.norm(kp[ra] - kp[rb]))
        total += abs(left - right) / (0.5 * (left + right) + 1e-8)
    return total / len(LIMB_PAIRS)


def select_person(persons: list) -> tuple[int, float]:
    """D8's rule: min limb asymmetry, default index 0. Returns (idx, asym)."""
    if not persons:
        return -1, float("nan")
    asyms = []
    for p in persons:
        kp = np.asarray(p["keypoints"], dtype=np.float32)
        if kp.shape != (N_JOINTS, 3):
            asyms.append(float("inf"))  # malformed detection never wins selection
            continue
        asyms.append(limb_asymmetry(kp))
    best = int(np.argmin(asyms))  # single person → 0; exact ties → lowest index
    return best, asyms[best]


def load_clip(pred_dir: Path) -> tuple[np.ndarray | None, int, int, float]:
    """(T_OUT, 102) features, n_frames, n_multiperson_frames, mean selected asym."""
    files = sorted(pred_dir.glob("*.json"), key=lambda p: frame_order_key(p.name))
    frames, n_multi, asum, acount = [], 0, 0.0, 0
    for f in files:
        persons = json.loads(f.read_text())
        if len(persons) > 1:
            n_multi += 1
        idx, asym = select_person(persons)
        if idx < 0:
            continue
        kp = np.asarray(persons[idx]["keypoints"], dtype=np.float32)
        if kp.shape != (N_JOINTS, 3):
            continue
        frames.append(kp)
        if np.isfinite(asym):
            asum += asym
            acount += 1
    if not frames:
        return None, 0, n_multi, float("nan")
    seq = np.stack(frames)                                   # (F, 17, 3)
    # Temporal resample to T_OUT by index interpolation over the ORDERED frames.
    src = np.linspace(0, len(seq) - 1, T_OUT)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (src - lo)[:, None, None].astype(np.float32)
    xyz = (1.0 - w) * seq[lo] + w * seq[hi]                  # (T, 17, 3)
    delta = np.diff(xyz, axis=0, prepend=xyz[:1])            # first difference, Δ[0]=0
    feats = np.concatenate([xyz, delta], axis=2)             # (T, 17, 6)
    return feats.reshape(T_OUT, N_FEATS), len(seq), n_multi, (asum / max(1, acount))


def skeleton_dir(split: str, clip_id: str, train_root: Path, test_root: Path) -> Path:
    if split == "train":
        return train_root / "Skeleton" / clip_id / "predictions"
    return test_root / clip_id / "Skeleton" / "predictions"


def build_cache(manifest: Path, train_root: Path, test_root: Path, out_dir: Path) -> int:
    rows = list(csv.DictReader(open(manifest)))
    stats = {}
    for split in ("train", "test"):
        srows = [r for r in rows if r["split"] == split]
        X = np.zeros((len(srows), T_OUT, N_FEATS), dtype=np.float32)
        present = np.zeros(len(srows), dtype=np.int8)
        clip_ids, labels, folds, users = [], [], [], []
        frames_tot = multi_tot = 0
        asyms = []
        for i, r in enumerate(srows):
            clip_ids.append(r["clip_id"])
            labels.append(int(r["action_id"]) if r.get("action_id") else -1)
            folds.append(int(r["val_fold"]) if r.get("val_fold") else -1)
            users.append(int(r["user"]) if r.get("user") else -1)
            d = skeleton_dir(split, r["clip_id"], train_root, test_root)
            if not d.is_dir():
                continue
            feats, n_frames, n_multi, masym = load_clip(d)
            frames_tot += n_frames
            multi_tot += n_multi
            if feats is None:
                continue
            X[i] = feats
            present[i] = 1
            if np.isfinite(masym):
                asyms.append(masym)
        out = out_dir / f"skeleton_{split}.npz"
        np.savez_compressed(
            out, X=X, present=present, clip_ids=np.array(clip_ids),
            labels=np.array(labels, dtype=np.int64),
            val_fold=np.array(folds, dtype=np.int64),
            user=np.array(users, dtype=np.int64),
        )
        rate = 100.0 * multi_tot / max(1, frames_tot)
        stats[split] = (len(srows), int(present.sum()), frames_tot, rate,
                        float(np.mean(asyms)) if asyms else float("nan"))
        print(f"[skeleton] {split}: {present.sum()}/{len(srows)} clips cached -> {out} "
              f"({out.stat().st_size/1e6:.1f} MB)")

    print("\n[skeleton] M-10 reproduction check (this chain vs the ruling's record):")
    print(f"  multi-person frame rate  train {stats['train'][3]:.1f}%  (M-10: ~4.1%)"
          f"   test {stats['test'][3]:.1f}%  (M-10: ~14.8%)")
    print(f"  mean SELECTED asymmetry  train {stats['train'][4]:.3f}  (M-10 after rule: ~0.242)")
    print(f"  frames parsed            train {stats['train'][2]:,}  test {stats['test'][2]:,}"
          f"  (M-10 parsed 86,050 + 9,200, 0 unparseable)")
    return 0


def main() -> int:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m_fresh.csv"))
    ap.add_argument("--train-root", type=Path, default=home / "cuhk-x/extracted/HAR/data")
    ap.add_argument("--test-root", type=Path,
                    default=home / "cuhk-x/test_extracted/small_model_track_test")
    ap.add_argument("--out", type=Path, default=ROOT / "cache")
    a = ap.parse_args()
    return build_cache(a.manifest, a.train_root, a.test_root, a.out)


if __name__ == "__main__":
    raise SystemExit(main())


# ── The training-path dataset ────────────────────────────────────────────────
# Emits EXACTLY ClipDataset's item schema (x, y, clip_id, user, n_frames,
# present) so src/dataset.py's collate and train.py's loops work unchanged.
# x is (T, 102) float32 — rank-3 batches; SkeletonTCN asserts the rank.

import torch  # noqa: E402  (kept below the parse half: that half is torch-free)
from torch.utils.data import Dataset  # noqa: E402

YAW_DEG = 15.0        # configs/baseline.yaml augment.skeleton.yaw_rotation_deg
JOINT_DROP_P = 0.05   # …and joint_dropout_p; both drawn ONCE PER CLIP


class SkeletonDataset(Dataset):
    """mode="train": rows with val_fold != fold, augmented. mode="val": == fold.

    Only skeleton-present clips enter either split (105 train clips lack pose
    frames); at fusion time their absence is handled by predict.py's
    renormalise-over-present convention, not by fabricating zeros here.
    """

    def __init__(self, mode: str, fold: int, cache: Path | None = None,
                 seed: int = 0) -> None:
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be train|val, got {mode!r}")
        z = np.load((cache or ROOT / "cache") / "skeleton_train.npz", allow_pickle=False)
        keep = z["present"].astype(bool) & (z["labels"] >= 0)
        keep &= (z["val_fold"] != fold) if mode == "train" else (z["val_fold"] == fold)
        self.X = z["X"][keep]
        self.y = z["labels"][keep]
        self.clip_ids = [str(c) for c in z["clip_ids"][keep]]
        self.users = z["user"][keep]
        self.augment = mode == "train"
        self.seed = seed
        if not len(self.X):
            raise RuntimeError(f"skeleton {mode} split for fold {fold} is empty")

    def __len__(self) -> int:
        return len(self.X)

    def _augmented(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        v = x.reshape(T_OUT, N_JOINTS, 6).copy()
        # Yaw about the vertical (z) axis — a rotation in the x-y plane, applied
        # identically to positions and first differences (both are vectors).
        th = np.deg2rad(rng.uniform(-YAW_DEG, YAW_DEG))
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        for cols in ((0, 1), (3, 4)):                    # (x,y) and (dx,dy)
            v[:, :, cols] = v[:, :, cols] @ rot.T
        drop = rng.random(N_JOINTS) < JOINT_DROP_P       # per clip, all T frames
        v[:, drop, :] = 0.0
        return v.reshape(T_OUT, N_FEATS)

    def __getitem__(self, i: int) -> dict:
        x = self.X[i]
        if self.augment:
            # ClipDataset's seeding lesson, inherited verbatim: the spark comes
            # from torch's per-worker per-epoch RNG so a clip's augmentation
            # varies across epochs; (seed, i) alone would freeze it forever.
            spark = int(torch.randint(0, 2**31 - 1, (1,)).item())
            x = self._augmented(x, np.random.default_rng((self.seed, i, spark)))
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": int(self.y[i]),
            "clip_id": self.clip_ids[i],
            "user": int(self.users[i]),
            "n_frames": T_OUT,
            "present": True,
        }
