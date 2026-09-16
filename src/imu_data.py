"""IMU branch data chain — parse → per-site order → index-resample → cache (M-15 / QUEUE §65; D23).

WHY THIS FILE EXISTS. M-15 (m47) ruled an IMU branch BUILDABLE from accel(3) + gyro(3) + pitch + roll,
and it was never built (day-1 `drop_modalities`). §65 builds the cache and measures separability BEFORE
any network exists, so a discouraging reading closes the question at desk cost. Mirror of
src/skeleton_data.py: parse → features → `.npz` cache → a Dataset emitting ClipDataset's item schema.

THE CLAUSES THIS FILE IS WRITTEN TO (GOAL-SPRINT-3 §7; each mutation-tested in tests/check_imu_data.py):
  L1 Timestamp — no timestamp VALUE enters any tensor. The time column ORDERS a site's rows (a permissive
     integer-field key over the 13 shapes m216 (c) found — never strptime, never epoch seconds) and the
     ordered rows are INDEX-resampled to T_OUT (skeleton_data.py:115–120's np.linspace precedent). The
     tensor is invariant to a constant offset on every timestamp and to any order-preserving re-labelling.
  L2 Device — 设备名称/DeviceName → the `WT*` prefix → a hard-coded five-slot table {LA, RA, C, LL, RL};
     an unknown prefix RAISES; the string (and the MAC inside it) never enters a tensor.
  L3 Channel — exactly EIGHT channels per site: accel xyz (idx 2–4), gyro xyz (5–7), pitch, roll (8–9),
     positional in BOTH schemas (B §2.6: indices 2–18 align; the schemas diverge at 19). Yaw (10),
     magnetometer (11–13), quaternion (14–17), temperature (18), version / height / pressure / battery
     (19+) are never read.
  L4 Filename — files are located by `*.csv` and classified by CONTENT (the device column); no parse
     branch on language, order or spacing; no feature from the variant.
  L5 Presence — the per-site mask records which sites contributed (absent sites are zero-filled). It is
     bookkeeping for the cache and the tests; it is NEVER a feature, a count, or a fitted weight, and it
     is not fed to a network (a missing-site flag is D2's session trap in another modality).
  L7 Statistics — pitch/roll are centred per clip, per site (the wrapped deviation from the site's own
     circular mean over the clip — a within-clip statistic, m216 (b); the Euler columns wrap at ±180°);
     nothing else is normalised here (fold-train z-scoring, if any, is the model's and is stamped into
     its checkpoint, §66).

VERIFICATION BUILT IN, NOT BOLTED ON. build_cache() prints the reproduction block against m47/m216 the way
skeleton_data.py prints against M-10: files, rows, unparseable, header-only files, all-five-site clips,
‖accel‖ median, the 2,186 English-schema rows — and reconciles per-clip row counts with the manifest's
`imu_n` (the sha-gated index) so the chain and the index cannot disagree silently.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# L2: the five-slot table. Identical to manifest.IMU_SITE_BY_DEVICE (asserted by tests/check_imu_data.py);
# duplicated rather than imported because manifest.py is sha-gated and this file must also load in the
# script context predict.py uses (sys.path = src/).
SITES = ("LA", "RA", "C", "LL", "RL")
SITE_BY_PREFIX = {"WTLA": "LA", "WTRA": "RA", "WTC": "C", "WTLL": "LL", "WTRL": "RL"}

T_OUT = 24                    # §65: per-site index-resample to T = 24 (m13's "~24 timesteps"; not tuned)
N_CH = 8                      # L3: accel xyz, gyro xyz, pitch, roll
N_SITES = len(SITES)
N_FEATS = N_CH * N_SITES      # 40
CH_IDX = (2, 3, 4, 5, 6, 7, 8, 9)   # positional in BOTH schemas
EULER_SLICE = slice(6, 8)     # pitch, roll inside the 8 (centred per clip, per site)
MIN_COLS = 19                 # "the 19 aligned columns are the contract" (manifest.py)

# The 13 timestamp shapes (m216 c): 4-digit year, 1–2-digit month/day/hour/minute/second, 0–6 fractional
# digits. Captured as INTEGER FIELDS for ordering only — no calendar arithmetic, no epoch.
_TS = re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.(\d{1,6}))?\s*$")


def ts_key(s: str) -> tuple[int, int, int, int, int, int, int]:
    """An ORDERING key for a timestamp string — used for nothing else (L1)."""
    m = _TS.match(s)
    if m is None:
        raise ValueError(f"unparseable timestamp {s!r}")
    y, mo, d, h, mi, sec, frac = m.groups()
    return (int(y), int(mo), int(d), int(h), int(mi), int(sec), int((frac or "0").ljust(6, "0")))


def site_of(device_cell: str) -> str:
    """設備名稱/DeviceName cell → site slot. The MAC in parentheses is discarded before the lookup (L2)."""
    prefix = device_cell.strip().split("(", 1)[0].strip().upper()
    site = SITE_BY_PREFIX.get(prefix)
    if site is None:
        raise ValueError(f"unknown IMU device prefix {prefix!r} — not in the five-slot table (L2)")
    return site


def read_rows(path: Path) -> list[list[str]]:
    """utf-8-sig: a BOM renames the first header otherwise (manifest.py trap 1)."""
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        return [r for r in csv.reader(fh)]


def parse_file(path: Path) -> tuple[dict[str, list[tuple[tuple, np.ndarray]]], dict]:
    """One CSV → {site: [(order_key, 8 values), …]} plus the counters the reproduction block prints.

    Content-classified: the site of every row comes from its device cell; the filename is never read (L4).
    Rows narrower than MIN_COLS are skipped (the manifest's contract); a row whose eight channels do not
    parse as floats, or whose timestamp does not match the 13 shapes, is counted as unparseable and skipped.
    An unknown device prefix RAISES (L2).
    """
    per_site: dict[str, list] = {s: [] for s in SITES}
    st = {"rows": 0, "unparseable": 0, "en_rows": 0, "width": 0, "header_only": 0}
    rows = read_rows(path)
    if not rows:
        st["header_only"] = 1
        return per_site, st
    header = rows[0]
    st["width"] = len(header)
    # The schema tag is COUNTED for the m216 reproduction print and used for nothing else — the eight
    # channels sit at the same positions in both schemas, so there is no parse branch on it (L4).
    is_en = "DeviceName" in ",".join(header)
    n_data = 0
    for row in rows[1:]:
        if len(row) < MIN_COLS:
            continue
        n_data += 1
        site = site_of(row[1])
        try:
            key = ts_key(row[0])
            vals = np.array([float(row[i]) for i in CH_IDX], dtype=np.float32)
        except ValueError:
            st["unparseable"] += 1
            continue
        st["rows"] += 1
        if is_en and len(row) >= 23:
            st["en_rows"] += 1
        per_site[site].append((key, vals))
    if n_data == 0:
        st["header_only"] = 1
    return per_site, st


def centre_angles(A: np.ndarray) -> np.ndarray:
    """(n, 2) pitch/roll in degrees → the wrapped deviation from the site's CIRCULAR mean over the clip.

    The Euler columns wrap at ±180° (an arm unit straddles it: 0_Wash_face/user4/1-1-1's left-arm pitch
    runs −180…+180 within one clip), so a plain median subtraction manufactures 360° jumps. The reference
    is atan2(mean sin, mean cos) per column — a within-clip statistic (L7) — and the deviation is wrapped
    into (−180, 180]. Still exactly two channels; nothing about the sensor frame is assumed.
    """
    rad = np.deg2rad(A.astype(np.float64))
    ref = np.arctan2(np.sin(rad).mean(0), np.cos(rad).mean(0))
    dev = np.rad2deg(np.angle(np.exp(1j * (rad - ref))))
    return dev.astype(np.float32)


def resample_site(V: np.ndarray) -> np.ndarray:
    """(n, 8) site-ordered rows → (T_OUT, 8) by index interpolation (no time value used — L1)."""
    n = len(V)
    src = np.linspace(0, n - 1, T_OUT)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None].astype(np.float32)
    return (1.0 - w) * V[lo] + w * V[hi]


def clip_tensor(per_site: dict[str, list], sort: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """{site: rows} → X (T_OUT, 40) float32 site-major, present (5,) int8.

    `sort=False` exists ONLY for tests/check_imu_data.py's T-L1b (it proves the sort is load-bearing);
    every production path sorts. Ties on the order key are broken by the row's own values, so the order is
    a function of content alone — never of which file a row came from (L4).
    """
    X = np.zeros((T_OUT, N_FEATS), dtype=np.float32)
    present = np.zeros(N_SITES, dtype=np.int8)
    for si, s in enumerate(SITES):
        rows = per_site.get(s) or []
        if not rows:
            continue
        if sort:
            rows = sorted(rows, key=lambda r: (r[0], r[1].tobytes()))
        V = np.stack([r[1] for r in rows]).astype(np.float32)        # (n, 8)
        V[:, EULER_SLICE] = centre_angles(V[:, EULER_SLICE])          # L7: within-clip, within-site
        X[:, si * N_CH:(si + 1) * N_CH] = resample_site(V)
        present[si] = 1
    return X, present


def csv_files(imu_dir: Path) -> list[Path]:
    """Every *.csv under the clip's IMU directory (case-insensitive) — located, never interpreted (L4)."""
    if not imu_dir.is_dir():
        return []
    return sorted(p for p in imu_dir.iterdir() if p.is_file() and p.name.lower().endswith(".csv"))


def load_clip(imu_dir: Path, sort: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """A clip's IMU directory → (X, present, stats). Sites are merged ACROSS files (a swapped or
    all-in-one file changes nothing — content decides). Raises only per L2 (unknown device); an absent
    directory, no CSVs, header-only files or zero usable rows give present = 0 everywhere, X = 0."""
    merged: dict[str, list] = {s: [] for s in SITES}
    st = Counter(files=0)
    accel_norms: list[np.ndarray] = []
    for p in csv_files(imu_dir):
        st["files"] += 1
        per_site, fst = parse_file(p)
        for k, v in fst.items():
            st[k] += v
        for s in SITES:
            merged[s].extend(per_site[s])
    for s in SITES:
        if merged[s]:
            accel_norms.append(np.linalg.norm(np.stack([r[1][:3] for r in merged[s]]), axis=1))
    X, present = clip_tensor(merged, sort=sort)
    stats = dict(st)
    stats["accel_norms"] = np.concatenate(accel_norms) if accel_norms else np.zeros(0, dtype=np.float32)
    stats["n_sites"] = int(present.sum())
    return X, present, stats


def imu_dir(split: str, clip_id: str, train_root: Path, test_root: Path) -> Path:
    if split == "train":
        return train_root / "IMU" / clip_id
    return test_root / clip_id / "IMU"


def build_cache(manifest: Path, train_root: Path, test_root: Path, out_dir: Path) -> int:
    rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8-sig")))
    rep = {}
    for split in ("train", "test"):
        srows = [r for r in rows if r["split"] == split]
        X = np.zeros((len(srows), T_OUT, N_FEATS), dtype=np.float32)
        present = np.zeros((len(srows), N_SITES), dtype=np.int8)
        clip_ids, labels, folds, users = [], [], [], []
        tot = Counter()
        sites_hist: Counter = Counter()
        norms: list[np.ndarray] = []
        manifest_rows = manifest_present = 0
        mismatched_rows = 0
        for i, r in enumerate(srows):
            clip_ids.append(r["clip_id"])
            labels.append(int(r["action_id"]) if r.get("action_id") not in (None, "", "-1") else -1)
            folds.append(int(r["val_fold"]) if r.get("val_fold") not in (None, "") else -1)
            users.append(int(r["user"]) if r.get("user") not in (None, "") else -1)
            manifest_rows += int(r.get("imu_n") or 0)
            manifest_present += int(r.get("imu_present") or 0)
            d = imu_dir(split, r["clip_id"], train_root, test_root)
            if not d.is_dir():
                continue
            x, p, st = load_clip(d)
            X[i], present[i] = x, p
            for k in ("files", "rows", "unparseable", "en_rows", "header_only"):
                tot[k] += st.get(k, 0)
            sites_hist[st["n_sites"]] += 1
            norms.append(st["accel_norms"])
            # The manifest's imu_n counts rows with ≥ 19 columns; this chain's `rows` are those that also
            # parsed. With 0 unparseable the two must agree per clip — a disagreement is a loader bug.
            if int(r.get("imu_n") or 0) != st.get("rows", 0) + st.get("unparseable", 0):
                mismatched_rows += 1
        out = out_dir / f"imu_{split}.npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, X=X, present=present, clip_ids=np.array(clip_ids),
            labels=np.array(labels, dtype=np.int64),
            val_fold=np.array(folds, dtype=np.int64),
            user=np.array(users, dtype=np.int64),
        )
        any_present = int((present.sum(1) > 0).sum())
        all_five = int((present.sum(1) == N_SITES).sum())
        an = np.concatenate(norms) if norms else np.zeros(0)
        rep[split] = dict(clips=len(srows), any=any_present, five=all_five, manifest_present=manifest_present,
                          manifest_rows=manifest_rows, mismatched=mismatched_rows, hist=dict(sorted(sites_hist.items())),
                          accel_med=float(np.median(an)) if len(an) else float("nan"), **tot)
        print(f"[imu] {split}: {any_present}/{len(srows)} clips with IMU content cached -> {out} "
              f"({out.stat().st_size / 1e6:.1f} MB); X {X.shape}, present {present.shape}")

    t, e = rep["train"], rep["test"]
    print("\n[imu] m47 / m216 reproduction check (this chain vs the record):")
    print(f"  files              {t['files'] + e['files']:,}  = train {t['files']:,} + test {e['files']:,}"
          f"   (record: 6,616 = 5,806 + 810)")
    print(f"  rows (≥19 cols)    {t['rows'] + e['rows']:,}  = {t['rows']:,} + {e['rows']:,}"
          f"   (record: 445,387 = 401,717 + 43,670)   unparseable {t['unparseable'] + e['unparseable']} (record: 0)")
    print(f"  manifest imu_n     {t['manifest_rows']:,} + {e['manifest_rows']:,}; clips whose row count disagrees with "
          f"the manifest: {t['mismatched']} + {e['mismatched']}  (must be 0)")
    print(f"  header-only files  train {t['header_only']}  test {e['header_only']}   (record: 128 / 4)")
    print(f"  all-five-site      train {t['five']}/{t['any']} (manifest imu_present {t['manifest_present']})"
          f"   test {e['five']}/{e['any']} (manifest {e['manifest_present']})   (record: 2,785/2,863 · 401/404)")
    print(f"  sites per clip     train {t['hist']}   test {e['hist']}")
    print(f"  ‖accel‖ median     train {t['accel_med']:.4f} g   test {e['accel_med']:.4f} g   (record: ≈ 0.995 g)")
    print(f"  23-col English rows (test) {e['en_rows']:,}   (record: 2,186 — m216 (a))")
    return 0


def main() -> int:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m_fresh.csv"))
    ap.add_argument("--train-root", type=Path, default=home / "cuhk-x/extracted/HAR/data")
    ap.add_argument("--test-root", type=Path, default=home / "cuhk-x/test_extracted/small_model_track_test")
    ap.add_argument("--out", type=Path, default=ROOT / "cache")
    a = ap.parse_args()
    return build_cache(a.manifest, a.train_root, a.test_root, a.out)


if __name__ == "__main__":
    raise SystemExit(main())


# ── The training-path dataset (QUEUE §66) ────────────────────────────────────
# Emits EXACTLY ClipDataset's item schema (x, y, clip_id, user, n_frames, present) so src/dataset.py's
# collate and train.py's loops work unchanged. x is (T_OUT, 40) float32 — the same rank-3 layout as the
# skeleton branch, so a SkeletonTCN-shaped network consumes it after one transpose.

import torch  # noqa: E402  (kept below the parse half: that half is torch-free)
from torch.utils.data import Dataset  # noqa: E402

ROT_DEG = 15.0       # §66 (a): per-site rotation of the accel/gyro triplets about the vertical, ±15°
SITE_DROP_P = 0.05   # §66 (b): per-site dropout — doubles as the missing-site simulation the test set needs
TCROP_MIN = 0.5      # §66 (c): time-crop keeps ≥ half the clip (the T6 shape), re-resampled to T_OUT
# NOT used (§66): left/right site mirroring (handedness is a subject property), magnitude scaling (destroys
# the 1 g anchor), anything from time-of-day.


def _rot_about(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    a = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class IMUDataset(Dataset):
    """mode="train": rows with val_fold != fold, augmented. mode="val": == fold.

    Only IMU-present clips enter either split (173 train clips have no IMU content); at fusion time their
    absence is handled by the renormalise-over-present convention, not by fabricating zeros here. Sites
    absent within a present clip stay zero-filled (L5) — the mask is not an input.
    """

    def __init__(self, mode: str, fold: int, cache: Path | None = None, seed: int = 0,
                 all_subjects: bool = False, shuffle_labels: bool = False) -> None:
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be train|val, got {mode!r}")
        z = np.load((cache or ROOT / "cache") / "imu_train.npz", allow_pickle=False)
        keep = (z["present"].sum(1) > 0) & (z["labels"] >= 0)
        if mode == "train" and all_subjects:
            pass                                      # J9: every labelled subject trains; no val exists
        else:
            keep &= (z["val_fold"] != fold) if mode == "train" else (z["val_fold"] == fold)
        self.X = z["X"][keep]
        self.present = z["present"][keep]
        self.y = z["labels"][keep]
        if shuffle_labels:
            # §66 control (a): the SAME clips, the SAME recipe, the labels permuted once with a fixed
            # seed — the member whose fused Δ is §67's dilution floor. Train mode only.
            if mode != "train":
                raise ValueError("shuffle_labels is a training-time control; the val labels stay real")
            self.y = np.random.default_rng(20260813 + fold).permutation(self.y)
        self.clip_ids = [str(c) for c in z["clip_ids"][keep]]
        self.users = z["user"][keep]
        self.augment = mode == "train"
        self.seed = seed
        if not len(self.X):
            raise RuntimeError(f"imu {mode} split for fold {fold} is empty")

    def __len__(self) -> int:
        return len(self.X)

    def _augmented(self, x: np.ndarray, present: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        v = x.reshape(T_OUT, N_SITES, N_CH).copy()
        for si in range(N_SITES):
            if not present[si]:
                continue
            # (a) rotation about the vertical: the site's vertical is its mean gravity direction over the
            # clip (accel ≈ g at rest); the same rotation is applied to the accel and gyro triplets.
            g = v[:, si, 0:3].mean(0)
            if np.linalg.norm(g) > 1e-3:
                R = _rot_about(g, np.deg2rad(rng.uniform(-ROT_DEG, ROT_DEG)))
                v[:, si, 0:3] = v[:, si, 0:3] @ R.T
                v[:, si, 3:6] = v[:, si, 3:6] @ R.T
            # (b) site dropout — the missing-site simulation
            if rng.random() < SITE_DROP_P:
                v[:, si, :] = 0.0
        # (c) time-crop ≥ half the clip, re-resampled to T_OUT by index interpolation
        length = int(round(rng.uniform(TCROP_MIN, 1.0) * T_OUT))
        length = max(2, min(T_OUT, length))
        start = int(rng.integers(0, T_OUT - length + 1))
        flat = v.reshape(T_OUT, N_FEATS)[start:start + length]
        return resample_site(flat) if length < T_OUT else flat

    def __getitem__(self, i: int) -> dict:
        x = self.X[i]
        if self.augment:
            # ClipDataset's seeding lesson, inherited verbatim (skeleton_data.py:252–256, trap 6): the spark
            # comes from torch's per-worker per-epoch RNG so a clip's augmentation varies across epochs.
            spark = int(torch.randint(0, 2**31 - 1, (1,)).item())
            x = self._augmented(x, self.present[i], np.random.default_rng((self.seed, i, spark)))
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            "y": int(self.y[i]),
            "clip_id": self.clip_ids[i],
            "user": int(self.users[i]),
            "n_frames": T_OUT,
            "present": True,
        }
