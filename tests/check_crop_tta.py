"""§105 (D34; P-A) — THE CROP-COUNT STAMP'S BITWISE TEST (the m286 ② shape), CPU only (CUDA hidden), the shipped 022 container on
three REAL test clips through the real decode path (predict.Ensemble._frames → _probs, from the original files — R8), no cache.
  r0   checkpoints/model.pth as shipped (no crops stamp)            ── the reference
  r0b  r0 again                                                     ── determinism (bitwise; no RNG on the inference path, T-L7)
  r1   the same members with crops=1 stamped EXPLICITLY             ── must equal r0 BITWISE (the flag's default is inert)
  r3   the same members with crops=3 stamped                        ── must DIFFER from r0 (the mutation: the stamp reaches the decode path)
  r3b  r3 again                                                     ── determinism of the six-view path (bitwise)
  r2   crops=2                                                      ── must be REFUSED at load
Usage:  CUDA_VISIBLE_DEVICES="" ./.venv/bin/python tests/check_crop_tta.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict import Ensemble  # noqa: E402

CONTAINER = ROOT / "checkpoints" / "model.pth"
TEST = Path(os.environ.get("CUHKX_TEST_ROOT", str(Path.home() / "cuhk-x" / "test_extracted"))) / "small_model_track_test"
PASS: list[str] = []
FAIL: list[str] = []


def check(desc: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(desc)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {desc}{(' — ' + detail) if detail else ''}")


def run(container: Path, clips: list[Path]):
    e = Ensemble([container], device_pref="cpu")
    out = []
    for c in clips:
        probs = {}
        for br in e.by_branch:
            fr = e._frames(c, br)
            if fr is None or not len(fr):
                continue
            probs[br] = np.asarray(e._probs(fr, br), dtype=np.float64)
        out.append((e.predict(c), probs))
    return out


def same(a, b) -> bool:
    return all(pa == pb and set(qa) == set(qb) and all(np.array_equal(qa[k], qb[k]) for k in qa) for (pa, qa), (pb, qb) in zip(a, b))


def stamped(crops: int, td: Path) -> Path:
    mems = torch.load(CONTAINER, map_location="cpu", weights_only=False)
    for m in mems:
        m["crops"] = crops
    p = td / f"model_crops{crops}.pt"; torch.save(mems, p); return p


def main() -> int:
    torch.set_num_threads(6)
    clips = sorted(p for p in TEST.iterdir() if p.is_dir() and p.name.startswith("SM_test_"))[:3]   # the organisers' clip dirs only (a stray dot-dir sorted first on 09-06)
    print(f"§105 P-A · crops stamp bitwise test — {CONTAINER.name} on {[c.name for c in clips]} (CPU)\n")
    r0 = run(CONTAINER, clips); r0b = run(CONTAINER, clips)
    check("r0 == r0b: the shipped path is deterministic (bitwise)", same(r0, r0b))
    check("r0 predicts every clip (both branches present)", all(p is not None and len(q) == 2 for p, q in r0), str([p for p, _ in r0]))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        r1 = run(stamped(1, td), clips)
        check("r1 == r0: an EXPLICIT crops=1 stamp is bitwise the unstamped path (P-A: inert default)", same(r1, r0))
        r3 = run(stamped(3, td), clips); r3b = run(stamped(3, td), clips)
        diff = [c.name for c, (_, qa), (_, qb) in zip(clips, r0, r3) if any(not np.array_equal(qa[k], qb[k]) for k in qa)]
        check("r3 != r0: a crops=3 stamp CHANGES the probabilities (the mutation — the stamp reaches the decode path)", len(diff) == len(clips), f"differs on {diff}")
        check("r3 == r3b: the six-view path is deterministic (bitwise)", same(r3, r3b))
        for c, (p0, q0), (p3, q3) in zip(clips, r0, r3):
            print(f"      {c.name}: argmax {p0} → {p3}; max |Δp| " + ", ".join(f"{k} {float(np.abs(q0[k] - q3[k]).max()):.4f}" for k in q0))
        try:
            run(stamped(2, td), clips); check("r2: crops=2 is REFUSED at load", False)
        except ValueError as e:
            check("r2: crops=2 is REFUSED at load", "crops=2" in str(e), str(e)[:90])
    print(f"\n  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:\n  " + "\n  ".join(FAIL) + "\n\nDiagnose these. Do not adjust the expected values to make them pass."); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
