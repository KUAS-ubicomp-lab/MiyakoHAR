"""P0d — efficiency metrics of the SHIPPED inference path (QUEUE §20 → m161).

The /729056 report requirement (inference memory / VRAM, latency). Runs
predict.Ensemble over checkpoints/model.pt exactly as inference.sh → predict.main
does — same row keys (find_row_keys), same per-clip predict_clip — and records
wall, per-clip wall (decode + inference), per-clip GPU compute (cuda-synchronised
around _probs), peak VRAM allocated AND reserved, peak RSS, container bytes,
params, versions.

CONTROL: the CSV this run writes must sha256-equal the SHIPPED submission of
the slot under test (`--control-csv`; 018's on m161, 022's on the P0 (c) re-measure
of 2026-08-31). If it does not, these numbers are not the shipped path's and are
NOT recorded.

Pre-registration: QUEUE §20 (208c7a3, BEFORE any arithmetic); GOAL-SPRINT-3 §4 P0 (c)
(the re-measure on 022 — m161's method unchanged, the control CSV parameterised).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch                                   # noqa: E402
from src import predict as P                   # noqa: E402

# 018's CSV sha (m152) was the hard-coded control until 2026-08-31; the control is
# now the sha of the CSV passed as --control-csv, so the tool cannot silently
# compare a new slot against an old submission.
SHA_018 = "83cd7ea96d8cd3ec61451d3e0606f3d753f8997f16091f045035015d9248aa8f"   # 018, m152 (reference)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "p0d_efficiency")
    ap.add_argument("--control-csv", type=Path, required=True,
                    help="the SHIPPED submission CSV of the slot under test; the run's CSV must sha256-equal it")
    args = ap.parse_args()
    expected_sha = hashlib.sha256(args.control_csv.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "gpu" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    data_dir = args.data_dir.expanduser().resolve()
    row_keys, prov = P.find_row_keys(data_dir, None)
    print(f"[p0d] {prov} · {len(row_keys)} rows · device pref {args.device}")

    if tag == "gpu":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ens = P.Ensemble([P.SHIPPED_CONTAINER], args.device)
    load_s = time.perf_counter() - t0
    print(f"[p0d] loaded {ens.n} members over {sorted(ens.by_branch)} on {ens.device} in {load_s:.2f}s")

    gpu_calls: list[float] = []
    orig = ens._probs

    def timed(frames, branch):
        if tag == "gpu":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig(frames, branch)
        if tag == "gpu":
            torch.cuda.synchronize()
        gpu_calls.append(time.perf_counter() - t)
        return out
    ens._probs = timed

    per_clip_wall, per_clip_gpu, preds = [], [], []
    t_run = time.perf_counter()
    for key in row_keys:
        cd = P.clip_dir_for(data_dir, key)
        n0 = len(gpu_calls)
        t = time.perf_counter()
        a = P.predict_clip(cd, ens)
        per_clip_wall.append(time.perf_counter() - t)
        per_clip_gpu.append(sum(gpu_calls[n0:]))
        preds.append((key, a))
    run_s = time.perf_counter() - t_run

    csv_path = args.out / f"submission_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "prediction"])
        w.writerows(preds)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    control = sha == expected_sha
    print(f"[p0d] CSV sha256 {sha[:16]}… vs {args.control_csv.name} {expected_sha[:16]}… → {'CONTROL PASS' if control else 'CONTROL FAIL'}")
    if not control:
        print(f"[p0d] 🔴 the produced CSV is not {args.control_csv.name} — these timings are not the shipped path's. NOT RECORDED.")
        return 1

    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        drv = "n/a"
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    params = sum(sum(p.numel() for p in mem["model"].parameters()) for v in ens.by_branch.values() for mem in v)
    m = {
        "device": tag, "torch": torch.__version__, "cuda": torch.version.cuda, "nvidia": drv,
        "cpu_threads": torch.get_num_threads(),
        "container_bytes": P.SHIPPED_CONTAINER.stat().st_size, "members": ens.n, "params": params,
        "fp16_mb_estimate": round(ens.fp16_mb, 2),
        "load_s": round(load_s, 3), "run_s": round(run_s, 2), "clips": len(row_keys),
        "clip_wall_s": {"mean": round(statistics.mean(per_clip_wall), 4), "median": round(q(per_clip_wall, .5), 4),
                        "p90": round(q(per_clip_wall, .9), 4), "max": round(max(per_clip_wall), 4)},
        "clip_compute_s": {"mean": round(statistics.mean(per_clip_gpu), 4), "median": round(q(per_clip_gpu, .5), 4),
                           "p90": round(q(per_clip_gpu, .9), 4), "max": round(max(per_clip_gpu), 4),
                           "share_of_wall": round(sum(per_clip_gpu) / sum(per_clip_wall), 3)},
        "branch_calls": len(gpu_calls),
        "peak_rss_gb": round(rss_gb, 3),
        "csv_sha256": sha, "control_csv": str(args.control_csv), "control_pass": control,
        "container": P.SHIPPED_CONTAINER.name,
    }
    if tag == "gpu":
        m["peak_vram_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        m["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        m["peak_vram_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        m["peak_vram_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    (args.out / f"metrics_{tag}.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))
    print(f"[p0d] wrote {args.out}/metrics_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
