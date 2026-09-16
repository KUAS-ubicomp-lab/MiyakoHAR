"""SWAD — dense weight averaging over an OVERFIT-AWARE window (QUEUE §24, T1).

Cha et al., "SWAD: Domain Generalization by Seeking Flat Minima", NeurIPS 2021
(arXiv 2102.08604). The reference implementation's LossValley rule — n_converge 3,
n_tolerance 6, tolerance_ratio 0.3 — applied here at EPOCH granularity to the
per-epoch validation loss (our epochs are ~121 steps; DomainBed evaluates every
100-300 steps over ~5k, a comparable number of points). The reference's queue
back-fill quirk is omitted (§24 states it).

WHY A SEPARATE MODULE. tests/check_ckpt_meta.py §5 counts train.py's `.save(`
call sites and requires exactly two (best.pt and swa.pt), so that a checkpoint can
never be written without the self-describing keys. The per-epoch parameter SUMS
are therefore written from HERE, and the SWAD checkpoint goes through train.py's
existing swa.pt site with `selection: "swad"` -- no third site, nothing unstamped.

WHAT IS AVERAGED. Parameters only, per ITERATION, summed on the GPU in fp32 (the
master weights under bf16 autocast are fp32) and flushed once per epoch as one
file. BN buffers are NOT averaged: the live one is refit by train.recompute_bn_stats
(trap 7), the frozen ones never move. The window average over any [t_s, t_e] is
then exact arithmetic on those files -- which is what lets tools/swad_finalize.py
build the LOFO-transferred window (§24's PRIMARY arm) from the same run.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

N_CONVERGE = 3
N_TOLERANCE = 6
TOLERANCE_RATIO = 0.3


def swad_window(losses, n_converge: int = N_CONVERGE, n_tolerance: int = N_TOLERANCE,
                ratio: float = TOLERANCE_RATIO) -> tuple[int, int, bool, float | None]:
    """(t_s, t_e, converged, threshold) over epoch indices, both ends INCLUSIVE.

    t_s = the first epoch i whose loss is the minimum of losses[i : i+n_converge]
          (the valley has been entered: nothing in the next n_converge-1 epochs
          beats it). threshold = mean of that window × (1 + ratio).
    t_e = the epoch before the first n_tolerance-epoch window at or after t_s
          whose MINIMUM exceeds the threshold (the valley is dead), else the last.
    Never converged (loss still falling at the end): the last n_converge epochs,
    flagged converged=False -- the flat-tail fallback, never a silent default.
    """
    losses = [float(v) for v in losses]
    L = len(losses)
    if L == 0:
        raise ValueError("swad_window needs at least one loss")
    if L < n_converge:
        return 0, L - 1, False, None
    ts, thr = None, None
    for i in range(L - n_converge + 1):
        w = losses[i:i + n_converge]
        if w.index(min(w)) == 0:
            ts, thr = i, sum(w) / len(w) * (1.0 + ratio)
            break
    if ts is None:
        return L - n_converge, L - 1, False, None
    te = L - 1
    for j in range(ts, L - n_tolerance + 1):
        if min(losses[j:j + n_tolerance]) > thr:
            te = j - 1
            break
    return ts, max(ts, te), True, thr


class EpochSums:
    """Per-iteration parameter sums, flushed to one fp32 file per epoch."""

    def __init__(self, model: nn.Module, out_dir: Path) -> None:
        self.params = [p for _, p in model.named_parameters()]
        self.names = [n for n, _ in model.named_parameters()]
        self.sums = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.n = 0
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flushed: list[tuple[int, int]] = []   # (epoch, n_iterations)

    @torch.no_grad()
    def add(self) -> None:
        torch._foreach_add_(self.sums, self.params)
        self.n += 1

    @torch.no_grad()
    def flush(self, epoch: int) -> Path:
        path = self.out_dir / f"e{epoch:02d}.pt"
        torch.save({"epoch": epoch, "n": self.n, "names": self.names,
                    "sums": [s.detach().cpu() for s in self.sums]}, path)
        self.flushed.append((epoch, self.n))
        torch._foreach_zero_(self.sums)
        self.n = 0
        return path


@torch.no_grad()
def window_average(sums_dir: Path, ts: int, te: int) -> tuple[dict[str, torch.Tensor], int]:
    """Mean of every iterate in epochs ts..te (inclusive), from the flushed sums."""
    total, names, n = None, None, 0
    for e in range(ts, te + 1):
        d = torch.load(Path(sums_dir) / f"e{e:02d}.pt", map_location="cpu", weights_only=False)
        if total is None:
            total, names = [s.double() for s in d["sums"]], d["names"]
        else:
            torch._foreach_add_(total, [s.double() for s in d["sums"]])
        n += int(d["n"])
    if not n:
        raise ValueError(f"window {ts}..{te} holds zero iterations under {sums_dir}")
    return {k: (t / n).float() for k, t in zip(names, total)}, n


@torch.no_grad()
def load_average(model: nn.Module, avg: dict[str, torch.Tensor]) -> None:
    """Copy the averaged parameters into `model` (buffers untouched; refit BN after)."""
    sd = dict(model.named_parameters())
    missing = [k for k in avg if k not in sd]
    extra = [k for k in sd if k not in avg]
    if missing or extra:
        raise KeyError(f"average/model parameter mismatch: missing {missing[:3]}, extra {extra[:3]}")
    for k, v in avg.items():
        sd[k].copy_(v.to(sd[k].device, dtype=sd[k].dtype))
