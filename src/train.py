"""Train one branch on one fold. The day-1 recipe from configs/baseline.yaml.

Every value here traces to DECISIONS.md §8. Change ONE at a time and log it;
the CV noise floor is ~1.0-1.5 pp paired, so an unlogged second change makes the
first uninterpretable.

TRAPS THIS FILE EXISTS TO HANDLE.

1. bfloat16, NEVER fp16. bf16 has fp32's exponent range, so there is no loss
   scaling and no GradScaler -- which deletes the entire overflow/underflow
   class of bug. fp16 here would need a scaler and would silently NaN without
   one. Both our GPUs are native bf16 (sm_86 and sm_120).

2. NO WEIGHT DECAY ON BN OR BIAS. Decaying BN affine parameters pulls gamma
   toward zero and quietly throttles the network. It is a two-line omission
   that costs real accuracy and never raises.

3. WARMUP IS ON ITERATIONS, NOT EPOCHS. Our epochs are ~126 steps, so a
   3-epoch warmup is ~380 iterations. Stepping the scheduler per epoch instead
   gives 3 steps of warmup, which is no warmup at all.

4. THE VALIDATION SET IS SUBJECT-DISJOINT AND MUST STAY THAT WAY. dataset.py
   enforces it from splits/folds.yaml and check_dataset.py asserts it; this file
   must never re-derive a split of its own.

5. SANITY FLOOR 0.10945. Majority class is 10.9%, uniform chance 2.5%. A run
   that finishes under ~0.11 has a bug, not a weak signal -- stop and find it
   rather than tuning.

6. SWA IS THE CHECKPOINT SELECTOR, NOT argmax OVER VALIDATION. The recipe says
   checkpoint_selection: swa_weights_at_end_of_schedule, and it says so because
   picking the best of 40 noisy validation evaluations fits the validation set:
   the reported number is then the maximum of 40 draws, which is biased high by
   construction and does not survive to the leaderboard. best.pt is still
   written, purely so every run measures that bias for free.

7. torch.optim.swa_utils.update_bn CANNOT BE USED HERE. It resets EVERY
   BatchNorm and only then calls model.train() -- at which point partial_bn
   returns 19 of the 20 to eval mode, so they never recompute and keep the
   (0, 1) the reset left. That deletes the ImageNet statistics partial_bn
   exists to preserve, silently, in a model that still runs. recompute_bn_stats
   below recomputes exactly the BN layers training itself updates.
   tests/check_swa.py runs the stock function as a control arm and asserts it
   does the damage, so the trap is demonstrated rather than described.

Usage -- as a MODULE, from the repo root. `python src/train.py` puts src/ on
sys.path instead of the root and dies on `from src.dataset import ...`:
    python -m src.train --branch thermal --fold 0 --epochs 40
    python -m src.train --branch thermal --fold 0 --epochs 2 --limit-batches 20   # smoke
    python -m src.train --branch thermal --fold 0 --no-swa                        # A/B control arm
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import resource
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from src.dataset import GEO_T, ORD_BRANCHES, ClipDataset, collate      # GEO_T: §95 (iii); ORD_BRANCHES: §96 (a)/(b)
from src.model import DEFAULT_ARCH, NON_IMAGE_ARCHS, NUM_CLASSES, RANK5_ARCHS, build, norm_key, param_count
from src.pk_sampler import PKSampler, supcon_loss
from src.swad import EpochSums, load_average, swad_window, window_average

ROOT = Path(__file__).resolve().parent.parent

# RELAY.md §6. `machine` is mandatory on every experiment row, and the two
# ledgers are SEPARATE FILES, for two reasons that both bite silently:
# one shared file conflicts in git on every append, and a comparison whose two
# rows differ in machine is not a comparison -- different CUDA build, different
# SM architecture, different cuDNN kernel selection, against a CV noise floor of
# ~1.0-1.5 pp paired. Run 0 lives on ALIEN, so a hardcoded "ZEPH" here would
# file ALIEN's baseline under ZEPH's name and nothing would ever raise.
# CVL-UBUNTU3090 registered 2026-08-24 (GOAL-SPRINT-2 §3.7, the m147 act):
# hostname read off the box, not guessed. Single-writer status transferred to
# LABPC at its first m-row (D14(c), RELAY-LOG A#011).
HOSTS = {"LAPTOP-EPM7HC10": "ZEPH", "JAK5505": "ALIEN", "CVL-UBUNTU3090": "LABPC"}


def resolve_machine(explicit: str | None = None) -> str:
    """Name this box, or refuse to run. Never guess -- see HOSTS above."""
    name = explicit or os.environ.get("CUHKX_MACHINE") or HOSTS.get(socket.gethostname().upper())
    if not name:
        raise SystemExit(
            f"[train] REFUSING TO RUN: cannot tell which machine this is.\n"
            f"[train]   hostname {socket.gethostname()!r} is not in HOSTS.\n"
            f"[train] Pass --machine ALIEN, or export CUHKX_MACHINE=ALIEN, and add the\n"
            f"[train] hostname to HOSTS in src/train.py so the next run needs neither.\n"
            f"[train] Guessing would file these rows under the wrong machine, which\n"
            f"[train] RELAY.md §6 says is not a comparison at all."
        )
    return name.upper()


def param_groups(model: nn.Module, weight_decay: float):
    """Trap 2: BN weights, BN biases and all biases are excluded from decay."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def apply_head_init(model: nn.Module, path: Path) -> dict:
    """§79 (D26): LP-FT — the 40-way head starts from tools/lpft_probe.py's probe (raw weights, no rescaling)."""
    hi = torch.load(path, map_location="cpu", weights_only=False)
    fc = getattr(model, "fc", None)
    if fc is None:
        raise SystemExit("--head-init needs an arch with a `fc` head (the ir-CSN family) (§79)")
    w, b = hi["weight"], hi["bias"]
    if tuple(w.shape) != tuple(fc.weight.shape) or tuple(b.shape) != tuple(fc.bias.shape):
        raise SystemExit(f"--head-init {path}: weight {tuple(w.shape)} / bias {tuple(b.shape)} do not fit fc "
                         f"{tuple(fc.weight.shape)} / {tuple(fc.bias.shape)} (§79)")
    with torch.no_grad():
        fc.weight.copy_(w)
        fc.bias.copy_(b)
    print(f"[train] §79 head-init from {path}: fold {hi.get('fold')} · λ {hi.get('lambda')} · LOTSO {hi.get('lotso_acc')} · "
          f"probe val {hi.get('val_acc')} — fc loaded {tuple(w.shape)}")
    return hi


def l2sp_groups(model: nn.Module, opt_params: list, weight_decay: float) -> tuple[list, list]:
    """§80 (D26): L2-SP — the backbone's decay set is anchored to its current (loaded) values: its optimiser decay goes
    to 0 and α·(p − p₀) is added to the gradient after the clip, before the step (l2sp_add); fc keeps `weight_decay`
    toward 0; the no-decay group (BN, biases — Trap 2) is untouched. Returns (groups, anchors[(p, p0)])."""
    fc = getattr(model, "fc", None)
    if fc is None or len(opt_params) != 2:
        raise SystemExit("--l2sp is wired for the ir-CSN recipe's two-group optimiser (§80)")
    head_ids = {id(p) for p in fc.parameters()}
    decay, no_decay = opt_params[0]["params"], opt_params[1]["params"]
    back = [p for p in decay if id(p) not in head_ids]
    fc_decay = [p for p in decay if id(p) in head_ids]
    anchors = [(p, p.detach().clone()) for p in back]
    groups = [{"params": back, "weight_decay": 0.0}, {"params": fc_decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return groups, anchors


def l2sp_add(anchors: list, alpha: float) -> None:
    """§80: grad += α·(p − p₀) on every anchored tensor — exactly zero at the loaded weights."""
    with torch.no_grad():
        for p, p0 in anchors:
            if p.grad is not None:
                p.grad.add_(p - p0, alpha=alpha)


def load_dup_map(path: Path, fold: int) -> dict:
    """§81 (D26): tools/dup_map.py's per-fold map; a map from another fold is refused."""
    dm = json.loads(Path(path).read_text())
    if int(dm.get("fold", -99)) != fold:
        raise SystemExit(f"{path} is fold {dm.get('fold')}, this run is fold {fold} — refusing (§81)")
    return dm


def drop_dup_exact(ds, dm: dict) -> int:
    """§81 (a): the map's `drop` clips leave THIS dataset's items (the caller passes the train set only)."""
    drop = set(dm["drop"])
    before = len(ds.items)
    ds.items = [r for r in ds.items if r.clip_id not in drop]
    return before - len(ds.items)


def dup_soft_targets(dm: dict, eps: float, num_classes: int = 40) -> dict:
    """§81 (b): clip_id → the LS-smoothed soft target q′ = (1 − eps)·q + eps/C with q ∝ {y: 1, k: share_k}."""
    out = {}
    for cid, e in dm["share"].items():
        q = np.zeros(num_classes, dtype=np.float32)
        q[int(e["y"])] = 1.0
        for k, v in e["share"].items():
            q[int(k)] += float(v)
        q /= q.sum()
        out[cid] = ((1.0 - eps) * q + eps / num_classes).astype(np.float32)
    return out


def dup_loss(out: torch.Tensor, y: torch.Tensor, clip_ids: list, targets: dict, eps: float):
    """§81 (b): the batch loss when ≥ 1 clip is mapped — mapped clips take −Σ q′·log p, the others the criterion's
    per-clip term (cross-entropy with LS eps); the sum is divided by the batch size as the criterion's mean is.
    Returns None when no clip in the batch is mapped, so the caller keeps the recipe's loss bitwise."""
    mapped = [j for j, c in enumerate(clip_ids) if c in targets]
    if not mapped:
        return None
    ms = set(mapped)
    keep = [j for j in range(out.shape[0]) if j not in ms]
    q = torch.from_numpy(np.stack([targets[clip_ids[j]] for j in mapped])).to(out.device)
    soft = -(q * F.log_softmax(out[mapped].float(), dim=1)).sum(1).sum()
    hard = (F.cross_entropy(out[keep].float(), y[keep], label_smoothing=eps, reduction="sum") if keep
            else soft.new_zeros(()))
    return (hard + soft) / out.shape[0]


class OnlineLS:
    """§83 (D26): online label smoothing (Zhang et al., IEEE TIP 2021) — target = (1 − α)·one-hot + α·S[y], where the C × C
    matrix S starts uniform and is REPLACED after every epoch by the per-class mean softmax over that epoch's CORRECTLY
    classified train clips (a row with no correct clip keeps its previous row); every row sums to 1. Replaces the
    criterion's label smoothing entirely (the hard part is the plain one-hot)."""

    def __init__(self, num_classes: int, alpha: float, device) -> None:
        self.C, self.alpha = num_classes, alpha
        self.S = torch.full((num_classes, num_classes), 1.0 / num_classes, device=device)
        self.acc = torch.zeros_like(self.S)
        self.cnt = torch.zeros(num_classes, device=device)

    def target(self, y: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.alpha) * F.one_hot(y, self.C).float() + self.alpha * self.S[y]

    def loss(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(out.float(), dim=1)
        with torch.no_grad():
            p = logp.exp()
            correct = p.argmax(1) == y
            if bool(correct.any()):
                self.acc.index_add_(0, y[correct], p[correct])
                self.cnt.index_add_(0, y[correct], torch.ones(int(correct.sum()), device=y.device))
        return -(self.target(y) * logp).sum(1).mean()

    def end_epoch(self) -> int:
        has = self.cnt > 0
        self.S[has] = self.acc[has] / self.cnt[has, None]
        self.acc.zero_()
        self.cnt.zero_()
        return int(has.sum())


def lr_at(it: int, total: int, warmup: int, base_lr: float, start_factor: float) -> float:
    """Linear warmup then cosine to zero. Trap 3: `it` is an ITERATION index."""
    if it < warmup:
        f = start_factor + (1.0 - start_factor) * (it / max(1, warmup))
        return base_lr * f
    prog = (it - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def swa_start_iter(total_iters: int, start_frac: float) -> int:
    """First iteration whose post-step weights enter the average.

    `start_frac` is a fraction of the WHOLE schedule, so 0.75 averages the last
    quarter -- the iterates the cosine has already annealed. Averaging earlier
    ones drags the mean back toward weights the schedule deliberately left.
    Snapshot count is exactly `total_iters - swa_start_iter(...)`.
    """
    if not 0.0 <= start_frac <= 1.0:
        raise ValueError(f"start_frac must be in [0, 1], got {start_frac}")
    return int(round(total_iters * start_frac))


@torch.no_grad()
def recompute_bn_stats(model, loader, device, amp_dtype, limit=None) -> int:
    """SWA's mandatory step: fit BN running stats to the AVERAGED weights.

    Necessary because AveragedModel(use_buffers=False) copies buffers straight
    from the live model on every update, so the averaged weights arrive paired
    with the FINAL iterate's BN statistics -- weights and stats from different
    models.

    Trap 7: this cannot delegate to torch.optim.swa_utils.update_bn, which
    resets every BatchNorm before calling model.train() and so wipes the 19
    partial_bn keeps frozen. Recompute exactly what training updates, which is
    what .train() reports once partial_bn has been applied.
    """
    model.train()  # re-applies partial_bn -- MUST precede the selection below
    live = [m for m in model.modules()
            if isinstance(m, nn.modules.batchnorm._BatchNorm)
            and m.training and m.track_running_stats]
    if not live:
        return 0
    momenta = {m: m.momentum for m in live}
    for m in live:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the pass, not an EMA
    seen = 0
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        x = batch["x"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            model(x)
        seen += 1
    for m, mom in momenta.items():
        m.momentum = mom
    return seen


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, limit=None):
    model.eval()
    correct = total = 0
    loss_sum = 0.0   # plain CE, no smoothing -- SWAD's window signal (QUEUE §24); logged for every run
    per_class_hit = torch.zeros(40, dtype=torch.long)
    per_class_n = torch.zeros(40, dtype=torch.long)
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        x = batch["x"].to(device, non_blocking=True)  # model applies channels_last after reshape
        y = batch["y"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(x)
        pred = logits.argmax(dim=1)
        loss_sum += float(F.cross_entropy(logits.float(), y, reduction="sum"))
        hit = pred == y
        correct += int(hit.sum())
        total += y.numel()
        for c, h in zip(y.cpu().tolist(), hit.cpu().tolist()):
            per_class_n[c] += 1
            per_class_hit[c] += int(h)
    acc = correct / max(1, total)
    seen = per_class_n > 0
    balanced = float((per_class_hit[seen].float() / per_class_n[seen].float()).mean()) if seen.any() else 0.0
    return acc, balanced, total, int(seen.sum()), loss_sum / max(1, total)


# Dests that deliberately do NOT appear on an experiment row: they change where
# the work reads and writes, never what is trained. Everything else must be on the
# row, either as a top-level field or inside hparams -- check_ledger.py asserts the
# partition is exhaustive, so adding an A/B knob and forgetting to log it FAILS a
# gate instead of producing rows that only their run-name distinguishes.
PLUMBING = {"workers", "cache", "manifest", "run_name", "machine", "out", "limit_batches"}


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Extracted so a test can assert every knob reaches the ledger row."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="thermal",
                choices=["thermal", "depthir", "depthir_crop", "depthir_cropsq", "depthir_ord3", "depthir_ordgrad", "skeleton", "imu"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    # §68 arm (b): VideoMAE V2's own fine-tune optimiser — AdamW + layer-wise lr decay. sgd = the record's recipe.
    ap.add_argument("--optim", default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--llrd", type=float, default=None,
                    help="layer-wise lr decay (adamw, archs exposing layer_id()): group lr = lr × llrd^(n_layers−1−layer)")
    # §71 (D24): freeze the stem + layer1..layer_n of an ir-CSN (params and BN); stamped into the checkpoint.
    ap.add_argument("--freeze-stages", type=int, default=0, choices=[0, 1, 2, 3, 4])
    # §72 (D24): TrivialAugment-Wide on top of the recipe's augmentation block (clip-consistent, one op per clip).
    ap.add_argument("--aug-trivial", action="store_true")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="§66 control (a), imu branch only: permute the TRAIN labels once (fixed seed); "
                         "the val labels stay real. The member this trains is the fused dilution floor.")
    ap.add_argument("--clip-grad-norm", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=20260813)
    # SWA is ON by default: it is IN the baseline recipe, not an A/B arm.
    ap.add_argument("--no-swa", dest="swa", action="store_false", default=True,
                    help="disable SWA and select by best epoch (the control arm)")
    ap.add_argument("--swa-start-frac", type=float, default=0.75)
    # T1 (QUEUE §24): SWAD replaces SWA's SELECTOR, not its averaging -- constant lr after
    # warmup, per-iteration sums per epoch (src/swad.py), and the window chosen by the
    # loss-valley rule on the per-epoch val loss. Requires SWA on; --swad-window s,e forces
    # the window (the val-less all-18 transfer path, and tools/swad_finalize.py's arithmetic).
    ap.add_argument("--swad", action="store_true",
                    help="SWAD: constant lr after warmup + overfit-aware averaging window (QUEUE §24)")
    ap.add_argument("--swad-window", default=None, metavar="S,E",
                    help="force SWAD's epoch window (inclusive); required with --all-subjects")
    # T3 (QUEUE §26): 2x widening of the clip-consistent affine augmentation ranges.
    ap.add_argument("--aug-wide", action="store_true",
                    help="widen the per-clip affine ranges 2x (rot 15, translate 15%%, scale 0.7-1.3)")
    # T6 (QUEUE §29): clip-consistent temporal crop (p=0.5, window U(0.5,1.0) of the clip), train only.
    ap.add_argument("--aug-tcrop", action="store_true",
                    help="temporal crop augmentation: sample the T segments over a random sub-window (QUEUE §29)")
    # T7 (QUEUE §30): clip-consistent spatial VideoMix. ALPHA = the Beta(alpha, alpha) parameter; p = 1.0.
    ap.add_argument("--videomix", type=float, default=None, metavar="ALPHA",
                    help="T7: CutMix-class spatial box from another clip in the batch, same box on all T frames")
    # T8 (QUEUE §35): MixStyle after layer1/layer2 of the CSN, train mode only, p 0.5, alpha 0.1.
    ap.add_argument("--mixstyle", action="store_true", help="T8: MixStyle feature-statistics mixing (ircsn_r50 only)")
    # §63 (D22 d): model-side thermal input adapters on the CSN branch; stamped into the checkpoint.
    ap.add_argument("--input-adapt", choices=["clipnorm", "motion", "irnorm", "valid"], default=None,
                    help="§63: clipnorm | motion (ircsn only); §96 (d): irnorm — the IR plane [p1, p99] -> [0, 1] per clip; "
                         "§96 (c): valid — the depth-validity plane as a fifth channel (both depthir ircsn only)")
    # A5 / T9 (QUEUE §36): subject-crossing box transplant (depthir only, train mode, p 0.5).
    ap.add_argument("--aug-transplant", action="store_true",
                    help="T9: paste the clip's tight-box subject region onto another training clip's frames")
    # T4 (QUEUE §27): skeleton as a TRAINING-TIME-ONLY privileged target. An auxiliary head built
    # OUTSIDE the model (J8's fence -- it can never reach a checkpoint) regresses a 153-d z-scored
    # pose summary from the pooled backbone feature; nothing pose-related ships. Pose CONTENT only
    # (D14(d)); targets from cache/skeleton_train.npz; absent-pose clips are masked.
    ap.add_argument("--pose-aux", type=float, default=None, metavar="ALPHA",
                    help="T4: weight of the auxiliary pose-regression loss (ircsn_r50 image branches only)")
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m.csv"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--machine", default=None,
                    help="ZEPH|ALIEN — derived from hostname; refuses to guess (RELAY.md §6)")
    # cudnn autotuning is the measured cause of run-to-run drift at a FIXED seed
    # (A#001: 2.00 pp between two identical commands). Default stays True so the
    # baseline is unchanged; this flag exists to measure what turning it off buys.
    ap.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false",
                    default=True, help="disable cudnn autotuning — slower, but removes "
                                       "kernel selection as a source of run-to-run variance")
    ap.add_argument("--limit-batches", type=int, default=0, help="smoke test: cap steps per epoch")
    # A/B #2's only knob. TSM is IN the baseline, so ON is the default and OFF is
    # the control arm -- a plain TSN. model.py trap 1: a dead TSM and a disabled
    # one are indistinguishable from the loss curve, so check_model.py asserts the
    # two differ (order-asymmetric with, exactly order-invariant without) and
    # check_ledger.py asserts this flag reaches build() rather than stopping here.
    ap.add_argument("--no-tsm", dest="tsm", action="store_false", default=True,
                    help="disable the temporal shift — the TSN control arm of A/B #2")
    ap.add_argument("--no-pretrained", action="store_true")
    # partial_bn IS A FINE-TUNING TECHNIQUE AND IT MUST BE TURNED OFF FOR A
    # FROM-SCRATCH ARM. It freezes every BatchNorm but the first, in eval mode,
    # so the layer normalises by its stored running statistics forever. On a
    # PRETRAINED net those are the Kinetics/ImageNet statistics and preserving
    # them is the entire point. On a RANDOM-INIT net they are the (mean 0,
    # var 1) initialisation and can never update — measured: 76 of s3d's 76
    # frozen BNs hold the trivial init on the scratch arm against 0 of 76 on the
    # pretrained one. The network then has no working normalisation and 76 dead
    # affine layers, and it does not train: J1's first s3d-SCRATCH cells sat at
    # train_loss 3.68 -> 3.52 over 40 epochs against ln(40) = 3.689, i.e. never
    # meaningfully below chance ON TRAINING DATA, and scored the majority class.
    # "Identical recipe" is the right instinct and the wrong control here: the
    # flag means "keep the pretrained statistics", and there are none to keep.
    ap.add_argument("--no-partial-bn", action="store_true",
                    help="train every BatchNorm. REQUIRED for a from-scratch arm — "
                         "partial_bn freezes BN at its random init and the run cannot learn")
    # J1's backbone factor. Recorded on the row AND stamped into the checkpoint,
    # because a run name is not an identifier (see hparams_of's docstring) and a
    # checkpoint that does not say what it is cannot be loaded correctly.
    ap.add_argument("--arch", default=DEFAULT_ARCH,
                    help="backbone architecture; must be a key of model._ARCH_BUILDERS")
    # J1's second factor. F.interpolate inside the dataset — no cache rebuild,
    # no new shard. Default None = the cache's native 120x160, bit-identical to
    # every run in the ledger.
    ap.add_argument("--input-size", default=None, metavar="HxW",
                    help="resize clips to HxW (e.g. 168x224). Default: the cache's native size")
    ap.add_argument("--train-subjects", type=int, default=None, metavar="N",
                    help="J5 (m86): train on only N of the fold's training subjects, "
                         "balanced across recording blocks. VALIDATION IS UNTOUCHED.")
    # J8/8b (m97/m102). NOT classic distillation: the teacher scores 33.6% on
    # our task against the student's 58.6% (m102), so pulling the 40-way head
    # toward it would HURT. The teacher gets its OWN 400-way auxiliary head on the
    # shared trunk; the 40-way head never sees a teacher gradient directly. This is
    # auxiliary-task regularisation, and m87's data-limited regime is where such a
    # term earns its keep.
    ap.add_argument("--kd-cache", type=Path, default=None, metavar="NPZ",
                    help="J8 (m101): offline teacher logits. Enables the auxiliary "
                         "400-way KD head. The head is NEVER saved — no teacher-derived "
                         "parameter ships (D1/R3, R9).")
    ap.add_argument("--kd-alpha", type=float, default=1.0,
                    help="weight on the auxiliary KD term")
    ap.add_argument("--kd-temp", type=float, default=4.0,
                    help="KD softmax temperature")
    # §75 (D25): cross-modal distillation on the model's OWN 40-way logits — teacher logits per clip_id from
    # tools/kd40_cache_teacher.py; loss = (1 − kd_alpha)·CE + kd_alpha·kd_temp²·KL where a target exists, CE alone otherwise.
    ap.add_argument("--kd40-cache", type=Path, default=None, metavar="NPZ")
    # §76 (D25): the head dropout as a declared scalar (CSNBranch's `dropout`; the recipe's 0.5 when absent).
    ap.add_argument("--head-dropout", type=float, default=None)
    # §77 (D25): start the recipe from an adapted backbone (tools/ssl_adapt.py's adapted_init.pt) instead of the
    # K400 weights; the file's full CSNBranch state dict is loaded strict after build; provenance stamped.
    ap.add_argument("--init-from", type=Path, default=None, metavar="PT")
    # §79–§82 (D26), the fine-tuning-geometry arms — every flag absent ⇒ the recipe's code path untouched.
    # --head-init: fc from tools/lpft_probe.py's head_init.pt (LP-FT); --l2sp: decay toward the loaded weights on the
    # backbone's decay set (L2-SP); --drop-dup-exact / --dup-soft-targets: tools/dup_map.py's per-fold map of the
    # corpus's cross-class duplicate clips, TRAIN side only; --csn-weights: another file in CSN_WEIGHT_DIR as the
    # pretrained start (the IG-65M-only init).
    ap.add_argument("--head-init", type=Path, default=None, metavar="PT")
    ap.add_argument("--l2sp", type=float, default=None, metavar="ALPHA")
    ap.add_argument("--drop-dup-exact", type=Path, default=None, metavar="JSON")
    ap.add_argument("--dup-soft-targets", type=Path, default=None, metavar="JSON")
    ap.add_argument("--csn-weights", default=None, metavar="FILE")
    # §83 (D26): online label smoothing — the hard/soft mix α (the paper's 0.5); replaces --label-smoothing entirely.
    ap.add_argument("--ols", type=float, default=None, metavar="ALPHA")
    # §84 (D26): P×K batches (P classes × K distinct subjects; P·K = the batch size) and, on top, the supervised-contrastive
    # term with cross-subject positives (weight λ; τ 0.1; projection 2048 → 512 → 128 discarded at inference).
    ap.add_argument("--pk-sampler", default=None, metavar="PxK")
    ap.add_argument("--supcon", type=float, default=None, metavar="LAMBDA")
    # Self-training, rules-class PSEUDO-LABEL(R7) — the one recorded LEGAL class of
    # test-input-derived technique. m105 prices it from m87's own subject-count
    # slope: the test set holds 4 subjects (10, 11, 25, 26) disjoint from all 18.
    ap.add_argument("--pseudo-labels", type=Path, default=None, metavar="CSV",
                    help="clip_id,pseudo_label,confidence from tools/make_pseudo_labels.py. "
                         "Test clips above --pseudo-conf join the TRAIN set.")
    ap.add_argument("--pseudo-val", action="store_true",
                    help="§58 fold instrument: the pseudo-labelled clips are the fold's VAL clips "
                         "(train split, mode=val) instead of the test shard; labels from the CSV only.")
    ap.add_argument("--pseudo-conf", type=float, default=0.7,
                    help="confidence threshold. Validation calibration (m107): "
                         "0.5 -> 83.8%% pure, 0.7 -> 92.5%%, 0.9 -> 98.4%%")
    # §95 (ii) (D29): the R7 transfer set as SOFT carriers — the pseudo-labelled clips get NO CE term (a per-sample mask) and take
    # their only signal from a second --kd40-cache file (the fold teacher's VALCARRIER targets); their CSV label is unread by the
    # loss. With the flag off every line of the incumbent's path is untouched (tests/check_pseudo_soft.py r5 == r0 bitwise).
    ap.add_argument("--pseudo-soft", action="store_true",
                    help="§95 (ii): the --pseudo-labels clips carry NO CE term; their KD targets come from --kd40-cache-extra")
    ap.add_argument("--kd40-cache-extra", type=Path, default=None, metavar="NPZ",
                    help="§95 (ii): the carriers' teacher targets (the VALCARRIER file, same fold); --pseudo-soft only")
    # §95 (iii) (D29): CONSISTENT GEOMETRIC TEACHING — the train loader replays K fixed geometric draws per clip (dataset.geo_draws) and the
    # KD target is the teacher's output under the SAME draw: --kd40-cache is the {…}_TRAIN_geo{K}.npz file keyed (clip_id, draw_k).
    ap.add_argument("--kd40-geo", type=int, default=None, metavar="K", help="§95 (iii): K replayed geometric draws; --kd40-cache is the geo file")
    ap.add_argument("--all-subjects", action="store_true",
                    help="J9 (m87): train on ALL 18 subjects. THERE IS NO VALIDATION SET in this "
                         "mode and no val number is written anywhere — the checkpoint is selected "
                         "by the SWA schedule, which never reads validation. Requires SWA.")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    return ap


def peak_rss_gb() -> tuple[float, float]:
    """(this process, worst single DataLoader worker) peak RSS in GB.

    HOST RAM IS THE BINDING RESOURCE ON THIS BOX AND NOTHING MEASURED IT.
    On 2026-08-15 six global_oom kills took the VM down five times in 90 minutes
    during A/B #6; the victims were `pt_data_worker` and `python`, on 7.8 GB of
    RAM against a 26 GB memmapped cache. Peak VRAM across that whole period was
    0.999 GB of 7.96 -- the card was never close, and peak_vram_gb was the only
    memory figure on the row. D6/R5 score efficiency on VRAM and latency, so
    nothing in the scoring model would ever have surfaced it either.

    ru_maxrss for RUSAGE_CHILDREN is the high-water mark of the WORST SINGLE
    reaped child, not the sum -- which is the number that matters here, because
    what died was one worker holding twice the frames at T=16. Linux reports KB.
    """
    kb = 1024 * 1024
    return (round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / kb, 3),
            round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / kb, 3))


def hparams_of(args) -> dict:
    """Every knob an A/B arm can move, recorded on the row itself.

    A run-name is not an identifier: two arms differing only in --weight-decay were
    once distinguishable solely by whatever the caller happened to name them, and
    RELAY.md §6 asks the ledger, not the caller, to referee the comparison.
    """
    return {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
            "T": args.T, "epochs": args.epochs, "label_smoothing": args.label_smoothing,
            "clip_grad_norm": args.clip_grad_norm, "warmup_epochs": args.warmup_epochs,
            "swa": args.swa, "swa_start_frac": args.swa_start_frac,
            "swad": args.swad, "swad_window": args.swad_window,
            "aug_wide": args.aug_wide, "pose_aux": args.pose_aux, "aug_tcrop": args.aug_tcrop,
            "videomix": args.videomix, "mixstyle": args.mixstyle, "aug_transplant": args.aug_transplant,
            "input_adapt": args.input_adapt,
            "tsm": args.tsm, "pretrained": not args.no_pretrained,
            "arch": args.arch, "input_size": args.input_size,
            "partial_bn": not args.no_partial_bn,
            "train_subjects": args.train_subjects,
            "all_subjects": args.all_subjects,
            "shuffle_labels": bool(args.shuffle_labels),
            "optim": args.optim, "llrd": args.llrd,
            "freeze_stages": args.freeze_stages, "aug_trivial": bool(args.aug_trivial),
            "kd40_cache": str(args.kd40_cache) if args.kd40_cache else None, "head_dropout": args.head_dropout,
            "pseudo_soft": bool(args.pseudo_soft), "kd40_cache_extra": str(args.kd40_cache_extra) if args.kd40_cache_extra else None,
            "kd40_geo": args.kd40_geo,
            "init_from": str(args.init_from) if args.init_from else None,
            "head_init": str(args.head_init) if args.head_init else None, "l2sp": args.l2sp,            # §79/§80
            "drop_dup_exact": str(args.drop_dup_exact) if args.drop_dup_exact else None,                 # §81 (a)
            "dup_soft_targets": str(args.dup_soft_targets) if args.dup_soft_targets else None,           # §81 (b)
            "csn_weights": args.csn_weights,                                                              # §82
            "ols": args.ols,                                                                              # §83
            "pk_sampler": args.pk_sampler, "supcon": args.supcon,                                            # §84
            "pseudo_labels": str(args.pseudo_labels) if args.pseudo_labels else None,
            "pseudo_conf": args.pseudo_conf if args.pseudo_labels else None,
            "pseudo_val": bool(args.pseudo_val) if args.pseudo_labels else None,
            "kd_cache": str(args.kd_cache) if args.kd_cache else None,
            "kd_alpha": args.kd_alpha if args.kd_cache else None,
            "kd_temp": args.kd_temp if args.kd_cache else None}


def main() -> int:
    args = build_parser().parse_args()

    # J9 (m87): without validation, best.pt cannot be selected, so SWA is the
    # ONLY checkpoint this mode can emit. --no-swa would run for hours and write
    # nothing. Fail in zero seconds instead, the same rule --arch already follows.
    if args.all_subjects and not args.swa:
        raise SystemExit("--all-subjects requires SWA: with no validation set there is no "
                         "best-epoch to select, so --no-swa would write no checkpoint at all.")

    if args.swad and not args.swa:
        raise SystemExit("--swad replaces SWA's selector, not its averaging: it needs SWA on (drop --no-swa).")
    if args.swad and args.all_subjects and not args.swad_window:
        raise SystemExit("--swad --all-subjects has no val loss to pick a window from: pass --swad-window S,E "
                         "(the fold-derived transfer, QUEUE §24).")
    if args.kd40_cache is not None and args.kd_cache is not None:
        raise SystemExit("--kd40-cache and --kd-cache are different distillations; one at a time (§75)")
    if args.pseudo_soft and not (args.pseudo_labels is not None and args.pseudo_val and args.kd40_cache is not None and args.kd40_cache_extra is not None):
        raise SystemExit("--pseudo-soft needs --pseudo-labels + --pseudo-val + --kd40-cache + --kd40-cache-extra (§95 ii)")
    if args.kd40_cache_extra is not None and not args.pseudo_soft:
        raise SystemExit("--kd40-cache-extra is the --pseudo-soft path's file only (§95 ii)")
    if args.kd40_geo is not None and (args.kd40_cache is None or args.pseudo_soft or args.kd40_geo < 1):
        raise SystemExit("--kd40-geo K needs --kd40-cache (the geo file) and is arm (iii) alone — not with --pseudo-soft (§95)")
    if args.videomix is not None and (args.pose_aux is not None or args.kd_cache is not None):
        raise SystemExit("--videomix is never cascaded with --pose-aux / --kd-cache (QUEUE §30)")
    if args.mixstyle and (args.arch != "ircsn_r50" or args.videomix is not None or args.pose_aux is not None):
        raise SystemExit("--mixstyle is wired for ircsn_r50 only and is never cascaded (QUEUE §35)")
    forced_window = None
    if args.swad_window:
        forced_window = tuple(int(v) for v in args.swad_window.split(","))
        if len(forced_window) != 2 or not (0 <= forced_window[0] <= forced_window[1] < args.epochs):
            raise SystemExit(f"--swad-window must be S,E with 0 <= S <= E < epochs, got {args.swad_window!r}")
    machine = resolve_machine(args.machine)  # before any work: a mislabelled run is wasted
    # Same rule, same reason: an unknown --arch must cost zero seconds, not fail
    # after the 26 GB cache is indexed. Resolved once here so that BOTH save
    # sites stamp the identical value and neither can recompute it differently.
    ckpt_norm = norm_key(args.branch, args.arch)
    depth_lut = None              # §96 (a)/(b): the ordinal branches' stamped LUT — set at the dataset block, saved at both sites
    if args.input_adapt in ("irnorm", "valid") and (args.branch != "depthir" or not args.arch.startswith("ircsn")):
        raise SystemExit(f"[train] --input-adapt {args.input_adapt} is the depth+IR ir-CSN arm only (QUEUE §96)")
    if args.input_adapt == "valid":
        # §96 (c) (D29): the validity plane is a FIFTH input channel, so the member trains, is cached, finalised, packed and
        # served under the 5-statistic key — the stamp every rebuild reads (dataset.VALID_MASK_NORMS).
        ckpt_norm = "depthir_valid"
    input_size = tuple(int(v) for v in args.input_size.split("x")) if args.input_size else None
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16  # trap 1
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    run = args.run_name or f"{args.branch}_f{args.fold}_{int(time.time())}"
    outdir = args.out / run
    outdir.mkdir(parents=True, exist_ok=True)

    hparams = hparams_of(args)

    if args.shuffle_labels and args.branch != "imu":
        raise SystemExit("--shuffle-labels is the §66 control for the imu branch only")
    if args.branch == "skeleton":
        # P2 (D8's build): pose clips come from src/skeleton_data.py's cache,
        # not the image shards — same item schema, same collate, same loops.
        # The unsupported knobs fail loudly rather than silently not applying.
        from src.skeleton_data import SkeletonDataset
        for flag, val in (("--train-subjects", args.train_subjects),
                          ("--all-subjects", args.all_subjects or None),
                          ("--pseudo-labels", args.pseudo_labels),
                          ("--kd-cache", args.kd_cache)):
            if val is not None:
                raise SystemExit(f"[train] {flag} is not wired for the skeleton branch")
        tr_ds = SkeletonDataset(mode="train", fold=args.fold, cache=args.cache,
                                seed=args.seed)
        va_ds = SkeletonDataset(mode="val", fold=args.fold, cache=args.cache)
    elif args.branch == "imu":
        # §66 (D23): the IMU member trains from src/imu_data.py's cache with the same item schema,
        # collate and loops. --all-subjects IS wired (a 023 needs it — the one difference from the
        # skeleton build, B §3.1 #5); the other knobs refuse loudly.
        from src.imu_data import IMUDataset
        for flag, val in (("--train-subjects", args.train_subjects),
                          ("--pseudo-labels", args.pseudo_labels),
                          ("--kd-cache", args.kd_cache)):
            if val is not None:
                raise SystemExit(f"[train] {flag} is not wired for the imu branch")
        if args.arch not in NON_IMAGE_ARCHS or args.arch == "tcn_1d":
            raise SystemExit(f"[train] --branch imu needs an imu arch (cnn1d_imu | dtcn_imu | xf_imu), got {args.arch!r}")
        tr_ds = IMUDataset(mode="train", fold=args.fold, cache=args.cache, seed=args.seed,
                           all_subjects=args.all_subjects, shuffle_labels=args.shuffle_labels)
        va_ds = None if args.all_subjects else IMUDataset(mode="val", fold=args.fold, cache=args.cache)
    else:
        common = dict(cache_dir=args.cache, manifest_csv=args.manifest, branch=args.branch,
                      split="train", fold=args.fold, T=args.T, seed=args.seed,
                      norm=ckpt_norm, input_size=input_size)
        # J5 (m86): the ablation applies to the TRAIN dataset only. Passing it to the
        # val dataset would change the metric, which is why it is not in `common`.
        geo_kw = dict(geo_draws=args.kd40_geo, geo_T=GEO_T) if args.kd40_geo else {}     # §95 (iii): the train loader only
        tr_ds = ClipDataset(mode="train", train_subjects=args.train_subjects,
                            all_subjects=args.all_subjects, aug_wide=args.aug_wide,
                            aug_tcrop=args.aug_tcrop, aug_transplant=args.aug_transplant,
                            aug_trivial=args.aug_trivial, **geo_kw, **common)
        # J9 (m87): with every subject in training there is no held-out side. Building
        # a val dataset anyway would produce an IN-SAMPLE number that looks exactly
        # like every other val_acc in the ledger, and §11.3 says a number in the docs
        # must be a reproducible number. The dataset is not built at all, so there is
        # nothing to accidentally quote.
        va_ds = None if args.all_subjects else ClipDataset(mode="val", **common)
        if args.branch in ORD_BRANCHES:
            # §96 (a)/(b) (D29): the depth inverse-LUT is a TRAIN statistic stamped into the member checkpoint (L7's second clause;
            # never a repository constant): the fold's rebuild (cache/depth_lut_f{f}.json; the all-18 member's depth_lut_train.json),
            # asserted BITWISE equal to the LUT the training shard was decoded with — the same table, or STOP.
            from src.preprocess import load_depth_lut, lut_sha16
            lut_path = ROOT / "cache" / ("depth_lut_train.json" if args.all_subjects else f"depth_lut_f{args.fold}.json")
            lut_np = load_depth_lut(lut_path)
            shard_sha = tr_ds.shard_meta.get("depth_lut_sha16")
            if shard_sha != lut_sha16(lut_np):
                raise SystemExit(f"🔴 the stamp {lut_path.name} (sha16 {lut_sha16(lut_np)}) is not the LUT the shard was decoded with ({shard_sha}) — STOP (§96 a/b)")
            depth_lut = torch.from_numpy(lut_np)
            print(f"[train] §96 (a)/(b): depth_lut stamp = {lut_path.name} (sha16 {lut_sha16(lut_np)}, {lut_np.shape[0]} levels) — equal to the shard's decode LUT")
        if args.drop_dup_exact is not None:
            # §81 (a): the exact cross-class duplicates leave the TRAIN items; va_ds is never touched.
            n_drop = drop_dup_exact(tr_ds, load_dup_map(args.drop_dup_exact, args.fold))
            print(f"[train] §81 (a) drop-dup-exact: {n_drop} train clips removed ({len(tr_ds)} remain); val untouched")

    # ── Self-training: the thresholded test clips join the TRAINING set ─────────
    # A ConcatDataset rather than a change to the shard logic — train and test live
    # in DIFFERENT shards, and one dataset spanning both would mean reworking the
    # memmap path that trap 1 exists to protect.
    carrier_ids = None            # §95 (ii): the clip ids whose CE term is masked under --pseudo-soft
    if args.pseudo_labels is not None:
        import csv as _csv
        from torch.utils.data import ConcatDataset
        with args.pseudo_labels.open(encoding="utf-8") as fh:
            keep = {r["clip_id"]: int(r["pseudo_label"]) for r in _csv.DictReader(fh)
                    if float(r["confidence"]) >= args.pseudo_conf}
        pl_common = dict(common)
        if args.pseudo_val:            # §58: the fold's val clips carry the teacher's labels
            pl_ds = ClipDataset(mode="val", pseudo_labels=keep, pseudo_val=True, augment=True, **pl_common)
            if args.pseudo_soft:
                carrier_ids = {r.clip_id for r in pl_ds.items}
                print(f"[train] §95 (ii) --pseudo-soft: {len(carrier_ids)} carriers take NO CE term — KD from --kd40-cache-extra only")
            print(f"[train] SELF-TRAINING (fold instrument): {len(keep)} VAL clips at conf >= {args.pseudo_conf} "
                  f"-> {len(pl_ds)} usable on branch {args.branch}; labels from the CSV only")
        else:
            pl_common.update(split="test", fold=args.fold)
            pl_ds = ClipDataset(mode="test", pseudo_labels=keep, augment=True, **pl_common)
            print(f"[train] SELF-TRAINING: {len(keep)} test clips at conf >= {args.pseudo_conf} "
                  f"-> {len(pl_ds)} usable on branch {args.branch}")
        # A zero-overlap here is a KEY-FORMAT bug, not an empty threshold, and it
        # is indistinguishable from "self-training does nothing" once training ends
        # (m107 — it happened). Fail loudly, before 2.5 h of GPU.
        if keep and len(pl_ds) == 0:
            raise SystemExit(
                f"--pseudo-labels matched {len(keep)} rows but ZERO clips in the test shard.\n"
                f"  file keys look like: {list(keep)[:2]}\n"
                f"  the dataset expects bare clip_ids, e.g. SM_test_0001.\n"
                f"Refusing to train on an empty pseudo set that would look like a null."
            )
        print(f"[train]   train {len(tr_ds)} + pseudo {len(pl_ds)} = {len(tr_ds) + len(pl_ds)} clips "
              f"({100*len(pl_ds)/max(1, len(tr_ds)):.1f}% added)")
        print("[train]    rules-class PSEUDO-LABEL(R7). No test LABEL is read anywhere; "
              "these are this project's own predictions on test INPUTS.")
        tr_ds = ConcatDataset([tr_ds, pl_ds])

    if args.pk_sampler is not None:
        # §84 (D26): P×K batches replace the shuffled loader; the step count per epoch is the recipe's (n // batch).
        P, K = (int(v) for v in args.pk_sampler.lower().split("x"))
        if P * K != args.batch_size or not hasattr(tr_ds, "items"):
            raise SystemExit(f"--pk-sampler {args.pk_sampler}: P·K must equal --batch-size {args.batch_size} on a plain ClipDataset (§84)")
        pk = PKSampler([r.label for r in tr_ds.items], [r.user for r in tr_ds.items], P, K, args.seed)
        tr = DataLoader(tr_ds, batch_sampler=pk, num_workers=args.workers, pin_memory=True, collate_fn=collate,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
        print(f"[train] §84 P×K sampler {P}×{K}: {len(pk)} batches/epoch over {len(pk.classes)} eligible classes")
    else:
        tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, collate_fn=collate,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    if args.supcon is not None and args.pk_sampler is None:
        raise SystemExit("--supcon needs --pk-sampler (positives exist only in P×K batches) (§84)")
    va = None if va_ds is None else DataLoader(
                    va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True, collate_fn=collate,
                    persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)

    build_kw = {"mixstyle": True} if args.mixstyle else {}
    if args.input_adapt:
        build_kw["input_adapt"] = args.input_adapt
    if args.freeze_stages:
        if args.arch not in ("ircsn_r50", "ircsn_r152"):
            raise SystemExit(f"--freeze-stages is wired for the ir-CSN archs only, got {args.arch!r} (§71)")
        build_kw["freeze_stages"] = args.freeze_stages
    if args.head_dropout is not None:
        build_kw["dropout"] = args.head_dropout
    if args.init_from is not None and args.no_pretrained:
        raise SystemExit("--init-from replaces the pretrained start; do not pass --no-pretrained with it (§77)")
    if args.csn_weights is not None:
        # §82 (D26): a different pretrained START from the same weight directory (strict load inside CSNBranch).
        if args.arch not in ("ircsn_r50", "ircsn_r152") or args.no_pretrained or args.init_from is not None:
            raise SystemExit("--csn-weights is the ir-CSN pretrained start; not with --no-pretrained or --init-from (§82)")
        build_kw["weights"] = args.csn_weights
    model = build(args.branch, n_segment=args.T, arch=args.arch, tsm=args.tsm,
                  pretrained=not args.no_pretrained,
                  partial_bn=not args.no_partial_bn, **build_kw).to(device)
    if args.init_from is not None:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if init.get("arch") != args.arch or int(init.get("T", args.T)) != args.T:
            raise SystemExit(f"--init-from {args.init_from} is arch {init.get('arch')!r} T {init.get('T')} — this run is {args.arch!r} T {args.T} (§77)")
        model.load_state_dict(init["model"], strict=True)
        print(f"[train] §77 init-from {args.init_from} (adapted on set {init.get('set')!r}, fold {init.get('fold')}, "
              f"{init.get('n_clips')} clips, {init.get('epochs')} epochs; head untouched) — strict load OK")
    if args.head_init is not None:
        if args.init_from is not None:
            raise SystemExit("--head-init and --init-from are different starts; one at a time (§79)")
        apply_head_init(model, args.head_init)
    # channels_last is a RANK-4 memory format. On a 3D backbone it raises
    # "required rank 4 tensor to use channels_last" -- loud, but it still has to
    # be asked rather than assumed. PRETRAINING-GAP.md §4's sixth site.
    # tcn_1d is rank-3 end to end -- channels_last is rank-4 and Tensor.to
    # raises on its Conv1d weights, so it is excluded the same way rank-5 is.
    if args.arch not in RANK5_ARCHS and args.arch not in NON_IMAGE_ARCHS:
        model = model.to(memory_format=torch.channels_last)

    if args.branch == "imu":
        # §66: per-site, per-channel z-scoring from the fold's TRAIN subjects only (L7), computed over the
        # sites that are PRESENT in each train clip (a zero-filled absent site is not a sample), stamped
        # into the model's buffers and therefore into every checkpoint this run writes. Never a NORM key.
        Xtr, Ptr = tr_ds.X, tr_ds.present.astype(bool)                 # (N, T, 40), (N, 5)
        mean = np.zeros(Xtr.shape[-1], dtype=np.float64)
        std = np.ones(Xtr.shape[-1], dtype=np.float64)
        for si in range(5):
            blk = Xtr[Ptr[:, si]][:, :, si * 8:(si + 1) * 8].reshape(-1, 8)
            mean[si * 8:(si + 1) * 8] = blk.mean(0)
            std[si * 8:(si + 1) * 8] = blk.std(0)
        model.set_input_stats(torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32))
        print(f"[train] imu input stats stamped from {len(Xtr)} fold-train clips (present sites only): "
              f"mean[:8] {np.round(mean[:8], 3).tolist()} · std[:8] {np.round(std[:8], 3).tolist()}"
              + (" · SHUFFLED-LABEL CONTROL (§66 a)" if args.shuffle_labels else ""))

    # ── J8/8b: the auxiliary KD head, built OUTSIDE the model on purpose ────────
    # It is a separate nn.Module, so `model.state_dict()` cannot contain it and no
    # teacher-derived parameter can reach a checkpoint or the packer even by
    # accident. tests/check_teacher_fence.py asserts the artefact side of the same
    # fence; this is the parameter side, and it is structural rather than a promise.
    kd40_targets = None
    if args.kd40_cache is not None:
        kz = np.load(args.kd40_cache, allow_pickle=False)
        if args.kd40_geo:
            # §95 (iii): the geo file is keyed (clip_id, draw_k); its K must be the loader's K, or the draws would not correspond
            if "K" not in kz.files or int(kz["K"]) != args.kd40_geo:
                raise SystemExit(f"--kd40-geo {args.kd40_geo} but the cache is stamped K = {int(kz['K']) if 'K' in kz.files else 'none'} — refusing (§95 iii)")
            kd40_targets = {f"{c}|{int(k)}": kz["logits"][i].astype(np.float32) for i, (c, k) in enumerate(zip(kz["clip_ids"], kz["draw_k"]))}
        else:
            kd40_targets = {str(c): kz["logits"][i].astype(np.float32) for i, c in enumerate(kz["clip_ids"])}
        print(f"[train] §75 KD-40: {len(kd40_targets)} teacher targets from {args.kd40_cache.name} "
              f"(teacher {kz['teacher']}, fold {int(kz['fold'])}) · alpha={args.kd_alpha} temp={args.kd_temp} · "
              "on the model's own 40-way logits; clips without a target get CE only")
        if int(kz["fold"]) != args.fold:
            raise SystemExit(f"--kd40-cache is fold {int(kz['fold'])}, this run is fold {args.fold} — refusing (§75)")
    if args.kd40_cache_extra is not None:
        # §95 (ii): the carriers' targets — the SAME fold's VALCARRIER file; its clip set must be disjoint from the train file's
        # (FM4/FM5: a train clip never takes a carrier target and vice versa); merged into the one dict the KD term reads.
        kx = np.load(args.kd40_cache_extra, allow_pickle=False)
        if int(kx["fold"]) != args.fold:
            raise SystemExit(f"--kd40-cache-extra is fold {int(kx['fold'])}, this run is fold {args.fold} — refusing (§95 ii)")
        extra = {str(c): kx["logits"][i].astype(np.float32) for i, c in enumerate(kx["clip_ids"])}
        overlap = set(extra) & set(kd40_targets)
        if overlap:
            raise SystemExit(f"--kd40-cache-extra shares {len(overlap)} clip ids with --kd40-cache — refusing (FM4/FM5)")
        kd40_targets.update(extra)
        print(f"[train] §95 (ii) KD-40 extra: {len(extra)} carrier targets from {args.kd40_cache_extra.name} (teacher {kx['teacher']}, fold {int(kx['fold'])})")
    dup_targets = None
    if args.dup_soft_targets is not None:
        if args.videomix is not None or args.pseudo_labels is not None or args.kd40_cache is not None or args.kd_cache is not None:
            raise SystemExit("--dup-soft-targets is the plain recipe's criterion only — not with videomix, pseudo-labels or KD (§81)")
        dup_targets = dup_soft_targets(load_dup_map(args.dup_soft_targets, args.fold), args.label_smoothing)
        print(f"[train] §81 (b) dup-soft-targets: {len(dup_targets)} train clips carry a duplicate-aware target "
              f"(LS {args.label_smoothing} on top)")
    kd_head = kd_targets = kd_feat = None
    if args.kd_cache is not None:
        if args.arch != "s3d":
            raise SystemExit(f"--kd-cache is wired for arch=s3d only, got {args.arch!r}. "
                             "The hook below targets backbone.classifier's input.")
        kdz = np.load(args.kd_cache, allow_pickle=False)
        kd_targets = {c: kdz["logits"][i].astype(np.float32)
                      for i, c in enumerate(kdz["clip_ids"])}
        print(f"[train] J8 KD: {len(kd_targets)} teacher targets from {args.kd_cache.name} "
              f"(adapter={kdz['adapter']}, T={kdz['T']}, bn_eps={kdz['bn_eps']})")
        print(f"[train] J8 KD: alpha={args.kd_alpha} temp={args.kd_temp} "
              f"· 400-way aux head on the 1024-d pooled feature · NOT saved")
        kd_head = nn.Linear(1024, 400).to(device)
        kd_feat = {}

        def _grab(_mod, inp, _out):
            kd_feat["z"] = inp[0].flatten(1)      # (B,1024,1,1,1) -> (B,1024)

        model.backbone.classifier.register_forward_hook(_grab)

    # ── T4: the pose-regression head, built OUTSIDE the model on purpose (QUEUE §27) ──
    pose_head = pose_targets = pose_feat = None
    if args.pose_aux is not None:
        if args.arch != "ircsn_r50" or args.branch == "skeleton":
            raise SystemExit(f"--pose-aux is wired for the ircsn_r50 image branches only, got "
                             f"arch={args.arch!r} branch={args.branch!r}")
        if args.all_subjects or args.pseudo_labels is not None:
            raise SystemExit("--pose-aux: the fold-train target statistics are defined for fold cells only")
        pz = np.load(args.cache / "skeleton_train.npz", allow_pickle=False)
        X, present, pids = pz["X"], pz["present"].astype(bool), [str(c) for c in pz["clip_ids"]]
        xyz = X[:, :, :51]                                   # (N, 24, 17x3)
        dxyz = X[:, :, 51:]
        summary = np.concatenate([xyz.mean(1), xyz.std(1), np.abs(dxyz).mean(1)], axis=1)  # (N, 153)
        train_clips = {it.clip_id for it in tr_ds.items}
        fit = np.array([present[i] and pids[i] in train_clips for i in range(len(pids))])
        mu, sd = summary[fit].mean(0), summary[fit].std(0) + 1e-6
        Z = np.clip((summary - mu) / sd, -5.0, 5.0).astype(np.float32)
        pose_targets = {pids[i]: Z[i] for i in range(len(pids)) if present[i]}
        print(f"[train] T4 pose-aux: alpha={args.pose_aux} · 153-d targets z-scored on {int(fit.sum())} "
              f"fold-train clips with pose · {sum(1 for c in train_clips if c in pose_targets)}/"
              f"{len(train_clips)} train clips carry a target · head Linear(2048,153) NOT saved")
        pose_head = nn.Linear(2048, 153).to(device)
        pose_feat = {}

        def _grab_pool(_mod, _inp, out):
            pose_feat["z"] = out.flatten(1)                  # (B, 2048)

        model.pool.register_forward_hook(_grab_pool)

    opt_params = param_groups(model, args.weight_decay)
    if args.llrd is not None:
        # §68 (b): one group per (layer, decay/no-decay); `lr_scale` is applied at every lr assignment below.
        if args.optim != "adamw" or not hasattr(model, "layer_id"):
            raise SystemExit("--llrd needs --optim adamw and an arch that exposes layer_id() (vmae2_vits)")
        groups: dict[tuple[int, bool], list] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            nd = p.ndim <= 1 or name.endswith(".bias")
            groups.setdefault((model.layer_id(name), nd), []).append(p)
        opt_params = [{"params": ps, "weight_decay": 0.0 if nd else args.weight_decay,
                       "lr_scale": args.llrd ** (model.n_layers - 1 - lid)}
                      for (lid, nd), ps in sorted(groups.items())]
        print(f"[train] §68 (b) layer-wise lr decay {args.llrd}: {len(opt_params)} groups over {model.n_layers} layers "
              f"(scale {args.llrd ** (model.n_layers - 1):.4f} at the stem → 1.0 at the head)")
    l2sp_anchors = None
    if args.l2sp is not None:
        if args.llrd is not None:
            raise SystemExit("--l2sp and --llrd are different group schemes; one at a time (§80)")
        opt_params, l2sp_anchors = l2sp_groups(model, opt_params, args.weight_decay)
        print(f"[train] §80 L2-SP α {args.l2sp}: {len(l2sp_anchors)} backbone tensors anchored to their loaded values "
              f"(their optimiser decay → 0); fc keeps wd {args.weight_decay} toward 0")
    if kd_head is not None:
        opt_params = opt_params + [{"params": list(kd_head.parameters()),
                                    "weight_decay": args.weight_decay}]
    if pose_head is not None:
        opt_params = opt_params + [{"params": list(pose_head.parameters()),
                                    "weight_decay": args.weight_decay}]
    sc_head = sc_feat = None
    if args.supcon is not None:
        # §84: the projection head is a SEPARATE module (never in the checkpoint); the pooled 2048-d vector is read by a hook.
        if args.videomix is not None or args.pseudo_labels is not None or args.kd40_cache is not None or args.kd_cache is not None \
                or args.dup_soft_targets is not None or args.ols is not None:
            raise SystemExit("--supcon is an auxiliary on the plain recipe only (§84)")
        sc_head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Linear(512, 128)).to(device)
        sc_feat = {}

        def _grab_sc(_mod, _inp, out):
            sc_feat["z"] = out.flatten(1)

        model.pool.register_forward_hook(_grab_sc)
        opt_params = opt_params + [{"params": list(sc_head.parameters()), "weight_decay": args.weight_decay}]
        print(f"[train] §84 SupCon λ {args.supcon} τ 0.1: positives = same class ∧ different subject; projection 2048→512→128 discarded")
    if args.optim == "adamw":
        opt = torch.optim.AdamW(opt_params, lr=args.lr, betas=(0.9, 0.999))
        print(f"[train] optimiser AdamW lr {args.lr} betas (0.9, 0.999) wd {args.weight_decay} (§68 arm b)")
    else:
        opt = torch.optim.SGD(opt_params, lr=args.lr, momentum=0.9, nesterov=True)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ols = None
    if args.ols is not None:
        if args.videomix is not None or args.pseudo_labels is not None or args.kd40_cache is not None or args.kd_cache is not None \
                or args.dup_soft_targets is not None:
            raise SystemExit("--ols replaces the plain criterion only — not with videomix, pseudo-labels, KD or --dup-soft-targets (§83)")
        ols = OnlineLS(NUM_CLASSES, args.ols, device)
        print(f"[train] §83 online label smoothing α {args.ols}: S starts uniform, replaced per epoch from the correctly classified "
              f"train clips; --label-smoothing {args.label_smoothing} is NOT applied")

    sc_run, sc_n = 0.0, 0                                    # §84 per-epoch SupCon mean (print only)
    steps_per_epoch = len(tr) if not args.limit_batches else min(len(tr), args.limit_batches)
    total_iters = steps_per_epoch * args.epochs
    warmup_iters = steps_per_epoch * args.warmup_epochs  # trap 3
    # Trap 6. Per-iteration snapshots: our epochs are ~121 steps, so per-epoch
    # would average 10 weights over the tail instead of ~1,210.
    swa_model = AveragedModel(model) if (args.swa and not args.swad) else None
    swa_start = swa_start_iter(total_iters, args.swa_start_frac) if args.swa else total_iters
    sums = EpochSums(model, outdir / "swad_sums") if args.swad else None   # T1: per-epoch iterate sums
    val_losses: list[float] = []

    print(f"[train] run={run}")
    print(f"[train] {args.branch} fold {args.fold} · train {len(tr_ds)} clips / {len(tr)} steps · "
          f"val {'NONE (J9 all-subjects)' if va_ds is None else str(len(va_ds)) + ' clips'}")
    def _subjects(ds):
        if ds is None:
            return "NONE"
        if hasattr(ds, "users"):  # SkeletonDataset carries a plain user array
            return sorted({int(u) for u in ds.users})
        items = (ds.items if hasattr(ds, "items")
                 else [i for d in ds.datasets for i in d.items])
        return sorted({i.user for i in items})
    print(f"[train] subjects train={_subjects(tr_ds)}")
    print(f"[train] subjects val  ={_subjects(va_ds)}")
    print(f"[train] params {param_count(model)/1e6:.2f} M · device {device} · amp {amp_dtype}")
    print(f"[train] {total_iters} iters, {warmup_iters} warmup")
    if sums is not None:
        print(f"[train] SWAD (QUEUE §24): CONSTANT lr {args.lr} after {warmup_iters} warmup iters · per-iteration "
              f"sums per epoch under {outdir / 'swad_sums'} · window "
              + (f"FORCED {forced_window}" if forced_window else "by the loss-valley rule on per-epoch val loss"))
    elif swa_model is not None:
        print(f"[train] SWA from iter {swa_start} (frac {args.swa_start_frac}) "
              f"· {total_iters - swa_start} per-iteration snapshots")
    else:
        print("[train] SWA DISABLED — selecting by best epoch, which overfits val")

    log_path = ROOT / f"experiments-{machine.lower()}.jsonl"
    best = {"val_acc": -1.0, "epoch": -1}
    it = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()  # re-applies partial_bn every epoch
        running = n = 0
        kd_run = kd_n = 0
        aux_run = aux_n = 0
        carrier_n = 0
        geo_hits = 0
        t0 = time.time()
        for i, batch in enumerate(tr):
            if args.limit_batches and i >= args.limit_batches:
                break
            lr = (args.lr if (args.swad and it >= warmup_iters)          # SWAD: constant after warmup
                  else lr_at(it, total_iters, warmup_iters, args.lr, 0.1))
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)      # lr_scale only exists under --llrd (§68 b)

            x = batch["x"].to(device, non_blocking=True)  # model applies channels_last after reshape
            y = batch["y"].to(device, non_blocking=True)
            y_mix, lam_mix = None, None
            if args.videomix is not None:
                # T7 (QUEUE §30): one box per BATCH, identical on every frame and channel of every clip
                # (clip-consistent by construction), filled from a permutation of the batch. Drawn from
                # torch's RNG so the run stays seed-reproducible in the same sense as the rest.
                lam = float(torch.distributions.Beta(args.videomix, args.videomix).sample())
                Hh, Ww = x.shape[-2], x.shape[-1]
                cut = math.sqrt(max(0.0, 1.0 - lam))
                ch, cw = int(round(Hh * cut)), int(round(Ww * cut))
                cy, cx = int(torch.randint(0, Hh, (1,))), int(torch.randint(0, Ww, (1,)))
                y0, y1 = max(0, cy - ch // 2), min(Hh, cy + (ch + 1) // 2)
                x0, x1 = max(0, cx - cw // 2), min(Ww, cx + (cw + 1) // 2)
                perm = torch.randperm(x.shape[0], device=x.device)
                if y1 > y0 and x1 > x0:
                    x[:, :, :, y0:y1, x0:x1] = x[perm][:, :, :, y0:y1, x0:x1]
                lam_mix = 1.0 - ((y1 - y0) * (x1 - x0)) / float(Hh * Ww)
                y_mix = y[perm]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(x)
                loss = (crit(out, y) if y_mix is None
                        else lam_mix * crit(out, y) + (1.0 - lam_mix) * crit(out, y_mix))
                if carrier_ids is not None:
                    # §95 (ii) --pseudo-soft: the carriers in THIS batch have NO CE term (a per-sample mask); a batch without a
                    # carrier keeps the line above untouched — bitwise the incumbent's arithmetic (tests/check_pseudo_soft.py).
                    is_c = torch.tensor([c in carrier_ids for c in batch["clip_id"]], device=device)
                    if bool(is_c.any()):
                        ce = F.cross_entropy(out, y, label_smoothing=args.label_smoothing, reduction="none")
                        loss = ce[~is_c].mean() if bool((~is_c).any()) else out.float().sum() * 0.0
                        carrier_n += int(is_c.sum())
                if dup_targets is not None:
                    dl = dup_loss(out, y, batch["clip_id"], dup_targets, args.label_smoothing)   # §81 (b)
                    if dl is not None:
                        loss = dl
                if ols is not None:
                    loss = ols.loss(out, y)                                                     # §83
                if sc_head is not None:
                    sc = supcon_loss(F.normalize(sc_head(sc_feat["z"].float()), dim=1), y, batch["user"].to(device), 0.1)
                    loss = loss + args.supcon * sc                                             # §84
                    sc_run += float(sc.detach()); sc_n += 1
                if kd_head is not None:
                    # Clips absent from the cache (a branch's unusable set differs
                    # from the other's) are MASKED, never zero-filled: a zero target
                    # is a confident wrong distribution, not a missing one.
                    ok = [j for j, c in enumerate(batch["clip_id"]) if c in kd_targets]
                    if ok:
                        tt = torch.from_numpy(
                            np.stack([kd_targets[batch["clip_id"][j]] for j in ok])
                        ).to(device)
                        z = kd_head(kd_feat["z"][ok].float())
                        kd = F.kl_div(F.log_softmax(z / args.kd_temp, dim=1),
                                      F.softmax(tt / args.kd_temp, dim=1),
                                      reduction="batchmean") * (args.kd_temp ** 2)
                        loss = loss + args.kd_alpha * kd
                        kd_run += float(kd.detach()) * len(ok)
                        kd_n += len(ok)
                if kd40_targets is not None:
                    if args.kd40_geo:
                        # §95 (iii): the target of THIS item's replayed draw
                        keys = [f"{c}|{int(k)}" for c, k in zip(batch["clip_id"], batch["draw_k"])]
                        ok = [j for j, kk in enumerate(keys) if kk in kd40_targets]
                        geo_hits += len(ok)
                    else:
                        keys = list(batch["clip_id"])
                        ok = [j for j, c in enumerate(batch["clip_id"]) if c in kd40_targets]
                    if ok:
                        tt = torch.from_numpy(np.stack([kd40_targets[keys[j]] for j in ok])).to(device)
                        kd = F.kl_div(F.log_softmax(out[ok].float() / args.kd_temp, dim=1),
                                      F.softmax(tt / args.kd_temp, dim=1),
                                      reduction="batchmean") * (args.kd_temp ** 2)
                        loss = (1.0 - args.kd_alpha) * loss + args.kd_alpha * kd
                        kd_run += float(kd.detach()) * len(ok)
                        kd_n += len(ok)
                if pose_head is not None:
                    ok = [j for j, c in enumerate(batch["clip_id"]) if c in pose_targets]
                    if ok:
                        tt = torch.from_numpy(np.stack([pose_targets[batch["clip_id"][j]] for j in ok])).to(device)
                        pa = F.mse_loss(pose_head(pose_feat["z"][ok].float()), tt)
                        loss = loss + args.pose_aux * pa
                        aux_run += float(pa.detach()) * len(ok)
                        aux_n += len(ok)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            if l2sp_anchors is not None:
                l2sp_add(l2sp_anchors, args.l2sp)   # §80: after the clip, before the step — where the optimiser's own decay acts
            opt.step()
            if swa_model is not None and it >= swa_start:
                swa_model.update_parameters(model)  # snapshot: per_iteration
            if sums is not None:
                sums.add()                          # SWAD: every iterate, every epoch
            running += float(loss.detach()) * y.numel()
            n += y.numel()
            it += 1
            if i % 50 == 0 or args.limit_batches:   # §103 (d): a smoke (--limit-batches) prints EVERY step's pre-clip gradient norm
                print(f"  e{epoch:02d} {i:4d}/{steps_per_epoch}  loss {float(loss.detach()):.3f}  lr {lr:.5f}  gnorm {float(gn):.1f}", flush=True)

        if ols is not None:
            print(f"[train] §83 OLS epoch {epoch:02d}: S rows refreshed for {ols.end_epoch()} classes")
        if sc_head is not None:
            print(f"[train] §84 SupCon epoch {epoch:02d}: mean term {sc_run / max(1, sc_n):.4f} over {sc_n} steps")
            sc_run, sc_n = 0.0, 0
        if va is None:
            acc = bal = vloss = None; seen = classes = 0
        else:
            acc, bal, seen, classes, vloss = evaluate(model, va, device, amp_dtype,
                                                      limit=args.limit_batches or None)
            val_losses.append(vloss)
        if sums is not None:
            sums.flush(epoch)
        dt = time.time() - t0
        train_loss = running / max(1, n)
        kd_loss = (kd_run / kd_n) if kd_n else None
        aux_loss = (aux_run / aux_n) if aux_n else None
        kd_str = (f" · kd {kd_loss:.4f} ({kd_n} clips)" if kd_loss is not None else "") + \
                 (f" · pose-aux {aux_loss:.4f} ({aux_n} clips)" if aux_loss is not None else "")
        if acc is None:
            # m63's audit signal is train_loss, not val_acc, and it survives here.
            print(f"[epoch {epoch:02d}] loss {train_loss:.4f}{kd_str} · val_acc NONE "
                  f"(J9: no held-out subjects exist) · {dt:.0f}s", flush=True)
        else:
            print(f"[epoch {epoch:02d}] loss {train_loss:.4f}{kd_str} · val_acc {acc:.4f} "
                  f"· val_loss {vloss:.4f} · balanced {bal:.4f} · {seen} clips / {classes} classes · {dt:.0f}s", flush=True)

        if acc is not None and acc > best["val_acc"]:
            best = {"val_acc": acc, "balanced": bal, "epoch": epoch}
            torch.save({"model": model.state_dict(), "branch": args.branch, "fold": args.fold,
                        "T": args.T, "epoch": epoch, "val_acc": acc,
                        "in_channels": model.in_channels,
                        # §J-DQ (c): how these weights were built and what
                        # distribution they were fed. predict.py cannot read YAML
                        # and cannot guess; a wrong norm does not raise.
                        "arch": args.arch, "norm": ckpt_norm, "input_adapt": args.input_adapt, "depth_lut": depth_lut,
                        "input_size": args.input_size, "freeze_stages": args.freeze_stages}, outdir / "best.pt")

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "run": run, "machine": machine, "branch": args.branch, "fold": args.fold,
                "seed": args.seed, "cudnn_benchmark": args.cudnn_benchmark, "hparams": hparams,
                "phase": "epoch", "epoch": epoch, "train_loss": round(train_loss, 5),
                "all_subjects": bool(args.all_subjects),
                "kd_loss": None if kd_loss is None else round(kd_loss, 5),
                "kd_clips": kd_n, "kd_carriers": carrier_n, "kd_geo_hits": geo_hits,
                "aux_loss": None if aux_loss is None else round(aux_loss, 5),
                "aux_clips": aux_n,
                "val_acc": None if acc is None else round(acc, 5),
                "val_loss": None if vloss is None else round(vloss, 5),
                "val_balanced_acc": None if bal is None else round(bal, 5),
                "val_clips": seen, "val_classes_seen": classes, "lr_end": round(lr, 6),
                "epoch_seconds": round(dt, 1), "smoke": bool(args.limit_batches),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else 0.0,
            }) + "\n")

    total_h = (time.time() - t_start) / 3600
    print(f"\n[train] best-epoch val_acc {best['val_acc']:.4f} at epoch {best['epoch']} · {total_h:.2f} h")

    # Trap 6: SWA at the end of the schedule is the recipe's checkpoint selector.
    selected, selected_acc = "best_epoch", best["val_acc"]
    core, n_avg, sel_name, swad_meta = None, 0, None, {}
    if sums is not None:
        # T1 (QUEUE §24): the window from the loss-valley rule (or forced), then the exact
        # mean of every iterate inside it, from the flushed per-epoch sums.
        if forced_window is not None:
            ts, te = forced_window
            conv, thr = None, None
        else:
            ts, te, conv, thr = swad_window(val_losses)
        avg, n_avg = window_average(outdir / "swad_sums", ts, te)
        core = copy.deepcopy(model)
        load_average(core, avg)
        sel_name = "swad"
        swad_meta = {"swad_window": [ts, te], "swad_converged": conv, "swad_threshold": thr,
                     "swad_forced": forced_window is not None,
                     "val_losses": [round(v, 5) for v in val_losses]}
        print(f"[train] SWAD window epochs [{ts}, {te}] ({n_avg} iterates) · converged {conv} · "
              f"threshold {thr if thr is None else round(thr, 4)} · forced {forced_window is not None}")
    elif swa_model is not None:
        n_avg = int(swa_model.n_averaged)
        if n_avg == 0:
            print(f"[train] ⚠ SWA collected 0 snapshots (start {swa_start} of {total_iters} iters). "
                  "No swa.pt written; selection falls back to the best epoch.")
        else:
            core = swa_model.module
            sel_name = "swa"
    if core is not None:
        if True:
            nb = recompute_bn_stats(core, tr, device, amp_dtype, limit=args.limit_batches or None)
            if va is None:
                acc = bal = None; seen = classes = 0
                print(f"[train] SWA written with NO val number — J9 all-subjects. {n_avg} snapshots "
                      f"· BN refit over {nb} batches")
                print("[train] This checkpoint's accuracy is UNMEASURED, not unreported. It can "
                      "only be predicted (m87's slope) and verified on the board.")
            else:
                acc, bal, seen, classes, _ = evaluate(core, va, device, amp_dtype,
                                                      limit=args.limit_batches or None)
                print(f"[train] {sel_name.upper()} val_acc {acc:.4f} · balanced {bal:.4f} · {n_avg} snapshots "
                      f"· BN refit over {nb} batches")
                print(f"[train] SWA − best-epoch = {acc - best['val_acc']:+.4f}  "
                      f"← what argmax over {args.epochs} validation evaluations was worth to itself")
            torch.save({"model": core.state_dict(), "branch": args.branch, "fold": args.fold,
                        "T": args.T, "epoch": args.epochs - 1, "val_acc": acc,
                        "in_channels": core.in_channels, "selection": sel_name,
                        **swad_meta,   # T1: window, convergence, loss curve (empty for plain SWA)
                        "arch": args.arch, "norm": ckpt_norm, "input_adapt": args.input_adapt, "depth_lut": depth_lut,  # §J-DQ (c)
                        "input_size": args.input_size,         # m73 — the third
                        "freeze_stages": args.freeze_stages,   # §71: the frozen set travels with the checkpoint
                        # J8 provenance. The teacher's PARAMETERS never reach this
                        # file (the aux head is a separate module), but the fact
                        # that a teacher shaped these weights must be readable
                        # from the artefact itself — §J-DQ (c)'s own principle.
                        "kd_cache": str(args.kd_cache) if args.kd_cache else None,
                        "kd_alpha": args.kd_alpha if (args.kd_cache or args.kd40_cache) else None,
                        "kd_temp": args.kd_temp if (args.kd_cache or args.kd40_cache) else None,
                        "kd40_cache": str(args.kd40_cache) if args.kd40_cache else None,   # §75 provenance
                        "init_from": str(args.init_from) if args.init_from else None,      # §77 provenance
                        "head_init": str(args.head_init) if args.head_init else None, "l2sp": args.l2sp,   # §79/§80
                        "drop_dup_exact": str(args.drop_dup_exact) if args.drop_dup_exact else None,        # §81 (a)
                        "dup_soft_targets": str(args.dup_soft_targets) if args.dup_soft_targets else None,  # §81 (b)
                        "csn_weights": args.csn_weights,                                                     # §82
                        "ols": args.ols,                                                                     # §83
                        "pk_sampler": args.pk_sampler, "supcon": args.supcon,                                   # §84

                        # J9: recorded IN the checkpoint so that anything reading
                        # it downstream can see val_acc is None BY CONSTRUCTION and
                        # not because a field went missing.
                        "all_subjects": bool(args.all_subjects),
                        "swa_n_averaged": n_avg, "swa_start_iter": swa_start,
                        "swa_start_frac": args.swa_start_frac}, outdir / "swa.pt")
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "run": run, "machine": machine, "branch": args.branch, "fold": args.fold,
                    "seed": args.seed, "cudnn_benchmark": args.cudnn_benchmark, "hparams": hparams,
                    "phase": "swa", "epoch": args.epochs - 1, "selection": sel_name,
                    "swad_window": swad_meta.get("swad_window"), "swad_converged": swad_meta.get("swad_converged"),
                    "all_subjects": bool(args.all_subjects),
                    "val_acc": None if acc is None else round(acc, 5),
                    "val_balanced_acc": None if bal is None else round(bal, 5),
                    "val_clips": seen, "val_classes_seen": classes,
                    "swa_n_averaged": n_avg, "swa_start_iter": swa_start,
                    "swa_start_frac": args.swa_start_frac, "bn_refit_batches": nb,
                    "best_epoch_val_acc": None if acc is None else round(best["val_acc"], 5),
                    "swa_minus_best_epoch": None if acc is None else round(acc - best["val_acc"], 5),
                    "train_hours": round(total_h, 3), "smoke": bool(args.limit_batches),
                    # Alongside train_hours, deliberately: cost is two numbers on
                    # this box and only one of them was ever recorded.
                    "peak_rss_gb": peak_rss_gb()[0],
                    "peak_rss_worker_gb": peak_rss_gb()[1],
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)
                    if device.type == "cuda" else 0.0,
                }) + "\n")
            selected, selected_acc = sel_name, acc

    if selected_acc is None:
        # J9: trap 5's floor check needs a val number and there is none. Say so
        # rather than skipping quietly — the floor check is how m63's broken cells
        # were caught, and this mode is BLIND TO IT. The train_loss curve above is
        # the only audit signal an all-subjects run has.
        print(f"[train] SELECTED {selected} · val_acc NONE (J9 all-subjects)")
        print(f"[train] ⚠ TRAP-5 FLOOR CHECK CANNOT RUN — no validation set. Final train_loss "
              f"{train_loss:.4f} is the only sanity signal; audit it before trusting this checkpoint.")
        return 0
    print(f"[train] SELECTED {selected} · val_acc {selected_acc:.4f}")
    # Trap 5, applied to the checkpoint actually selected -- not to the best epoch,
    # which is exactly the number trap 6 says not to believe.
    if not args.limit_batches and selected_acc < 0.11:
        if args.shuffle_labels:
            # §66 control (a): the ONE run class that is SUPPOSED to sit at the floor — its labels were
            # permuted, so a chance-level val_acc is the correct result, not a bug. Stated, exit 0.
            print(f"[train] {selected_acc:.4f} is at the floor BY DESIGN — this is the shuffled-label control "
                  "(§66 a); the checkpoint stands as the dilution-floor member.")
            return 0
        print(f"[train] 🔴 {selected_acc:.4f} is at or below the 0.10945 majority-class floor.")
        print("[train]    That is a BUG, not a weak signal. Do not tune — diagnose.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
