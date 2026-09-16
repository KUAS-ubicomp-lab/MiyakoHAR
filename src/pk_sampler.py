"""§84 (D26) — the P×K batch sampler and the supervised-contrastive term with CROSS-SUBJECT positives.

PKSampler: every batch draws P distinct classes (uniform over ALL classes present) and, within each, K clips of K DISTINCT
subjects — so every clip has ≥ K − 1 same-class, different-subject positives in its batch. A class with fewer than K
training subjects (fold 0 and fold 2 each have one) keeps its K slots: every one of its subjects once, then DISTINCT
other clips of the class (a repeat only if the class has < K clips) — declared fallback; no class is ever dropped from
training, its clips simply have fewer cross-subject positives.
Steps per epoch = n_items // (P·K), the recipe's count; draws come from a torch.Generator seeded by (seed, epoch), so the
run stays reproducible in the same sense as the rest. The subject field is the labelled TRAIN tree's `user` — the same
field the frozen folds group on — a training-side sampling key, never an inference input.

supcon_loss (Khosla et al., NeurIPS 2020) on L2-normalised projections: positives = same class ∧ DIFFERENT subject
(same-subject pairs are masked out of the numerator, never counted as positives); every other clip in the batch is a
negative in the denominator; anchors without a positive are skipped. τ declared 0.1.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Sampler


class PKSampler(Sampler):
    def __init__(self, labels: list[int], users: list[int], P: int, K: int, seed: int) -> None:
        self.P, self.K, self.seed, self.epoch = P, K, seed, 0
        by = defaultdict(lambda: defaultdict(list))
        for i, (c, u) in enumerate(zip(labels, users)):
            by[int(c)][int(u)].append(i)
        self.by_class = {c: {u: idx for u, idx in d.items()} for c, d in by.items()}
        self.classes = sorted(self.by_class)
        self.short = sorted(c for c, d in self.by_class.items() if len(d) < K)   # classes that need subject repeats
        if len(self.classes) < P:
            raise ValueError(f"only {len(self.classes)} classes present; P = {P} is impossible")
        self.n_batches = len(labels) // (P * K)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1000 + self.epoch)
        self.epoch += 1
        for _ in range(self.n_batches):
            batch = []
            for ci in torch.randperm(len(self.classes), generator=g)[: self.P].tolist():
                d = self.by_class[self.classes[ci]]
                users = sorted(d)
                picks = []
                for ui in torch.randperm(len(users), generator=g)[: self.K].tolist():
                    idx = d[users[ui]]
                    picks.append(idx[int(torch.randint(len(idx), (1,), generator=g))])
                if len(picks) < self.K:                 # the declared fallback for a class with < K subjects:
                    rest = [i for u in users for i in d[u] if i not in picks]   # each subject once, then DISTINCT other
                    perm = torch.randperm(len(rest), generator=g).tolist() if rest else []   # clips of the class; a repeat
                    while len(picks) < self.K:                                   # only if the class has < K clips
                        picks.append(rest[perm.pop()] if perm else picks[int(torch.randint(len(picks), (1,), generator=g))])
                batch.extend(picks)
            yield batch


def supcon_loss(z: torch.Tensor, y: torch.Tensor, u: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """z (B, d) L2-normalised; y, u (B,) class and subject. Same-class ∧ different-subject positives; anchors without one skipped."""
    n = z.shape[0]
    sim = z @ z.T / tau
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & (u[:, None] != u[None, :]) & ~eye
    npos = pos.sum(1)
    has = npos > 0
    if not bool(has.any()):
        return z.new_zeros(())
    per_anchor = -(log_prob.masked_fill(~pos, 0.0).sum(1)[has] / npos[has])
    return per_anchor.mean()
