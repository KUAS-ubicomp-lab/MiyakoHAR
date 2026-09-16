"""Build the REAL fp16 submission artefact, measure it, and score it against fp32.

WHY THIS EXISTS: NO fp16 ARTEFACT HAS EVER BEEN BUILT. Every packaged figure
in this project -- D6's 95 MB budget, measurement 55's cap curve, the 44.8 MB
that predict.py prints and refuses to exceed, the 89.6 MB and 134.4 MB rows --
is `params x 2`, computed from a parameter count. Not one of them has ever been
weighed, and none of them has ever been scored. They are estimates presented as
measurements, and the constraint they estimate is the one that decides what
ships.

TWO WAYS `params x 2` IS WRONG, AND THEY PULL IN OPPOSITE DIRECTIONS:

  · IT UNDERCOUNTS. A state dict is not only parameters. BatchNorm carries
    running_mean, running_var and num_batches_tracked as BUFFERS; the first two
    halve under fp16 and `num_batches_tracked` is int64 and does NOT. Then
    torch.save writes a zip container with a pickle header per tensor. A
    reviewer who built the 6-member s3d artefact rather than multiplying got
    96.38 MB against a predicted 95.4 -- past D6's ceiling, on a cell that had
    been marked legal.
  · IT OVERCOUNTS what we actually have to ship, because nothing forces the
    optimiser state, the epoch counter or the val_acc into the artefact.

Which of the two dominates is an empirical question about OUR checkpoints, and
this file answers it by weighing them.

AND THE ACCURACY SIDE IS NOT FREE EITHER. TECHNICAL.md §F5 already flagged BN
`running_var` in fp16 as a genuine range hazard: measured min-nonzero 2.76e-14,
which is subnormal in fp16 and flushes toward zero, and 1/sqrt(var) is what that
value feeds. s3d carries 77 BatchNorm3d against ResNet-18's 20 -- 3.85x the
exposure -- so "fp16 is free" is a claim that has to be re-checked per backbone,
not inherited.

Usage:
    ./.venv/bin/python tools/package.py --checkpoints 'runs/depthir_f0_alien_ab5_lrhi_s1' ...
    ./.venv/bin/python tools/package.py --shipping            # predict.py's default set
    ./.venv/bin/python tools/package.py --shipping --score --manifest /tmp/m_derived.csv
    ./.venv/bin/python tools/package.py --shipping --out checkpoints/model.pth   # the R4 container —
        # `model.pth` is the site's deliverable name (P0 a, 2026-08-31); predict.py/verify.sh
        # also accept the legacy `model.pt`. Writing INTO the slot is an R4 act (Arthur + the path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.model import DEFAULT_ARCH, adapt_kw, build, norm_key  # noqa: E402

# What predict.py loads when nobody says otherwise. Kept in one place so the
# thing we weigh and the thing we ship cannot drift.
# m72: this was TWO checkpoints while predict.py shipped FOUR, so
# `--shipping` had not weighed the set that actually ships since submission 003.
# There is now ONE definition and the packer imports it -- `--shipping` cannot
# drift from what predict.py loads again.
sys.path.insert(0, str(ROOT / "src"))
from predict import SHIPPING, input_size_of, parse_fusion  # noqa: E402


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fusion_from_json(path: Path | None) -> dict | None:
    """§119 (D50; P-A): read the fusion stamp from a JSON file and VALIDATE it through the loader's own parser.

    ONE definition of what a legal fusion stamp is — predict.parse_fusion — imported here rather than re-stated, the
    m72 lesson. A packer that could write a stamp predict.py refuses is a packer that can build an artefact nothing
    can load; this raises at pack time instead, before any byte is written.
    """
    if path is None:
        return None
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    parse_fusion(spec, f"{path} (--fusion-json)")
    return spec


def source_wh_from_args(specs: list[str] | None) -> dict[str, tuple[int, int]]:
    """§124 (P-A): parse `--source-wh BRANCH=WxH` (repeatable) into {branch: (W, H)}.

    Per BRANCH, not blanket like --fusion-json, because it is a property of the data the MEMBER trained on and a
    §124 container mixes them: the thermal member was decoded at 320x240 (src/preprocess.py --size-wh 320x240,
    cache_hr) while the depth+IR member was decoded at the shipped 160x120. A branch not named here is stamped
    None, which is the shipped decode at load — so omitting the flag entirely rebuilds today's containers bitwise.
    """
    out: dict[str, tuple[int, int]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"🔴 --source-wh {spec!r}: expected BRANCH=WxH, e.g. thermal=320x240")
        branch, _, wh = spec.partition("=")
        try:
            w, h = (int(v) for v in wh.lower().split("x"))
        except ValueError:
            raise SystemExit(f"🔴 --source-wh {spec!r}: {wh!r} is not WxH") from None
        if w < 1 or h < 1:
            raise SystemExit(f"🔴 --source-wh {spec!r}: both dimensions must be ≥ 1")
        if branch in out and out[branch] != (w, h):
            raise SystemExit(f"🔴 --source-wh names branch {branch!r} twice with different sizes")
        out[branch] = (w, h)
    return out


def pack(ckpts: list[Path], out: Path, half: bool, fusion: dict | None = None,
         source_wh: dict[str, tuple[int, int]] | None = None) -> dict:
    """Write the artefact exactly as it would be submitted, and weigh it.

    Only what inference needs: the tensors, plus the metadata predict.py
    dispatches on. Not the optimiser, not val_acc, not the epoch counter.
    """
    members, n_params, n_buf, n_int = [], 0, 0, 0
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"]
        cast = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                cast[k] = v
                continue
            if v.is_floating_point():
                cast[k] = v.half() if half else v.float()
            else:
                # num_batches_tracked is int64 and does NOT halve. This is
                # one of the two things `params x 2` silently drops.
                n_int += v.numel() * v.element_size()
                cast[k] = v
        members.append({
            "model": cast, "branch": ck["branch"], "T": ck["T"],
            "in_channels": ck["in_channels"],
            "arch": ck.get("arch", DEFAULT_ARCH),
            "norm": ck.get("norm") or norm_key(ck["branch"], ck.get("arch", DEFAULT_ARCH)),
            "input_adapt": ck.get("input_adapt"),   # §63: the adapter stamp travels into the container
            # §96 (a)/(b) (D29): the depth inverse-LUT, a fold-TRAIN statistic stamped into the member checkpoint (L7's second
            # clause), travels into the container as a fixed key — never a repo constant (.gitignore:1-2; m236). None for every
            # member that carries none. predict.py applies it per clip (R8) once the ordinal decode lands; until then a LUT-bearing
            # member is REFUSED at load, so the key can never be a silent passenger.
            "depth_lut": ck.get("depth_lut"),
            "crops": ck.get("crops"),               # §105 (P-A): the stamped crop count travels; None when absent (= 1 at load)
            # §119 (D50; P-A): the SECOND stamped key. It is a property of the ARTEFACT's decode, not of the member's
            # training, so unlike `crops` it comes from --fusion-json rather than from the run checkpoint — and it is
            # written on EVERY member, identical, because predict.py refuses a container whose members disagree.
            # None (the default, and every container built before 2026-09-12) ⇒ the arithmetic mean at load.
            "fusion": fusion,
            # §124 (P-A): the THIRD stamped decode key — the SOURCE resolution this member's frames were decoded at
            # before the resize to input_size. Like `fusion` it comes from the CLI (train.py stamps no cache
            # resolution), but unlike `fusion` it is PER BRANCH, because a §124 container mixes a thermal member
            # decoded at 320x240 with a depth+IR member decoded at the shipped 160x120. None ⇒ the shipped decode at
            # load, and every container built before 2026-09-13 carries None on every member.
            "source_wh": (source_wh or {}).get(ck["branch"]),
            "run": p.parent.name,
            # B1's SECOND DEFECT, AND IT IS m73 ALL OVER AGAIN. Until this
            # line the container carried arch and norm but NOT the resolution --
            # so a member trained at 168x224 loaded from the container would have
            # run at the cache's native 120x160. predict.input_size_of falls back
            # to matching `path.parent.name` against experiments-*.jsonl, and a
            # container's parent is a directory, not a run, so the fallback finds
            # nothing and returns native. That does not raise. It costs ~4 pp,
            # silently, which is exactly how m73 reached submission 004.
            # Resolved HERE, at pack time, where the run directory still exists.
            "input_size": input_size_of(ck, p),
        })
        m = build(members[-1]["branch"], n_segment=ck["T"], arch=members[-1]["arch"],
                  pretrained=False, **adapt_kw(ck))   # §63
        # FM3 (GOAL-SPRINT-4 §7): the key allowlist -- a container may hold ONLY the tensors of the member
        # module that ships; a foreign (teacher) tensor is a hard STOP, never a silent passenger.
        foreign, missing = set(cast) - set(m.state_dict()), set(m.state_dict()) - set(cast)
        if foreign or missing:
            raise SystemExit(f"🔴 FM3: {p} — keys not owned by the member module {sorted(foreign)[:5]} / "
                             f"module keys absent {sorted(missing)[:5]} — STOP, nothing packed")
        n_params += sum(q.numel() for q in m.parameters())
        n_buf += sum(b.numel() for b in m.buffers())

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(members, out)
    size = out.stat().st_size
    return {"members": len(members), "params": n_params, "buffers": n_buf,
            "int_bytes": n_int, "bytes": size, "sha256": sha256_of(out),
            "predicted": n_params * (2 if half else 4)}


def score(ckpts: list[Path], half: bool, cache: Path, manifest: Path, workers: int) -> dict:
    """Val accuracy of the packed weights, per fold, on that fold's own subjects."""
    from torch.utils.data import DataLoader

    from src.dataset import ClipDataset, collate

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_fold: dict[int, list] = {}
    for p in ckpts:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        arch = ck.get("arch", DEFAULT_ARCH)
        nk = ck.get("norm") or norm_key(ck["branch"], arch)
        m = build(ck["branch"], n_segment=ck["T"], arch=arch, pretrained=False, **adapt_kw(ck))   # §63
        m.load_state_dict(ck["model"])
        if half:
            # Cast exactly as the artefact stores it, then run in fp32 so that
            # the measurement isolates PRECISION LOSS IN THE STORED WEIGHTS from
            # precision loss in the arithmetic. Those are different questions and
            # only the first is what the packaged file commits us to.
            m.half().float()
        m.eval().to(dev)
        # m73: same defect as src/predict.py had — this scored the 168x224
        # checkpoints at the cache's native 120x160 and reported the result as
        # the packaged accuracy. The fp16-vs-fp32 DELTA survived it (both arms
        # ran at the same wrong scale) but the absolute numbers did not.
        ds = ClipDataset(cache, manifest, ck["branch"], "train", "val",
                         fold=ck["fold"], T=ck["T"], norm=nk,
                         input_size=input_size_of(ck, p))
        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=workers,
                        pin_memory=True, collate_fn=collate)
        probs, ys, ids = [], [], []
        with torch.no_grad():
            for b in dl:
                x = b["x"].to(dev, non_blocking=True)
                xb = torch.cat([x, torch.flip(x, dims=[-1])], 0)   # hflip TTA, as shipped
                o = torch.softmax(m(xb).float(), 1)
                probs.append(((o[: len(x)] + o[len(x):]) / 2).cpu())
                ys.append(b["y"])
                ids.extend(b["clip_id"])
        per_fold.setdefault(ck["fold"], []).append(
            (ids, torch.cat(probs), torch.cat(ys)))
        del m
        torch.cuda.empty_cache()

    out = {}
    for f, members in sorted(per_fold.items()):
        common = sorted(set.intersection(*[set(i) for i, _p, _y in members]))
        stack, lab = [], {}
        for ids, pr, yy in members:
            ix = {c: i for i, c in enumerate(ids)}
            stack.append(torch.stack([pr[ix[c]] for c in common]))
            lab.update({c: int(yy[ix[c]]) for c in common})
        y = torch.tensor([lab[c] for c in common])
        out[f] = float((torch.stack(stack).mean(0).argmax(1) == y).float().mean())
    return out


def resolve(specs: list[str]) -> list[Path]:
    from predict import find_checkpoints
    return find_checkpoints([str(ROOT / s) if not Path(s).is_absolute() else s for s in specs])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="*", default=None)
    ap.add_argument("--shipping", action="store_true", help="predict.py's default set")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write the fp16 container; the shippable slot is checkpoints/model.pth")
    ap.add_argument("--score", action="store_true", help="also measure fp16 vs fp32 accuracy")
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m_derived.csv"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-mb", type=float, default=95.0, help="D6's packaged ceiling")
    ap.add_argument("--source-wh", action="append", default=None, metavar="BRANCH=WxH",
                    help="§124 (P-A): stamp the SOURCE decode resolution of one branch's members, e.g. "
                         "thermal=320x240 (repeatable). Omitted ⇒ every member is stamped None ⇒ the shipped "
                         "160x120 decode at load, bitwise as before.")
    ap.add_argument("--fusion-json", type=Path, default=None,
                    help="§119 (P-A): stamp a fusion formula into every member — a JSON "
                         '{"form": "logit_adjusted_geo", "lambda": 0.25, "prior": [40 floats]}. '
                         "Omitted ⇒ no stamp ⇒ the arithmetic mean, bitwise the shipped decode.")
    args = ap.parse_args()
    fusion = fusion_from_json(args.fusion_json)     # raises here, before anything is packed
    src_wh = source_wh_from_args(args.source_wh)   # raises here too, before anything is packed

    specs = SHIPPING if (args.shipping or not args.checkpoints) else args.checkpoints
    ckpts = resolve(specs)
    if not ckpts:
        raise SystemExit(f"no checkpoint matched {specs}")
    # D20 / m203: a SHIPPING entry that resolves to NOTHING must stop the packer, not
    # shrink the artefact. On 2026-08-29 the first repack of 022 from SHIPPING silently
    # produced a ONE-member 58.5 MB file because the depth+IR member's run directory did
    # not exist on the box — the m59 class (a green step that cannot fail). Each spec
    # must resolve to exactly one checkpoint.
    if len(ckpts) != len(specs):
        missing = [s_ for s_ in specs if not any(str(c).startswith(str(ROOT / s_) if not Path(s_).is_absolute() else s_) for c in ckpts)]
        raise SystemExit(f"🔴 {len(specs)} specs resolved to {len(ckpts)} checkpoints — missing: {missing}. STOP.")

    print("=" * 78)
    print("  THE REAL fp16 ARTEFACT — weighed, not multiplied")
    print("=" * 78)
    if fusion is not None:
        print(f"  §119 fusion stamp: {fusion['form']} λ={fusion['lambda']} over a {len(fusion['prior'])}-class TRAIN prior "
              f"(from {args.fusion_json})")
    if src_wh:
        print("  §124 source_wh stamp: " + " · ".join(f"{b} at {w}x{h}" for b, (w, h) in sorted(src_wh.items()))
              + " (every other branch: None = the shipped decode)")
    for c in ckpts:
        print(f"  · {c.parent.name}/{c.name}")

    tmp = Path(tempfile.mkdtemp())
    rows = {}
    for half in (False, True):
        out = args.out if (args.out and half) else tmp / f"pack_{'fp16' if half else 'fp32'}.pt"
        rows["fp16" if half else "fp32"] = pack(ckpts, out, half, fusion, src_wh)

    a, b = rows["fp32"], rows["fp16"]
    print(f"\n  {'':<10}{'MEASURED':>12} {'params x N':>12} {'delta':>10} {'sha256':>18}")
    print("  " + "-" * 66)
    for name, r in (("fp32", a), ("fp16", b)):
        print(f"  {name:<10}{r['bytes']/1e6:>11.2f}M {r['predicted']/1e6:>11.2f}M "
              f"{(r['bytes']-r['predicted'])/1e6:>+9.2f}M   {r['sha256'][:16]}")
    print(f"\n  members {b['members']} · params {b['params']/1e6:.2f} M · "
          f"buffers {b['buffers']/1e3:.1f} k · int64 buffer bytes {b['int_bytes']}")
    print(f"  fp16 is {a['bytes']/b['bytes']:.3f}x smaller than fp32 "
          f"(2.000x would be the naive expectation)")
    legal = b["bytes"] / 1e6 <= args.max_mb
    print(f"\n  🔴 D6 ceiling {args.max_mb:.0f} MB packaged → MEASURED {b['bytes']/1e6:.2f} MB "
          f"→ {'LEGAL' if legal else 'OVER'}")
    err = b["bytes"] / 1e6 - b["predicted"] / 1e6
    print(f"  the ledger's estimate was {b['predicted']/1e6:.2f} MB, so it was off by "
          f"{err:+.2f} MB ({100*err/(b['predicted']/1e6):+.1f}%)")
    if args.out:
        print(f"  wrote {args.out}")

    if args.score:
        # J9/m89: an all-subjects checkpoint TRAINED ON the very subjects this
        # function scores it against — every fold's val set is inside its training
        # set. The fp16-vs-fp32 DELTA survives that (both arms run on identical
        # data, exactly as the delta survived m73's wrong resolution above), but
        # the ABSOLUTE columns are in-sample and would print in the same shape as
        # a real packaged CV. That is how a §11.3 violation gets made by accident,
        # so the absolutes are labelled at the point of printing rather than in a
        # comment nobody reads at 2 a.m.
        in_sample = [p for p in ckpts
                     if torch.load(p, map_location="cpu", weights_only=False).get("all_subjects")]
        print("\n  " + "-" * 66)
        print("  ACCURACY: does fp16 storage cost anything? (per fold, own val subjects)")
        if in_sample:
            print(f"  🔴 {len(in_sample)} of {len(ckpts)} checkpoints carry all_subjects=True.")
            print("  THE ABSOLUTE COLUMNS BELOW ARE IN-SAMPLE AND ARE NOT A CV. Those models")
            print("  trained on every fold's val subjects, so these numbers are inflated by")
            print("  construction and MUST NOT be quoted as packaged CV anywhere (§11.3).")
            print("  The fp16−fp32 DELTA is unaffected: both arms score identical data.")
        f32 = score(ckpts, False, args.cache, args.manifest, args.workers)
        f16 = score(ckpts, True, args.cache, args.manifest, args.workers)
        tag = " IN-SAMPLE" if in_sample else ""
        print(f"\n  {'fold':>6} {'fp32':>10} {'fp16':>10} {'delta pp':>10}{tag}")
        for f in sorted(f32):
            print(f"  f{f:<5} {100*f32[f]:>10.3f} {100*f16[f]:>10.3f} "
                  f"{100*(f16[f]-f32[f]):>+10.3f}{tag}")
        m32 = sum(f32.values()) / len(f32)
        m16 = sum(f16.values()) / len(f16)
        label = "MEAN" if in_sample else "MEAN"
        print(f"  {label:>6} {100*m32:>10.3f} {100*m16:>10.3f} {100*(m16-m32):>+10.3f}{tag}")
        print(f"\n  → fp16 storage costs {100*(m32-m16):+.3f} pp"
              f"{'  (free)' if abs(m16-m32) < 1e-4 else ''}")
        if in_sample:
            print("  → and the MEAN above is NOT this artefact's CV. An all-subjects artefact")
            print("     has no CV by construction (m87 ④); it can only be predicted and then")
            print("     verified on the board.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
