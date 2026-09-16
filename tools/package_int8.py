"""§88 (2) (m251; the laptop pass) — the torch.ao per-channel int8 STORAGE weigh-in: the real two-member byte count, and
the int8-vs-fp16 fold accuracy delta measured exactly as package.py measures fp16-vs-fp32 (storage round-trip isolated
from arithmetic; hflip TTA; fused per fold on the common clips).

DECLARED IMPLEMENTATION READING (QUEUE §88): storage quantisation under D6's torch.ao clause — every weight tensor of
ndim ≥ 2 is stored `torch.quantize_per_channel(axis 0, qint8)` with fp32-scale doubles; biases/norm affines/vectors fp16;
int64 buffers untouched (package.py's lesson); load = dequantise → the shipped inference path. Static-PTQ int8 COMPUTE is
not attempted (the byte question does not need it); no bespoke packing; D6's sub-int8 prohibition untouched; the shipped
packer (tools/package.py) and slot are NOT modified by this tool.

  --weigh RUN [RUN ...]   pack those members int8 → the exact byte count (the §88 (2) number when given BOTH members)
  --vmae-depth N          §90 (1) (D27): a `vmae2_vitb` member is packed with blocks N…11 REMOVED (the §74 arm-(b) weights as the
                          stand-in — identical architecture ⇒ identical bytes), its arch re-stamped `vmae2_vitbN`, the dropped names
                          printed; every member is then loaded STRICT (after dequantisation) on the module its arch names.
  --vit-per-tensor        D28 amendment to §88 (2): the vmae2_vit* member(s) stored torch.ao PER-TENSOR int8 (one scale + one
                          zero-point per weight tensor — still torch.ao int8, never bespoke; D27 (a)(iii) untouched); --all-per-tensor
                          applies it to every member. Removes the per-channel scale/zero-point overhead (≈ 16 B per output channel).
  --score --fold N        the fold's two members (R50 d+IR s1 + the thermal prefix's s1), int8 round-trip vs fp16 round-trip, CPU
                          (the same storage flags apply, so the packaging gate measures the scheme that would ship)
Usage: .venv/bin/python tools/package_int8.py --weigh runs/depthir_all18_alien_q16_ircsn224_s1 runs/labpc_vmaebw_thermal_f0_s1
       .venv/bin/python tools/package_int8.py --weigh runs/depthir_all18_alien_q16_ircsn224_s1 runs/labpc_vmaebw_thermal_f0_s1 --vmae-depth 11 --out runs/p8_int8_vitb11.pt
       .venv/bin/python tools/package_int8.py --score --fold 0 [--thermal-prefix labpc_vmaeb11w_thermal] [--limit N] [--workers 3]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from src.model import DEFAULT_ARCH, adapt_kw, build, norm_key      # noqa: E402
from src.vmae import truncate_backbone                             # noqa: E402
from tools.package import fusion_from_json                        # noqa: E402  (§119: ONE definition of the stamp)


def int8_state(sd: dict, per_tensor=False) -> tuple[dict, int, int]:
    """Weights (ndim ≥ 2, floating) → per-channel qint8, or per-tensor qint8 where `per_tensor` says so (a bool for every
    tensor, or a callable key → bool — the §93 amendment-2 mixed storage); other floats → fp16; ints untouched.
    Returns (state, nq, nf)."""
    out, nq, nf = {}, 0, 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            # §94 (D29) / FM3: the allowlist is over keys AND values — a member state holds only the member module's own tensors;
            # a provenance string (or any non-tensor) is a hard STOP, never a silent passenger (tests/check_int8_container.py §3).
            raise SystemExit(f"🔴 FM3: {k!r} is not a tensor ({type(v).__name__}) — a member state holds only the member module's tensors; STOP, nothing packed")
        if torch.is_floating_point(v) and v.ndim >= 2:
            if (per_tensor(k) if callable(per_tensor) else per_tensor):
                scale = max(float(v.detach().abs().amax()) / 127.0, 1e-12)
                out[k] = torch.quantize_per_tensor(v.detach().float(), scale, 0, torch.qint8)
            else:
                vmax = v.detach().abs().amax(dim=tuple(range(1, v.ndim)))
                scale = (vmax / 127.0).clamp(min=1e-12).double()
                zp = torch.zeros_like(scale, dtype=torch.int64)
                out[k] = torch.quantize_per_channel(v.detach().float(), scale, zp, 0, torch.qint8)
            nq += 1
        elif torch.is_floating_point(v):
            out[k] = v.detach().half(); nf += 1
        else:
            out[k] = v.detach()
    return out, nq, nf


def deq_state(sd: dict) -> dict:
    return {k: (v.dequantize() if v.is_quantized else (v.float() if torch.is_floating_point(v) else v)) for k, v in sd.items()}


def find(spec: str) -> Path:
    from predict import find_checkpoints
    c = find_checkpoints([str(ROOT / spec) if not Path(spec).is_absolute() else spec])
    if len(c) != 1:
        raise SystemExit(f"🔴 {spec} resolved to {len(c)} checkpoints — STOP")
    return c[0]


def per_tensor_for(arch: str, vit_pt: bool, all_pt: bool, mlp_pt: bool = False, csn_pt: bool = False):
    """The storage rule for one member: True/False for every tensor, or a per-key callable (variant iii: the ViT-B's MLP
    tensors per-tensor, its attention + patch-embedding per-channel; every ir-CSN tensor per-tensor)."""
    if all_pt or (vit_pt and arch.startswith("vmae2_vit")) or (csn_pt and arch.startswith("ircsn")):
        return True
    if mlp_pt and arch.startswith("vmae2_vit"):
        return lambda k: ".mlp." in k
    return False


def member_of(p: Path, vmae_depth: int | None = None, vit_pt: bool = False, all_pt: bool = False,
              mlp_pt: bool = False, csn_pt: bool = False, fusion: dict | None = None,
              source_wh: dict | None = None) -> dict:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd, arch, dropped = ck["model"], ck.get("arch", DEFAULT_ARCH), []
    if vmae_depth is not None and arch == "vmae2_vitb":
        # §90 (1): blocks vmae_depth…11 removed from the 12-block member; a trained ViT-B[depth] cell already carries
        # arch vmae2_vitb{depth} and passes through untouched.
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        _kept, dropped = truncate_backbone(bb, vmae_depth)
        dropped = [f"backbone.{k}" for k in dropped]
        sd = {k: v for k, v in sd.items() if k not in set(dropped)}
        arch = f"vmae2_vitb{vmae_depth}"
    pt = per_tensor_for(arch, vit_pt, all_pt, mlp_pt, csn_pt)
    q, nq, nf = int8_state(sd, pt)
    n_pt = sum(1 for k, v in q.items() if getattr(v, "is_quantized", False) and v.qscheme() == torch.per_tensor_affine)
    # The load test: the container's member must load STRICT, after dequantisation, on the module its arch names --
    # FM3's key allowlist in the same line: a foreign (teacher) tensor is an unexpected key and a hard STOP.
    m = build(ck["branch"], n_segment=ck["T"], arch=arch, pretrained=False, **adapt_kw(ck))
    try:
        m.load_state_dict(deq_state(q), strict=True)
    except RuntimeError as e:
        raise SystemExit(f"🔴 FM3: {p} — the member state does not match the module it names ({str(e).splitlines()[0][:160]}) — STOP, nothing packed")
    return {"branch": ck["branch"], "arch": arch, "T": ck["T"], "fold": ck.get("fold"),
            "in_channels": ck.get("in_channels"), "run": p.parent.name,
            "norm": ck.get("norm") or norm_key(ck["branch"], arch),
            "input_size": ck.get("input_size"), "input_adapt": ck.get("input_adapt"),
            "freeze_stages": ck.get("freeze_stages"),
            "depth_lut": ck.get("depth_lut"),          # §96 (a)/(b): the fold-TRAIN inverse-LUT stamp travels; None when absent
            "crops": ck.get("crops"),                  # §105 (P-A): the stamped crop count travels; None when absent (= 1 at load)
            "fusion": fusion,                          # §119 (P-A): the SECOND stamped key — the artefact's decode rule, from
                                                       # --fusion-json (never from the run ck); None ⇒ the arithmetic mean at load
            # §124 (P-A): the THIRD stamped key — the SOURCE decode resolution, PER BRANCH (a §124 container mixes a
            # thermal member decoded at 320x240 with a depth+IR member at the shipped 160x120). None ⇒ the shipped decode.
            "source_wh": (source_wh or {}).get(ck["branch"]),

            "quant": ("torch.ao-per-tensor-int8-storage" if n_pt == nq else "torch.ao-per-channel-int8-storage" if n_pt == 0
                      else f"torch.ao-mixed-int8-storage({n_pt}-per-tensor/{nq - n_pt}-per-channel)"),
            "src": str(p), "model": q, "_nq": nq, "_nf": nf, "_dropped": dropped, "_n_loaded": len(q), "_pt": n_pt}


@torch.no_grad()
def fold_probs(ck_path: Path, arm: str, cache: Path, manifest: Path, workers: int, limit: int | None,
               vit_pt: bool = False, all_pt: bool = False, mlp_pt: bool = False, csn_pt: bool = False):
    from torch.utils.data import DataLoader
    from src.dataset import ClipDataset, collate
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", DEFAULT_ARCH)
    m = build(ck["branch"], n_segment=ck["T"], arch=arch, pretrained=False, **adapt_kw(ck))
    if arm == "int8":
        pt = per_tensor_for(arch, vit_pt, all_pt, mlp_pt, csn_pt)
        print(f"[int8]   {ck_path.parent.name}: {'per-tensor' if pt is True else 'mixed' if callable(pt) else 'per-channel'} int8 round-trip")
        m.load_state_dict(deq_state(int8_state(ck["model"], pt)[0]))
    else:
        m.load_state_dict(ck["model"]); m.half().float()
    m.eval()
    size = tuple(int(v) for v in ck["input_size"].split("x")) if ck.get("input_size") else None
    nk = ck.get("norm") or norm_key(ck["branch"], arch)
    ds = ClipDataset(cache, manifest, ck["branch"], "train", "val", fold=ck["fold"], T=ck["T"], norm=nk, input_size=size)
    if limit:
        ds.items = ds.items[:limit]
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=workers, collate_fn=collate)
    probs, ys, ids = [], [], []
    for b in dl:
        x = b["x"]
        xb = torch.cat([x, torch.flip(x, dims=[-1])], 0)
        o = torch.softmax(m(xb).float(), 1)
        probs.append((o[: len(x)] + o[len(x):]) / 2)
        ys.append(b["y"]); ids.extend(b["clip_id"])
    return ids, torch.cat(probs), torch.cat(ys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weigh", nargs="*", default=None, metavar="RUN")
    ap.add_argument("--shipping", action="store_true", help="pack predict.SHIPPING (the R4 build of an int8 candidate)")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "p7_int8_container.pt")
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/m_derived.csv"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N val clips")
    ap.add_argument("--vmae-depth", type=int, default=None, help="§90 (1): weigh a vmae2_vitb member with blocks N…11 removed (arch → vmae2_vitbN)")
    ap.add_argument("--thermal-prefix", default="labpc_vmaebw_thermal", help="--score: the thermal member's run prefix")
    ap.add_argument("--vit-per-tensor", action="store_true", help="D28 / §88 (2) amendment: vmae2_vit* member(s) stored per-tensor int8")
    ap.add_argument("--all-per-tensor", action="store_true", help="every member stored per-tensor int8")
    ap.add_argument("--vit-mlp-per-tensor", action="store_true", help="§93 amendment 2 (variant iii): the ViT-B's .mlp. tensors per-tensor, its attention + patch-embed per-channel")
    ap.add_argument("--csn-per-tensor", action="store_true", help="§93 amendment 2 (variant iii): every ir-CSN tensor per-tensor")
    ap.add_argument("--fusion-json", type=Path, default=None,
                    help="§119 (P-A): stamp a fusion formula into every member — the same JSON tools/package.py takes; "
                         "omitted ⇒ no stamp ⇒ the arithmetic mean, bitwise the shipped decode")
    a = ap.parse_args()
    fusion = fusion_from_json(a.fusion_json)     # §119: validated through predict.parse_fusion before anything is packed
    if not a.out.is_absolute():
        a.out = ROOT / a.out
    if a.shipping:
        from predict import SHIPPING
        a.weigh = list(SHIPPING)
    if a.weigh:
        members = []
        for spec in a.weigh:
            p = find(spec)
            mm = member_of(p, a.vmae_depth, a.vit_per_tensor, a.all_per_tensor, a.vit_mlp_per_tensor, a.csn_per_tensor, fusion)
            print(f"[int8] {spec} → {p.name}: {mm['branch']}/{mm['arch']} T={mm['T']} · {mm['_nq']} tensors int8 "
                  f"({mm['_pt']} per-tensor / {mm['_nq'] - mm['_pt']} per-channel) · {mm['_nf']} fp16")
            if mm["_dropped"]:
                print(f"[int8]   ViT-B[{a.vmae_depth}] stand-in: {len(mm['_dropped'])} tensors dropped by name: {mm['_dropped']}")
            print(f"[int8]   strict load OK on {mm['arch']}: {mm['_n_loaded']} tensors consumed after dequantisation")
            members.append({k: v for k, v in mm.items() if not k.startswith('_')})
        torch.save(members, a.out)
        n = a.out.stat().st_size
        sha = hashlib.sha256(a.out.read_bytes()).hexdigest()[:16]
        print(f"[int8] WEIGHED: {len(members)} member(s) → {a.out.relative_to(ROOT)} = {n:,} B ({n / 1e6:.2f} MB decimal) · sha16 {sha}")
        if len(members) != 2:
            print("[int8] (a one-member pack is machinery validation only — the §88/§90 number is the TWO-member file)")
        elif a.vmae_depth is not None:
            if n < 95_000_000:
                verdict = f"🟢 < 95,000,000 B by {95_000_000 - n:,} B — depth {a.vmae_depth} is the arm"
            elif a.vmae_depth > 10:
                verdict = f"🔴 ≥ 95,000,000 B by {n - 95_000_000:,} B — re-weigh at depth 10"
            else:
                verdict = f"🔴 ≥ 95,000,000 B by {n - 95_000_000:,} B at depth 10 — §90 CLOSES ON BYTES"
            print(f"[int8] the §90 (1) gate at depth {a.vmae_depth}: {verdict}")
        else:
            print(f"[int8] the §88 (2) gate: {'≥ 100,000,000 B — §88 CLOSES ON BYTES' if n >= 100_000_000 else 'under 100,000,000 B by ' + format(100_000_000 - n, ',') + ' B'}")
        return 0
    if a.score:
        assert a.fold is not None
        pair = [find(f"runs/labpc_s3_depthir_f{a.fold}_t16"), find(f"runs/{a.thermal_prefix}_f{a.fold}_s1")]
        res = {}
        for arm in ("fp16", "int8"):
            mem = [fold_probs(p, arm, a.cache, a.manifest, a.workers, a.limit, a.vit_per_tensor, a.all_per_tensor, a.vit_mlp_per_tensor, a.csn_per_tensor) for p in pair]
            common = sorted(set(mem[0][0]) & set(mem[1][0]))
            stack, lab = [], {}
            for ids, pr, yy in mem:
                ix = {c: i for i, c in enumerate(ids)}
                stack.append(torch.stack([pr[ix[c]] for c in common]))
                lab.update({c: int(yy[ix[c]]) for c in common})
            y = torch.tensor([lab[c] for c in common])
            res[arm] = 100.0 * float((torch.stack(stack).mean(0).argmax(1) == y).float().mean())
            print(f"[int8] fold {a.fold} {arm}: fused {res[arm]:.3f} % on {len(common)} common clips" + (f" (limit {a.limit} — SMOKE, not a reading)" if a.limit else ""))
        print(f"[int8] fold {a.fold} Δ(int8 − fp16) = {res['int8'] - res['fp16']:+.3f} pp   (D6 packaging gate: ≥ −0.2 pp)")
        return 0
    ap.print_help(); return 2


if __name__ == "__main__":
    raise SystemExit(main())
