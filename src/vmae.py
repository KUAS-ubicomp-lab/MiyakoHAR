"""VideoMAE V2 ViT-S/16 (K710-distilled, Kinetics-400 fine-tuned), vendored — QUEUE §68 (D23).

WHY VENDORED AND NOT A DEPENDENCY. The weights are mmaction2's re-host (Apache-2.0) of OpenGVLab's
VideoMAE V2 checkpoint (MIT); mmaction2 pulls mmcv + mmengine + a registry to build one backbone. Same
precedent as src/csn.py: copy-and-trim the structure, take no dependency, name every module so the
checkpoint loads with `strict=True` and ZERO key remapping.

THE KEY NAMES ARE THE CONTRACT. `patch_embed.projection` (Conv3d) · `blocks.{i}.norm1/attn/norm2/mlp`
with `attn.q_bias`, `attn.v_bias`, a bias-free `attn.qkv`, `attn.proj` · mmcv-FFN keys `mlp.layers.0.0`
(fc1) and `mlp.layers.1` (fc2) · `fc_norm`. A remap is a silent-failure surface; a strict load is a proof.

EVERY CONSTANT BELOW WAS READ, NOT ASSUMED (DECISIONS m225, 2026-08-31): the checkpoint's own tensors —
patch_embed.projection.weight (384, 3, 2, 16, 16) ⇒ tubelet 2 × patch 16, embed 384; 12 blocks; qkv
(1152, 384) bias-free + q_bias/v_bias (the VideoMAE convention: k carries a zero bias); mlp 384 → 1536 → 384;
fc_norm (384); NO learnable position embedding and NO cls token in the file — and mmaction2's zoo config
(`vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py` + its base): num_heads 6,
mlp_ratio 4, qkv_bias True, norm LN eps 1e-6, num_frames 16, img_size 224, use_mean_pooling (fc_norm on
the token mean), init_values 0 (no layer-scale gammas — none in the file), fixed sinusoid positions from
mmaction2's `get_sinusoid_encoding`. Normalisation (the zoo's data_preprocessor): mean [123.675, 116.28,
103.53] / std [58.395, 57.12, 57.375] at 0–255 = ImageNet after /255 = this repo's `thermal`/`depthir` keys.

Derived from mmaction2 (https://github.com/open-mmlab/mmaction2, Apache-2.0) and VideoMAEv2
(https://github.com/OpenGVLab/VideoMAEv2, MIT). Structure reimplemented; no mmaction2/mmcv code imported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding(n_position: int, embed_dims: int) -> torch.Tensor:
    """mmaction2's fixed table, verbatim in arithmetic: (1, n_position, embed_dims), float32."""
    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(10000, -vec).view(1, -1)
    table = torch.arange(n_position, dtype=torch.float64).view(-1, 1) * vec
    table[:, 0::2] = table[:, 0::2].sin()
    table[:, 1::2] = table[:, 1::2].cos()
    return table.unsqueeze(0).float()


class DropPath(nn.Module):
    """Stochastic depth per sample (train mode only); parameter-free, so it never touches a key."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    """(B, C, T, H, W) → (B, N, D) with N ordered (T', H', W') — mmcv's `flatten(2).transpose(1, 2)`."""

    def __init__(self, in_channels: int, embed_dims: int, tubelet: int, patch: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(in_channels, embed_dims, kernel_size=(tubelet, patch, patch),
                                    stride=(tubelet, patch, patch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                    # (B, heads, N, d) each
        x = F.scaled_dot_product_attention(q, k, v)             # softmax(q·kᵀ / √d)·v, the same arithmetic
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class FFN(nn.Module):
    """mmcv FFN layout — keys `layers.0.0` (fc1, inside a Sequential with GELU) and `layers.1` (fc2)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.0)),
                                    nn.Linear(hidden, dim), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop_path: float, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = FFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    """The `backbone.*` half of the checkpoint. (B, C, T, H, W) in, (B, embed_dims) out (fc_norm of the mean token)."""

    def __init__(self, in_channels: int = 3, num_frames: int = 16, img_size: int = 224, patch_size: int = 16,
                 tubelet_size: int = 2, embed_dims: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, drop_path_rate: float = 0.0, norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dims, tubelet_size, patch_size)
        self.grid = (num_frames // tubelet_size, img_size // patch_size, img_size // patch_size)
        n_pos = self.grid[0] * self.grid[1] * self.grid[2]
        # Fixed (non-learnable) sinusoid positions; persistent=False so the buffer is neither expected from
        # the checkpoint (it has none) nor written into ours — it is recomputed from the formula at build.
        self.register_buffer("pos_embed", get_sinusoid_encoding(n_pos, embed_dims), persistent=False)
        dpr = [float(v) for v in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(embed_dims, num_heads, mlp_ratio, dpr[i], norm_eps) for i in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dims, eps=norm_eps)
        self.embed_dims, self.depth = embed_dims, depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                                 # (B, N, D)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(f"token count {x.shape[1]} != the position table's {self.pos_embed.shape[1]} — "
                             f"this backbone was built for grid {self.grid} (T×H×W after tubelet/patch); "
                             "build it with the input's num_frames/img_size rather than interpolating silently")
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.fc_norm(x.mean(dim=1))


def strip_backbone(sd: dict) -> tuple[dict, dict]:
    """Split a checkpoint into (backbone.* without the prefix, cls_head.*). Anything else RAISES."""
    backbone, head = {}, {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            backbone[k[len("backbone."):]] = v
        elif k.startswith("cls_head."):
            head[k] = v
        else:
            raise KeyError(f"unexpected checkpoint key {k!r} — not backbone.* / cls_head.*")
    return backbone, head


def truncate_backbone(backbone_sd: dict, depth: int) -> tuple[dict, list[str]]:
    """§90 (D27): keep blocks 0 … depth−1 of a stripped backbone state (patch embed and fc_norm untouched); returns
    (kept, the dropped tensor names in checkpoint order). The caller PRINTS the dropped names — the load test's evidence
    that exactly the dropped block's tensors are absent. RAISES unless 0 < depth < the state's own block count
    (dropping nothing is not a truncation; a depth the state cannot supply is a wiring error, never a silent default)."""
    n_blocks = 1 + max(int(k.split(".")[1]) for k in backbone_sd if k.startswith("blocks."))
    if not 0 < depth < n_blocks:
        raise ValueError(f"depth must lie in [1, {n_blocks - 1}] to truncate a {n_blocks}-block backbone, got {depth}")
    kept, dropped = {}, []
    for k, v in backbone_sd.items():
        if k.startswith("blocks.") and int(k.split(".")[1]) >= depth:
            dropped.append(k)
        else:
            kept[k] = v
    return kept, dropped
