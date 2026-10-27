"""Original FeatLens 0.3.2 kernels; see THIRD_PARTY_NOTICES.md for MIT license.
Source: https://github.com/turhancan97/FeatLens
Commit: 07eddd718a5a53d11ca5cc21e69a21d0f7b9f2c3
Only import context is reduced; function bodies are unchanged.
"""
from typing import Sequence, Tuple
import numpy as np
import torch

def seed_to_cell(seed_xy: Sequence[float], h: int, w: int) -> Tuple[int, int]:
    """Normalized image coords ``(x, y) in [0, 1]`` -> grid cell ``(row, col)``."""
    x, y = float(seed_xy[0]), float(seed_xy[1])
    r = min(max(int(y * h), 0), h - 1)
    c = min(max(int(x * w), 0), w - 1)
    return r, c

def cosine_similarity_map(fmap: torch.Tensor, seed_xy: Sequence[float]) -> np.ndarray:
    """Cosine similarity of each patch in ``[h, w, D]`` to the seed patch -> ``[h, w]`` in [-1, 1]."""
    if fmap.dim() == 4:
        fmap = fmap[0]
    h, w, _ = fmap.shape
    r, c = seed_to_cell(seed_xy, h, w)
    flat = fmap.reshape(h * w, -1).float()
    flat = torch.nn.functional.normalize(flat, dim=1)
    seed_vec = torch.nn.functional.normalize(fmap[r, c].reshape(1, -1).float(), dim=1)
    sim = (flat @ seed_vec.T).reshape(h, w)
    return sim.cpu().numpy()

def _rollout(attn_layers, head: str, discard_ratio: float) -> np.ndarray:
    """Compose per-layer attention ``[1, heads, N, N]`` into a single ``[N, N]`` rollout matrix."""
    reduce = {"mean": lambda a: a.mean(0), "max": lambda a: a.max(0).values,
              "min": lambda a: a.min(0).values}
    if head not in reduce:
        raise ValueError(f"head must be one of {list(reduce)} (got {head!r}).")
    result = None
    for attn in attn_layers:
        A = reduce[head](attn[0].float())  # [N, N]
        if discard_ratio > 0:
            flat = A.flatten()
            n_zero = int(flat.numel() * discard_ratio)
            if n_zero > 0:
                _, idx = flat.topk(n_zero, largest=False)
                flat[idx] = 0.0
                A = flat.reshape(A.shape)
        N = A.shape[-1]
        A = A + torch.eye(N)
        A = A / A.sum(dim=-1, keepdim=True)
        result = A if result is None else A @ result
    return result.cpu().numpy()
