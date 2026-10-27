"""CLS attention and FeatLens cosine/rollout views with intact prefix transitions."""

from typing import Sequence
import numpy as np
import torch
from .models import TokenLayout
from ._featlens import cosine_similarity_map, _rollout


def cls_attention(
    attention: torch.Tensor, layout: TokenLayout
) -> tuple[np.ndarray, dict[str, float]]:
    """Average heads, then slice CLS-to-patch probabilities without renormalizing."""
    cls = attention[0, :, 0, :].mean(dim=0)
    torch.testing.assert_close(cls.sum(), cls.new_tensor(1), rtol=1e-4, atol=1e-5)
    patches = cls[layout.prefix :]
    assert patches.numel() == layout.grid_size**2
    mass = {
        "cls_mass": float(cls[0]),
        "register_mass": float(cls[1 : layout.prefix].sum()),
        "patch_mass": float(patches.sum()),
    }
    return patches.reshape(layout.grid_size, layout.grid_size).cpu().numpy(), mass


def cosine_map(features: torch.Tensor, seed_xy: Sequence[float]) -> np.ndarray:
    """Call the original FeatLens cosine-to-seed kernel on an H×W×D grid."""
    return cosine_similarity_map(features, seed_xy)


def attention_rollout(
    attentions: tuple[torch.Tensor, ...], layout: TokenLayout
) -> tuple[np.ndarray, np.ndarray]:
    """Compose every full attention matrix with the original FeatLens kernel."""
    averaged = [a.detach().mean(dim=1, keepdim=True).cpu() for a in attentions]
    rolled = _rollout(averaged, head="mean", discard_ratio=0.0)
    assert np.allclose(rolled.sum(axis=1), 1, rtol=2e-4, atol=2e-5)
    heat = rolled[0, layout.prefix :].reshape(layout.grid_size, layout.grid_size)
    mass = np.array([rolled[0, 0], rolled[0, 1 : layout.prefix].sum(), heat.sum()])
    return heat, mass
