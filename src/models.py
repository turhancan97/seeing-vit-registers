"""Pinned Hugging Face models, preprocessing, and prefix-token handling."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import Dinov2Model, Dinov2WithRegistersModel
from .config import Config


@dataclass(frozen=True)
class TokenLayout:
    """Describe CLS, optional registers, and a square image-patch grid."""

    registers: int
    grid_size: int = 37

    @property
    def prefix(self) -> int:
        """Number of non-spatial tokens, including CLS."""
        return 1 + self.registers

    def patches(self, sequence: torch.Tensor) -> torch.Tensor:
        """Remove every prefix token and verify the expected spatial count."""
        patches = sequence[:, self.prefix :, :]
        assert patches.shape[1] == self.grid_size**2, (sequence.shape, self)
        return patches


def preprocess(path: Path, config: Config) -> tuple[Image.Image, torch.Tensor]:
    """Bicubic squash-resize to 518² and apply ImageNet normalization."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image = image.resize((config["image_size"], config["image_size"]), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
    mean = torch.tensor(config["image_mean"])[:, None, None]
    std = torch.tensor(config["image_std"])[:, None, None]
    return image, ((pixels - mean) / std).unsqueeze(0)


def load_model(
    spec: dict[str, Any], config: Config, device: str
) -> tuple[torch.nn.Module, TokenLayout, dict[str, Any]]:
    """Load the exact official checkpoint using eager attention and float32."""
    cls = Dinov2WithRegistersModel if spec["registers"] else Dinov2Model
    model = cls.from_pretrained(spec["id"], revision=spec["revision"], attn_implementation="eager")
    model = model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    registers = getattr(model.config, "num_register_tokens", 0)
    assert registers == spec["registers"]
    assert model.config.patch_size == config["patch_size"]
    assert model.config._attn_implementation == "eager"
    layout = TokenLayout(registers, config["image_size"] // config["patch_size"])
    metadata = {
        "model_id": spec["id"],
        "revision": spec["revision"],
        "model_class": cls.__name__,
        "backend": "transformers",
        "registers": registers,
        "layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "dtype": "float32",
        "attention_implementation": "eager",
        "sequence_length": layout.prefix + layout.grid_size**2,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return model, layout, metadata
