"""Deterministic execution, portable I/O, and environment provenance."""

from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import site
from typing import Any, Iterable
import numpy as np
import torch
from .config import Config


def configure(config: Config) -> None:
    """Set seeds and float32 deterministic settings before using CUDA."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLBACKEND", "Agg")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: Any) -> None:
    """Write UTF-8 JSON without nonstandard NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows with a stable header and portable newlines."""
    iterator = iter(rows)
    first = next(iterator)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a small exported result table."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    """Hash a file without loading model-sized files into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    """Record installed packages and compute hardware without private host paths."""
    names = [
        "featlens",
        "torch",
        "torchvision",
        "transformers",
        "timm",
        "numpy",
        "matplotlib",
        "pillow",
        "scikit-learn",
        "scipy",
        "pycocotools",
        "pyyaml",
        "requests",
        "huggingface_hub",
        "safetensors",
        "nbformat",
        "nbclient",
    ]
    return {
        "python": sys.version.split()[0],
        "virtual_environment": sys.prefix != sys.base_prefix,
        "user_site_enabled": site.ENABLE_USER_SITE,
        "platform": platform.system(),
        "versions": {name: importlib.metadata.version(name) for name in names},
        "featlens_direct_url": json.loads(
            importlib.metadata.distribution("featlens").read_text("direct_url.json") or "null"
        ),
        "torch_runtime_version": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else None,
        "cpu_threads": torch.get_num_threads(),
    }
