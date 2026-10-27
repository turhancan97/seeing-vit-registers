"""Load the shared experiment configuration."""

from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
Config = dict[str, Any]


def load_config(path: str | Path | None = None) -> Config:
    """Read settings and reject unsupported token layouts early."""
    with Path(path or ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["image_size"] == 518 and config["patch_size"] == 14
    assert len(config["selected_indices"]) == 6
    assert len({m["key"] for m in config["models"]}) == len(config["models"])
    return config
