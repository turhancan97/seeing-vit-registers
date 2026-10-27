"""Check the saved artifacts used to build the figures and result tables."""

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from .config import Config
from .runtime import read_csv, read_json, write_json


def validate_results(config: Config, results: Path) -> list[dict[str, Any]]:
    """Verify shapes, prefix exclusion, attention mass, CSV counts, and PNGs."""
    metadata = read_json(results / "data/models.json")
    summary = read_csv(results / "data/summary.csv")
    grid = config["image_size"] // config["patch_size"]
    checks = []
    for spec in config["models"]:
        key = spec["key"]
        with np.load(results / f"data/{key}.npz") as data:
            assert data["normalized_norms"].shape == (config["image_count"], grid**2)
            assert data["layer_norms"].shape == (
                config["image_count"],
                metadata[key]["layers"] + 1,
                grid**2,
            )
            assert metadata[key]["sequence_length"] == grid**2 + 1 + spec["registers"]
            assert np.allclose(
                np.linalg.norm(data["features"], axis=-1),
                data["normalized_norms"][data["selected_indices"]],
                atol=2e-5,
            )
            assert np.allclose(data["rollout_mass"].sum(axis=1), 1, atol=2e-4)
            assert data["cosine"].min() >= -1.00001 and data["cosine"].max() <= 1.00001
            assert np.array_equal(data["layer_norms"][:, -1], data["final_norms"])
            count = int((data["normalized_norms"] > config["threshold"]["value"]).sum())
        row = next(
            r for r in summary if r["model"] == key and r["feature_space"] == "post_layernorm"
        )
        assert int(row["high_count"]) == count
        checks.append(
            {
                "model": key,
                "status": "passed",
                "images": config["image_count"],
                "patches_per_image": grid**2,
            }
        )
    pngs, pdfs = (
        list((results / "figures").glob("*.png")),
        list((results / "figures").glob("*.pdf")),
    )
    assert len(pngs) == 15 and len(pdfs) == 14
    for path in pngs:
        with Image.open(path) as image:
            assert min(image.info["dpi"]) > config["figure_dpi"] - 1
            image.verify()
    for path in pdfs:
        assert path.read_bytes().startswith(b"%PDF")
    write_json(results / "data/validation.json", checks)
    return checks
