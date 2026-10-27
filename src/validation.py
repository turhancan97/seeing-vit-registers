"""Domain checks for dual-space, six-model outputs and image bootstrap."""

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from .config import Config
from .runtime import read_csv, read_json, write_json
from .spaces import SPACES, reference_threshold


def validate_results(config: Config, results: Path) -> list[dict[str, Any]]:
    """Check patch/prefix layouts, both spaces, norm decomposition and intervals."""
    models = read_json(results / "data/models.json")
    summary = read_csv(results / "data/summary.csv")
    checks = []
    grid = config["image_size"] // config["patch_size"]
    for spec in config["models"]:
        key = spec["key"]
        meta = models[key]
        with np.load(results / f"data/{key}.npz") as d:
            assert meta["sequence_length"] == grid**2 + 1 + spec["registers"]
            for space, desc in SPACES.items():
                n = d[desc["norms"]]
                layer = d[desc["layers"]]
                prefix = d[desc["prefix"]]
                assert n.shape == (config["image_count"], grid**2)
                assert layer.shape == (config["image_count"], meta["layers"] + 1, grid**2)
                assert prefix.shape == (
                    config["image_count"],
                    meta["layers"] + 1,
                    1 + spec["registers"],
                )
                assert np.isfinite(prefix).all() and np.isfinite(layer).all()
                np.testing.assert_allclose(layer[:, -1], n, rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(
                    np.linalg.norm(d[desc["features"]], axis=-1),
                    n[d["selected_indices"]],
                    rtol=2e-5,
                    atol=2e-5,
                )
                row = next(r for r in summary if r["model"] == key and r["feature_space"] == space)
                assert int(row["above_reference_count"]) == int(
                    (n > reference_threshold(config, space)).sum()
                )
                assert d[desc["cosine"]].min() >= -1.00001 and d[desc["cosine"]].max() <= 1.00001
            np.testing.assert_allclose(d["rollout_mass"].sum(axis=1), 1, atol=2e-4)
            np.testing.assert_allclose(
                d["raw_centered_norms"] ** 2 + meta["hidden_size"] * d["raw_feature_mean"] ** 2,
                d["final_norms"] ** 2,
                rtol=2e-5,
                atol=2e-3,
            )
        checks.append(
            dict(
                model=key,
                status="passed",
                feature_spaces=list(SPACES),
                images=config["image_count"],
                patches_per_image=grid**2,
            )
        )
    for size in config["sizes"]:
        for space in SPACES:
            for stem in [
                "token_norm_histogram",
                "visual_comparison",
                "high_norm_locations",
                "featlens_comparison",
            ]:
                for ext in ["png", "pdf"]:
                    assert (results / f"figures/{stem}_{size}_{space}.{ext}").exists()
        assert (results / f"figures/prefix_token_norms_{size}.png").exists()
    indices = np.load(results / "data/bootstrap_image_indices.npz")["indices"]
    assert indices.shape == (config["bootstrap"]["resamples"], config["image_count"])
    assert indices.min() >= 0 and indices.max() < config["image_count"]
    for row in read_csv(results / "data/background_proxy_summary.csv"):
        if int(row["high_count"]) == 0:
            assert (
                row["outside_fraction_ci_lower_percent"] == ""
                and int(row["bootstrap_valid_resamples"]) == 0
            )
        else:
            lo, hi = (
                float(row["outside_fraction_ci_lower_percent"]),
                float(row["outside_fraction_ci_upper_percent"]),
            )
            assert 0 <= lo <= hi <= 100
    # Independently reconstruct image-bootstrap intervals from the exported image counts.
    image_rows = read_csv(results / "data/background_proxy_per_image.csv")
    for row in read_csv(results / "data/background_proxy_summary.csv"):
        per_image = sorted(
            [
                r
                for r in image_rows
                if all(r[k] == row[k] for k in ["model", "feature_space", "kind"])
            ],
            key=lambda r: int(r["image_index"]),
        )
        totals = np.array([int(r["high_count"]) for r in per_image])
        outside = np.array([int(r["high_outside_instances"]) for r in per_image])
        denominator = totals[indices].sum(axis=1)
        valid = denominator > 0
        assert int(valid.sum()) == int(row["bootstrap_valid_resamples"])
        if valid.any():
            samples = outside[indices].sum(axis=1)[valid] / denominator[valid] * 100
            alpha = (1 - config["bootstrap"]["confidence"]) / 2
            interval = np.quantile(samples, [alpha, 1 - alpha])
            np.testing.assert_allclose(
                interval,
                [
                    float(row["outside_fraction_ci_lower_percent"]),
                    float(row["outside_fraction_ci_upper_percent"]),
                ],
                atol=1e-10,
            )
    for path in (results / "figures").glob("*.png"):
        with Image.open(path) as image:
            assert min(image.info["dpi"]) > config["figure_dpi"] - 1
            image.verify()
    for path in (results / "figures").glob("*.pdf"):
        assert path.read_bytes().startswith(b"%PDF")
    write_json(results / "data/validation.json", checks)
    return checks
