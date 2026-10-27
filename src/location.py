"""COCO instance-mask complements as an explicitly limited background proxy."""

from pathlib import Path
from typing import Any
import numpy as np
from pycocotools import mask as coco_mask
from .config import Config, ROOT
from .runtime import read_json, write_csv, write_json
from .spaces import SPACES, reference_threshold
from .norms import choose_threshold, choose_lower_threshold
from .models import preprocess
from .plotting import save_figure


def background_patch_mask(
    image: dict[str, Any], annotations: list[dict[str, Any]], grid: int
) -> np.ndarray:
    """True if patch center is outside every annotated COCO object (crowds included)."""
    height, width = image["height"], image["width"]
    foreground = np.zeros((height, width), dtype=bool)
    for ann in annotations:
        seg = ann["segmentation"]
        if isinstance(seg, list):
            rles = coco_mask.frPyObjects(seg, height, width)
            rle = coco_mask.merge(rles)
        elif isinstance(seg["counts"], list):
            rle = coco_mask.frPyObjects(seg, height, width)
        else:
            rle = seg
        decoded = coco_mask.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        foreground |= decoded.astype(bool)
    yy = np.minimum(((np.arange(grid) + 0.5) / grid * height).astype(int), height - 1)
    xx = np.minimum(((np.arange(grid) + 0.5) / grid * width).astype(int), width - 1)
    return ~foreground[np.ix_(yy, xx)]


def bootstrap_fraction(
    outside: np.ndarray, total: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, int]:
    """Resample whole images and form the pooled outside/total token fraction."""
    denominators = total[indices].sum(axis=1)
    samples = np.full(len(indices), np.nan)
    valid = denominators > 0
    samples[valid] = outside[indices].sum(axis=1)[valid] / denominators[valid] * 100
    return samples, int(valid.sum())


def analyze_locations(
    config: Config, sources: list[dict[str, Any]], results: Path, image_dir: Path
) -> list[dict[str, Any]]:
    """Report both spaces and percentile bootstrap intervals over images."""
    annotations = read_json(ROOT / config["annotation_manifest"])
    grid = config["image_size"] // config["patch_size"]
    images = {im["id"]: im for im in annotations["images"]}
    masks = np.stack(
        [
            background_patch_mask(
                images[s["coco_id"]],
                [a for a in annotations["annotations"] if a["image_id"] == s["coco_id"]],
                grid,
            )
            for s in sources
        ]
    )
    np.savez_compressed(
        results / "data/background_proxy_masks.npz", outside_coco_instance_masks=masks
    )
    settings = config["bootstrap"]
    rng = np.random.default_rng(settings["seed"])
    indices = rng.integers(0, len(sources), size=(settings["resamples"], len(sources)))
    np.savez_compressed(results / "data/bootstrap_image_indices.npz", indices=indices)
    alpha = (1 - settings["confidence"]) / 2
    rows = []
    per_image = []
    bootstrap_rows = []
    for spec in config["models"]:
        key = spec["key"]
        data = np.load(results / f"data/{key}.npz")
        for space, desc in SPACES.items():
            norms = data[desc["norms"]].reshape(len(sources), grid, grid)
            ref = reference_threshold(config, space)
            rules = {
                "reference": ref,
                "mean_plus_3std": float(norms.mean() + config["layer_k"] * norms.std()),
            }
            gap = choose_threshold(norms, config)
            if gap["clear_gap"]:
                rules["gap"] = gap["threshold"]
            lower_gap = choose_lower_threshold(norms, config)
            if lower_gap["clear_gap"]:
                rules["lower_gap"] = lower_gap["threshold"]
            for t in config["fixed_thresholds"]:
                rules[f"fixed_{t}"] = t
            for kind, t in rules.items():
                high = norms < t if kind == "lower_gap" else norms > t
                totals = high.sum(axis=(1, 2))
                outside = (high & masks).sum(axis=(1, 2))
                samples, valid = bootstrap_fraction(outside, totals, indices)
                lo, hi = (
                    np.quantile(samples[np.isfinite(samples)], [alpha, 1 - alpha])
                    if valid
                    else (None, None)
                )
                n = int(totals.sum())
                bg = int(outside.sum())
                fg = n - bg
                bg_rate = float(bg / masks.sum() * 100)
                fg_rate = float(fg / (~masks).sum() * 100)
                rows.append(
                    dict(
                        model=key,
                        feature_space=space,
                        threshold=t,
                        kind=kind,
                        selection_direction="below" if kind == "lower_gap" else "above",
                        high_count=n,
                        high_outside_instances=bg,
                        high_inside_instances=fg,
                        percent_high_outside_instances=bg / n * 100 if n else None,
                        outside_fraction_ci_lower_percent=float(lo) if valid else None,
                        outside_fraction_ci_upper_percent=float(hi) if valid else None,
                        bootstrap_resamples=settings["resamples"],
                        bootstrap_valid_resamples=valid,
                        percent_all_patches_outside_instances=float(masks.mean() * 100),
                        high_rate_outside_instances_percent=bg_rate,
                        high_rate_inside_instances_percent=fg_rate,
                        rate_ratio=bg_rate / fg_rate if fg_rate else None,
                    )
                )
                for i, s in enumerate(sources):
                    per_image.append(
                        dict(
                            model=key,
                            feature_space=space,
                            image_index=i,
                            coco_id=s["coco_id"],
                            threshold=t,
                            kind=kind,
                            selection_direction="below" if kind == "lower_gap" else "above",
                            high_count=int(totals[i]),
                            high_outside_instances=int(outside[i]),
                            percent_high_outside_instances=float(outside[i] / totals[i] * 100)
                            if totals[i]
                            else None,
                        )
                    )
                if kind in {"reference", "mean_plus_3std", "gap", "lower_gap"}:
                    bootstrap_rows.extend(
                        dict(
                            model=key,
                            feature_space=space,
                            kind=kind,
                            selection_direction="below" if kind == "lower_gap" else "above",
                            resample=i,
                            outside_fraction_percent=float(value) if np.isfinite(value) else None,
                        )
                        for i, value in enumerate(samples)
                    )
    write_csv(results / "data/background_proxy_summary.csv", rows)
    write_csv(results / "data/background_proxy_per_image.csv", per_image)
    write_csv(results / "data/bootstrap_outside_fraction.csv", bootstrap_rows)
    write_json(
        results / "data/bootstrap_settings.json",
        dict(
            **settings,
            unit="image",
            statistic="ratio of pooled outside/high counts after resampling images",
            zero_denominator="undefined; excluded from percentile calculation and counted explicitly",
        ),
    )
    # Show precisely which areas were counted, without using these masks to select inputs.
    import matplotlib.pyplot as plt

    selected = config["selected_indices"]
    fig, axes = plt.subplots(
        len(selected), 2, figsize=(5.8, 2.9 * len(selected)), layout="constrained"
    )
    for r, idx in enumerate(selected):
        image, _ = preprocess(image_dir / Path(sources[idx]["path"]).name, config)
        axes[r, 0].imshow(image)
        axes[r, 1].imshow(masks[idx], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r, 0].set_title(Path(sources[idx]["path"]).stem, fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Location proxy · white: outside annotated COCO objects\nNot a complete semantic background mask",
        fontsize=11,
    )
    save_figure(fig, results, config, "background_proxy_masks")
    return rows
