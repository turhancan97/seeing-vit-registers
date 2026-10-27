"""COCO instance-mask complements as an explicitly limited background proxy."""

from pathlib import Path
from typing import Any
import numpy as np
from pycocotools import mask as coco_mask
from .config import Config, ROOT
from .runtime import read_json, write_csv
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


def analyze_locations(
    config: Config, sources: list[dict[str, Any]], results: Path, image_dir: Path
) -> list[dict[str, Any]]:
    """Count high-norm patch centers outside annotated COCO objects and crowds."""
    annotations = read_json(ROOT / config["annotation_manifest"])
    thresholds = read_json(results / "data/thresholds.json")
    grid = config["image_size"] // config["patch_size"]
    images = {im["id"]: im for im in annotations["images"]}
    masks = []
    for s in sources:
        anns = [a for a in annotations["annotations"] if a["image_id"] == s["coco_id"]]
        masks.append(background_patch_mask(images[s["coco_id"]], anns, grid))
    masks = np.stack(masks)
    np.savez_compressed(
        results / "data/background_proxy_masks.npz", outside_coco_instance_masks=masks
    )
    rows = []
    per_image = []
    for size in config["sizes"]:
        threshold = thresholds[size]["threshold"]
        for suffix in ["no_registers", "registers"]:
            key = f"{size}_{suffix}"
            data = np.load(results / f"data/{key}.npz")
            norms = data["normalized_norms"].reshape(len(sources), grid, grid)
            for t in [threshold] + config["fixed_thresholds"]:
                high = norms > t
                n = int(high.sum())
                bg = int((high & masks).sum())
                fg = n - bg
                bg_rate = float((high & masks).sum() / masks.sum() * 100)
                fg_rate = float((high & ~masks).sum() / (~masks).sum() * 100)
                rows.append(
                    dict(
                        model=key,
                        threshold=t,
                        kind="data_driven" if t == threshold else "fixed",
                        high_count=n,
                        high_outside_instances=bg,
                        high_inside_instances=fg,
                        percent_high_outside_instances=bg / n * 100 if n else None,
                        percent_all_patches_outside_instances=float(masks.mean() * 100),
                        high_rate_outside_instances_percent=bg_rate,
                        high_rate_inside_instances_percent=fg_rate,
                        rate_ratio=bg_rate / fg_rate if fg_rate else None,
                    )
                )
                for i, s in enumerate(sources):
                    n_i = int(high[i].sum())
                    bg_i = int((high[i] & masks[i]).sum())
                    per_image.append(
                        dict(
                            model=key,
                            image_index=i,
                            coco_id=s["coco_id"],
                            threshold=t,
                            high_count=n_i,
                            high_outside_instances=bg_i,
                            percent_high_outside_instances=bg_i / n_i * 100 if n_i else None,
                            percent_patches_outside_instances=float(masks[i].mean() * 100),
                        )
                    )
    write_csv(results / "data/background_proxy_summary.csv", rows)
    write_csv(results / "data/background_proxy_per_image.csv", per_image)
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
