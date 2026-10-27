"""Checksum-verified COCO downloads and explicit source/licensing metadata."""

from __future__ import annotations
import concurrent.futures
from pathlib import Path
import re
from typing import Any
from PIL import Image, ImageDraw, ImageOps
import requests
from .config import Config, ROOT
from .runtime import read_json, sha256, write_csv, write_json


def write_sources(config: Config, root: Path = ROOT) -> None:
    """Write attribution links and licenses for every downloaded photograph."""
    rows = read_json(root / config["image_manifest"])
    lines = [
        "# Image sources",
        "",
        "The pipeline downloads these 20 COCO val2017 photographs. JPEGs are ignored by Git.",
        "The code is MIT licensed; photographs and image-containing figures retain their source-image terms.",
        "License metadata comes from the official COCO annotations. Follow each photo link for original credits.",
        "",
        "| COCO ID | Download | Original photograph | COCO-recorded license | SHA-256 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        photo = re.search(r"/(\d+)_", row["original_flickr_url"]).group(1)
        lines.append(
            f"| {row['coco_id']:012d} | [JPEG]({row['url']}) | [Flickr](https://www.flickr.com/photo.gne?id={photo}) | [{row['license_name']}]({row['license_url']}) | `{row['sha256']}` |"
        )
    lines += [
        "",
        "Manifest: [`manifest.json`](manifest.json). Selected annotation/caption metadata: [`coco_annotations.json`](coco_annotations.json).",
        "Official annotation archive: https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "",
    ]
    (root / "images/SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


def download_images(
    config: Config, root: Path = ROOT, image_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Fetch missing images, verify exact bytes, and preserve manifest order."""
    rows = read_json(root / config["image_manifest"])
    assert len(rows) == config["image_count"]
    target = image_dir or root / "images"
    target.mkdir(parents=True, exist_ok=True)

    def fetch(row: dict[str, Any]) -> None:
        path = target / Path(row["path"]).name
        if not path.exists():
            response = requests.get(row["url"], timeout=90)
            response.raise_for_status()
            temporary = path.with_suffix(".download")
            temporary.write_bytes(response.content)
            if sha256(temporary) != row["sha256"]:
                raise ValueError(f"Checksum mismatch for {path.name}")
            temporary.replace(path)
        if sha256(path) != row["sha256"]:
            raise ValueError(f"{path.name} differs from the recorded original")
        with Image.open(path) as image:
            assert image.size == (row["original_width"], row["original_height"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, rows))
    write_sources(config, root)
    return rows


def save_attribution(sources: list[dict[str, Any]], results: Path) -> None:
    """Export a compact CSV alongside the numerical results."""
    write_json(results / "data/image_sources.json", sources)
    write_csv(
        results / "data/image_attribution.csv",
        (
            {
                "image_index": i,
                "coco_id": row["coco_id"],
                "local_path": row["path"],
                "download_url": row["url"],
                "original_flickr_url": row["original_flickr_url"],
                "license_name": row["license_name"],
                "license_url": row["license_url"],
                "object_categories": row["object_categories"],
                "example_caption": row["example_caption"],
            }
            for i, row in enumerate(sources)
        ),
    )


def contact_sheet(
    config: Config, sources: list[dict[str, Any]], image_dir: Path, results: Path
) -> None:
    """Show every input in the fixed, outcome-independent manifest order."""
    sheet = Image.new("RGB", (5 * 225, ((len(sources) + 4) // 5) * 250), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(sources):
        with Image.open(image_dir / Path(row["path"]).name) as source:
            image = ImageOps.contain(source.convert("RGB"), (220, 220))
        x, y = i % 5 * 225, i // 5 * 250
        sheet.paste(image, (x, y))
        draw.text((x, y + 223), f"{i}: {Path(row['path']).stem}", fill="black")
    (results / "figures").mkdir(parents=True, exist_ok=True)
    sheet.save(results / "figures/dataset_contact_sheet.png", dpi=(config["figure_dpi"],) * 2)
