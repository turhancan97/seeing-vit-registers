"""One entry point for downloading inputs, fresh inference, analysis, and figures."""

from __future__ import annotations
import gc
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
from .config import Config, ROOT, load_config

os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from .runtime import configure, environment, sha256, write_csv, write_json
from .data import download_images, save_attribution, contact_sheet
from .models import load_model, preprocess
from .attention import cls_attention, cosine_map, attention_rollout
from .pca import pca_rgb


def neighbor_similarity(embeddings: torch.Tensor, grid: int) -> np.ndarray:
    """Average cosine to each patch's available 2–4 spatial neighbors."""
    embedded = torch.nn.functional.normalize(embeddings.reshape(grid, grid, -1), dim=-1)
    neighbors = torch.zeros((grid, grid), device=embedded.device)
    counts = torch.zeros_like(neighbors)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        similarity = (embedded[tuple(left)] * embedded[tuple(right)]).sum(-1)
        neighbors[tuple(left)] += similarity
        neighbors[tuple(right)] += similarity
        counts[tuple(left)] += 1
        counts[tuple(right)] += 1
    return (neighbors / counts).cpu().numpy()


def run_inference(
    config: Config,
    sources: list[dict[str, Any]],
    results: Path,
    image_dir: Path,
    device: str,
    reuse: bool = False,
) -> dict[str, Any]:
    """Measure every image and model; saved tensors are reused only on request."""
    metadata = {}
    selected = sorted(config["selected_indices"])
    for spec in config["models"]:
        key = spec["key"]
        output = results / f"data/{key}.npz"
        if reuse and output.exists():
            from .runtime import read_json

            metadata[key] = read_json(results / "data/models.json")[key]
            continue
        started = time.monotonic()
        model, layout, meta = load_model(spec, config, device)
        grid = layout.grid_size
        layers, final_norms, normalized_norms, neighbors = [], [], [], []
        views = {
            k: []
            for k in [
                "features",
                "pca_rgb",
                "pca_variance_ratio",
                "cls_attention",
                "cosine",
                "rollout",
                "rollout_mass",
            ]
        }
        attention_rows = []
        for index, source in enumerate(sources):
            _, pixels = preprocess(image_dir / Path(source["path"]).name, config)
            with torch.inference_mode():
                output_states = model(
                    pixels.to(device), output_hidden_states=True, output_attentions=True
                )
                torch.testing.assert_close(
                    model.layernorm(output_states.hidden_states[-1]),
                    output_states.last_hidden_state,
                )
            assert len(output_states.hidden_states) == meta["layers"] + 1
            assert len(output_states.attentions) == meta["layers"]
            assert all(a is not None for a in output_states.attentions)
            layer_norms = np.stack(
                [
                    layout.patches(h).norm(dim=-1)[0].cpu().numpy()
                    for h in output_states.hidden_states
                ]
            )
            assert np.isfinite(layer_norms).all()
            normalized = layout.patches(output_states.last_hidden_state)[0].cpu()
            layers.append(layer_norms)
            final_norms.append(layer_norms[-1])
            normalized_norms.append(normalized.norm(dim=-1).numpy())
            neighbors.append(
                neighbor_similarity(layout.patches(output_states.hidden_states[0])[0], grid)
            )
            attention, mass = cls_attention(output_states.attentions[-1], layout)
            attention_rows.append({"image_index": index, **mass})
            if index in selected:
                rgb, variance = pca_rgb(normalized.numpy(), grid, config["seed"])
                rollout, rollout_mass = attention_rollout(output_states.attentions, layout)
                values = {
                    "features": normalized.numpy(),
                    "pca_rgb": rgb,
                    "pca_variance_ratio": variance,
                    "cls_attention": attention,
                    "cosine": cosine_map(
                        normalized.reshape(grid, grid, -1), config["cosine_seed_xy"]
                    ),
                    "rollout": rollout,
                    "rollout_mass": rollout_mass,
                }
                for name, value in values.items():
                    views[name].append(value)
            print(
                f"{key}: {index + 1}/{len(sources)}; normalized max {normalized_norms[-1].max():.3f}",
                flush=True,
            )
            del output_states, normalized
        np.savez_compressed(
            output,
            layer_norms=np.stack(layers),
            final_norms=np.stack(final_norms),
            normalized_norms=np.stack(normalized_norms),
            embedding_neighbor_cosine=np.stack(neighbors),
            selected_indices=np.array(selected),
            **{k: np.stack(v) for k, v in views.items()},
        )
        write_csv(results / f"data/{key}_attention_mass.csv", attention_rows)
        meta.update(image_count=len(sources), elapsed_seconds=time.monotonic() - started)
        metadata[key] = meta
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(results / "data/models.json", metadata)
    return metadata


def run_all(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    device: str | None = None,
    reuse: bool = False,
) -> Path:
    """Reproduce every experimental number and figure from verified inputs."""
    config = load_config(config_path)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    configure(config)
    results = Path(output_dir).resolve() if output_dir else ROOT / config["results_dir"]
    images = Path(image_dir).resolve() if image_dir else ROOT / "images"
    (results / "data").mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    sources = download_images(config, image_dir=images)
    save_attribution(sources, results)
    contact_sheet(config, sources, images, results)
    write_json(results / "data/environment.json", environment())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}; fresh inference: {not reuse}", flush=True)
    run_inference(config, sources, results, images, device, reuse)
    from .plotting import make_figures, main_comparison
    from .location import analyze_locations

    make_figures(config, sources, results, images)
    analyze_locations(config, sources, results, images)
    main_comparison(config, sources, results, images)
    from .validation import validate_results

    validate_results(config, results)
    code = {
        str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src").glob("*.py"))
    }
    write_json(
        results / "data/run_provenance.json",
        {
            "requirements_sha256": sha256(ROOT / "requirements.txt"),
            "seed": config["seed"],
            "fresh_inference": not reuse,
            "device": device,
            "configuration": config,
            "source_sha256": code,
            "image_manifest_sha256": sha256(ROOT / config["image_manifest"]),
            "annotation_manifest_sha256": sha256(ROOT / config["annotation_manifest"]),
        },
    )
    print(f"Finished: {results}", flush=True)
    return results
