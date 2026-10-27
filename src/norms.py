"""Experiments A and B: final patch norms, thresholds, and layer-wise tails."""

from pathlib import Path
from typing import Any
import numpy as np
from .config import Config
from .runtime import write_csv, write_json


def choose_threshold(values: np.ndarray, config: Config) -> dict[str, Any]:
    """Largest multiplicative gap in the upper tail, with >=0.01% (at least 3 patches here) above it.

    Require a >=20% jump before describing it as a clear observed gap.
    Otherwise use Otsu on log norms and report the lack of a clear gap.
    """
    a = np.sort(values.ravel().astype(np.float64))
    q_low, q_high = config["threshold"]["search_quantiles"]
    lower = int(q_low * len(a))
    upper = min(int(q_high * len(a)), len(a) - config["threshold"]["minimum_upper_count"])
    ratios = a[lower + 1 : upper + 1] / a[lower:upper]
    j = lower + int(np.argmax(ratios))
    ratio = float(a[j + 1] / a[j])
    if ratio >= config["threshold"]["minimum_gap_ratio"]:
        return dict(
            threshold=float(np.sqrt(a[j] * a[j + 1])),
            method="largest relative upper-tail gap",
            gap_lower=float(a[j]),
            gap_upper=float(a[j + 1]),
            gap_ratio=ratio,
            clear_gap=True,
            rule="Search the configured upper-tail quantiles; require the configured gap ratio; use its geometric midpoint.",
        )
    log_values = np.log(a)
    hist, edges = np.histogram(log_values, bins=config["threshold"]["otsu_bins"])
    centers = (edges[:-1] + edges[1:]) / 2
    weights = hist / hist.sum()
    p = np.cumsum(weights)
    m = np.cumsum(weights * centers)
    score = (m[-1] * p - m) ** 2 / np.maximum(p * (1 - p), 1e-12)
    t = int(np.argmax(score[:-1]))
    return dict(
        threshold=float(np.exp(edges[t + 1])),
        method="Otsu on log norms (no clear gap)",
        gap_lower=float(a[j]),
        gap_upper=float(a[j + 1]),
        gap_ratio=ratio,
        clear_gap=False,
        rule="No >=20% adjacent gap in baseline percentiles 90–99.99; maximize 256-bin log-norm Otsu between-class variance. Descriptive split, not proof of a separate artifact mode.",
    )


def analyze(
    config: Config, sources: list[dict[str, Any]], results: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Export final norms (A), per-layer tails (B), and all sensitivity checks."""
    grid = config["image_size"] // config["patch_size"]
    all_stats, layer_rows, image_rows, threshold_rows, redundancy_rows = [], [], [], [], []
    threshold_info = {}
    for size in config["sizes"]:
        baseline = np.load(results / f"data/{size}_no_registers.npz")
        info = choose_threshold(baseline["normalized_norms"], config)
        if not info["clear_gap"] and "large" in threshold_info:
            info["unimodal_descriptive_split"] = info["threshold"]
            info["threshold"] = threshold_info["large"]["threshold"]
            info["method"] = "large-model gap cutoff; no clear separate gap in this size"
            info["rule"] += (
                " The unimodal Otsu split is not used to label artifacts. Apply the large-model cutoff for a common absolute-scale comparison."
            )
        info["estimated_threshold"] = info["threshold"]
        info["threshold"] = config["threshold"]["value"]
        threshold_info[size] = info
        for suffix in ["no_registers", "registers"]:
            key = f"{size}_{suffix}"
            data = np.load(results / f"data/{key}.npz")
            for space, array_key in [
                ("pre_layernorm", "final_norms"),
                ("post_layernorm", "normalized_norms"),
            ]:
                norms = data[array_key]
                all_stats.append(
                    dict(
                        model=key,
                        feature_space=space,
                        images=len(sources),
                        patch_tokens=int(norms.size),
                        mean_norm=float(norms.mean()),
                        std_norm=float(norms.std()),
                        max_norm=float(norms.max()),
                        p50=float(np.median(norms)),
                        p95=float(np.quantile(norms, 0.95)),
                        p99=float(np.quantile(norms, 0.99)),
                        chosen_threshold=info["threshold"],
                        high_count=int((norms > info["threshold"]).sum()),
                        high_percent=float((norms > info["threshold"]).mean() * 100),
                    )
                )
                for threshold in [info["threshold"]] + config["fixed_thresholds"]:
                    threshold_rows.append(
                        dict(
                            model=key,
                            feature_space=space,
                            threshold=threshold,
                            kind="data_driven" if threshold == info["threshold"] else "fixed",
                            high_count=int((norms > threshold).sum()),
                            high_percent=float((norms > threshold).mean() * 100),
                        )
                    )
            norms = data["normalized_norms"]
            if "embedding_neighbor_cosine" in data:
                neighbors = data["embedding_neighbor_cosine"].reshape(norms.shape)
                for name, mask in [
                    ("high_norm", norms > info["threshold"]),
                    ("other", norms <= info["threshold"]),
                ]:
                    redundancy_rows.append(
                        dict(
                            model=key,
                            group=name,
                            patch_count=int(mask.sum()),
                            mean_embedding_neighbor_cosine=float(neighbors[mask].mean())
                            if mask.any()
                            else None,
                            median_embedding_neighbor_cosine=float(np.median(neighbors[mask]))
                            if mask.any()
                            else None,
                        )
                    )
            for i, n in enumerate(norms):
                image_rows.append(
                    dict(
                        model=key,
                        image_index=i,
                        image=sources[i]["path"],
                        mean_norm=float(n.mean()),
                        max_norm=float(n.max()),
                        threshold=info["threshold"],
                        high_count=int((n > info["threshold"]).sum()),
                        high_percent=float((n > info["threshold"]).mean() * 100),
                    )
                )
            layers = data["layer_norms"]
            for li in range(layers.shape[1]):
                a = layers[:, li, :]
                mean = float(a.mean())
                std = float(a.std())
                threshold = mean + config["layer_k"] * std
                layer_rows.append(
                    dict(
                        model=key,
                        layer=li,
                        stage="embedding" if li == 0 else "block",
                        k=config["layer_k"],
                        mean_norm=mean,
                        std_norm=std,
                        threshold=threshold,
                        high_count=int((a > threshold).sum()),
                        high_percent=float((a > threshold).mean() * 100),
                        max_norm=float(a.max()),
                    )
                )
            # Every last-layer patch norm is available in a human-readable CSV.
            raw_norms = data["final_norms"]
            write_csv(
                results / f"data/{key}_patch_norms.csv",
                (
                    dict(
                        image_index=i,
                        patch_row=p // grid,
                        patch_col=p % grid,
                        pre_layernorm_norm=float(raw_norms[i, p]),
                        post_layernorm_norm=float(n),
                        high=bool(n > info["threshold"]),
                    )
                    for i, ns in enumerate(norms)
                    for p, n in enumerate(ns)
                ),
            )
    write_json(results / "data/thresholds.json", threshold_info)
    original_rows = []
    for size in config["sizes"]:
        for suffix in ["no_registers", "registers"]:
            key = f"{size}_{suffix}"
            d = np.load(results / f"data/{key}.npz")
            for t in config["original_thresholds"]:
                n = d["normalized_norms"]
                original_rows.append(
                    dict(
                        model=key,
                        feature_space="post_layernorm",
                        threshold=t,
                        high_count=int((n > t).sum()),
                        high_percent=float((n > t).mean() * 100),
                    )
                )
    write_csv(results / "data/original_threshold_diagnostic.csv", original_rows)
    for name, rows in [
        ("summary", all_stats),
        ("threshold_sensitivity", threshold_rows),
        ("per_image", image_rows),
        ("layer_statistics", layer_rows),
        ("redundancy", redundancy_rows),
    ]:
        write_csv(results / f"data/{name}.csv", rows)
    write_scale_adjusted_tails(config, results)
    return all_stats, layer_rows, threshold_info


def write_scale_adjusted_tails(config: Config, results: Path) -> None:
    """Check whether tails persist after dividing by each model's median."""
    rows = []
    for spec in config["models"]:
        with np.load(results / f"data/{spec['key']}.npz") as data:
            norms = data["normalized_norms"]
        median = float(np.median(norms))
        cutoff = config["relative_threshold_multiple"] * median
        rows.append(
            {
                "model": spec["key"],
                "median_norm": median,
                "max_to_median": float(norms.max() / median),
                "relative_threshold_multiple": config["relative_threshold_multiple"],
                "relative_threshold": cutoff,
                "high_count": int((norms > cutoff).sum()),
                "high_percent": float((norms > cutoff).mean() * 100),
            }
        )
    write_csv(results / "data/scale_adjusted_tail.csv", rows)
