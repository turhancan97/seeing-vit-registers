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


def norm_statistics(values: np.ndarray) -> dict[str, float]:
    """Describe magnitude and shape without equating a tail with artifacts."""
    mean, std = float(values.mean()), float(values.std())
    return {
        "mean_norm": mean,
        "std_norm": std,
        "max_norm": float(values.max()),
        "p50": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "mean_plus_3std_threshold": mean + 3 * std,
        "mean_plus_3std_percent": float((values > mean + 3 * std).mean() * 100),
    }


def analyze(
    config: Config, sources: list[dict[str, Any]], results: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Export A/B in both spaces, including individual prefix-token layers."""
    from .spaces import SPACES, reference_threshold

    summary, layer_rows, per_image, sensitivity, prefix_rows, redundancy, scaled = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    token_statistics = []
    thresholds = {}
    for size in config["sizes"]:
        with np.load(results / f"data/{size}_no_registers.npz") as baseline:
            thresholds[size] = {}
            for space, desc in SPACES.items():
                gap = choose_threshold(baseline[desc["norms"]], config)
                gap["lower_gap"] = choose_lower_threshold(baseline[desc["norms"]], config)
                gap.update(
                    feature_space=space,
                    reference_threshold=reference_threshold(config, space),
                    reference_kind="historical normalized cutoff"
                    if space == "normalized"
                    else "fixed raw scale diagnostic",
                )
                thresholds[size][space] = gap
    for spec in config["models"]:
        key = spec["key"]
        data = np.load(results / f"data/{key}.npz")
        for space, desc in SPACES.items():
            norms = data[desc["norms"]]
            layers = data[desc["layers"]]
            prefix = data[desc["prefix"]]
            cutoff = reference_threshold(config, space)
            stats = norm_statistics(norms)
            gap = choose_threshold(norms, config)
            lower_gap = choose_lower_threshold(norms, config)
            summary.append(
                dict(
                    model=key,
                    feature_space=space,
                    images=len(sources),
                    patch_tokens=int(norms.size),
                    **stats,
                    reference_threshold=cutoff,
                    above_reference_count=int((norms > cutoff).sum()),
                    above_reference_percent=float((norms > cutoff).mean() * 100),
                    clear_lower_gap=lower_gap["clear_gap"],
                    lower_gap_threshold=lower_gap["threshold"],
                    below_gap_count=int((norms < lower_gap["threshold"]).sum())
                    if lower_gap["clear_gap"]
                    else None,
                    clear_gap=gap["clear_gap"],
                    gap_ratio=gap["gap_ratio"],
                    gap_threshold=gap["threshold"] if gap["clear_gap"] else None,
                    gap_above_count=int((norms > gap["threshold"]).sum())
                    if gap["clear_gap"]
                    else None,
                    gap_above_percent=float((norms > gap["threshold"]).mean() * 100)
                    if gap["clear_gap"]
                    else None,
                )
            )
            for t in dict.fromkeys(
                [cutoff] + config["fixed_thresholds"] + config["original_thresholds"]
            ):
                sensitivity.append(
                    dict(
                        model=key,
                        feature_space=space,
                        threshold=t,
                        kind="reference" if t == cutoff else "fixed",
                        high_count=int((norms > t).sum()),
                        high_percent=float((norms > t).mean() * 100),
                    )
                )
            median = float(np.median(norms))
            relative = config["relative_threshold_multiple"] * median
            scaled.append(
                dict(
                    model=key,
                    feature_space=space,
                    median_norm=median,
                    max_to_median=float(norms.max() / median),
                    relative_threshold_multiple=config["relative_threshold_multiple"],
                    relative_threshold=relative,
                    high_count=int((norms > relative).sum()),
                    high_percent=float((norms > relative).mean() * 100),
                )
            )
            neighbors = data["embedding_neighbor_cosine"].reshape(norms.shape)
            for name, mask in [("above_reference", norms > cutoff), ("other", norms <= cutoff)]:
                redundancy.append(
                    dict(
                        model=key,
                        feature_space=space,
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
                per_image.append(
                    dict(
                        model=key,
                        feature_space=space,
                        image_index=i,
                        image=sources[i]["path"],
                        **norm_statistics(n),
                        reference_threshold=cutoff,
                        above_reference_count=int((n > cutoff).sum()),
                        above_reference_percent=float((n > cutoff).mean() * 100),
                    )
                )
            for li in range(layers.shape[1]):
                n = layers[:, li]
                s = norm_statistics(n)
                layer_rows.append(
                    dict(
                        model=key,
                        feature_space=space,
                        layer=li,
                        stage="embedding" if li == 0 else "block",
                        k=config["layer_k"],
                        **s,
                        threshold=float(n.mean() + config["layer_k"] * n.std()),
                        high_count=int((n > n.mean() + config["layer_k"] * n.std()).sum()),
                        high_percent=float(
                            (n > n.mean() + config["layer_k"] * n.std()).mean() * 100
                        ),
                    )
                )
                for label, values in [("patches", n)] + [
                    ("CLS" if t == 0 else f"register_{t}", prefix[:, li, t])
                    for t in range(prefix.shape[-1])
                ]:
                    token_statistics.append(
                        dict(
                            model=key,
                            feature_space=space,
                            layer=li,
                            token=label,
                            token_count=int(values.size),
                            **norm_statistics(values),
                        )
                    )
                for token in range(prefix.shape[-1]):
                    for i, value in enumerate(prefix[:, li, token]):
                        prefix_rows.append(
                            dict(
                                model=key,
                                feature_space=space,
                                image_index=i,
                                layer=li,
                                token="CLS" if token == 0 else f"register_{token}",
                                norm=float(value),
                            )
                        )
        raw, normalized = data["final_norms"], data["normalized_norms"]
        grid = config["image_size"] // config["patch_size"]
        write_csv(
            results / f"data/{key}_patch_norms.csv",
            (
                dict(
                    image_index=i,
                    patch_row=j // grid,
                    patch_col=j % grid,
                    raw_norm=float(raw[i, j]),
                    normalized_norm=float(normalized[i, j]),
                    raw_above_reference=bool(raw[i, j] > reference_threshold(config, "raw")),
                    normalized_above_reference=bool(
                        normalized[i, j] > reference_threshold(config, "normalized")
                    ),
                )
                for i in range(len(sources))
                for j in range(grid**2)
            ),
        )
    relative_rows = []
    for spec in config["models"]:
        with np.load(results / f"data/{spec['key']}.npz") as d:
            for space, desc in SPACES.items():
                n = d[desc["norms"]]
                for multiple in config["tail_diagnostics"]["median_multiples"]:
                    t = multiple * float(np.median(n))
                    relative_rows.append(
                        dict(
                            model=spec["key"],
                            feature_space=space,
                            multiple=multiple,
                            threshold=t,
                            count=int((n > t).sum()),
                            percent=float((n > t).mean() * 100),
                        )
                    )
    write_csv(results / "data/scale_adjusted_sensitivity.csv", relative_rows)
    write_json(results / "data/thresholds.json", thresholds)
    for name, rows in [
        ("layer_token_statistics", token_statistics),
        ("summary", summary),
        ("layer_statistics", layer_rows),
        ("per_image", per_image),
        ("threshold_sensitivity", sensitivity),
        ("prefix_token_norms", prefix_rows),
        ("redundancy", redundancy),
        ("scale_adjusted_tail", scaled),
    ]:
        write_csv(results / f"data/{name}.csv", rows)
    # Keep the original diagnostic filename, now explicitly covering both spaces.
    write_csv(
        results / "data/original_threshold_diagnostic.csv",
        [r for r in sensitivity if r["threshold"] in config["original_thresholds"]],
    )
    return summary, layer_rows, thresholds


def choose_lower_threshold(values: np.ndarray, config: Config) -> dict[str, Any]:
    """Find a separated low-norm group using the same configured gap criterion."""
    a = np.sort(values.ravel().astype(np.float64))
    qlow, qhigh = config["threshold"]["low_search_quantiles"]
    lo = max(config["threshold"]["minimum_upper_count"] - 1, int(qlow * len(a)))
    hi = int(qhigh * len(a))
    j = lo + int(np.argmax(a[lo + 1 : hi + 1] / a[lo:hi]))
    ratio = float(a[j + 1] / a[j])
    clear = ratio >= config["threshold"]["minimum_gap_ratio"]
    return {
        "clear_gap": clear,
        "gap_ratio": ratio,
        "gap_lower": float(a[j]),
        "gap_upper": float(a[j + 1]),
        "threshold": float(np.sqrt(a[j] * a[j + 1])) if clear else None,
    }
