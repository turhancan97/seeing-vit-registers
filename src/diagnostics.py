"""Investigate the Large raw-tail reversal without claiming an untested cause."""

from pathlib import Path
from typing import Any
import numpy as np
from .config import Config
from .runtime import write_csv, write_json


def investigate_large_tail(config: Config, sources: list[dict[str, Any]], results: Path) -> None:
    """Compare scale-invariant rules, image concentration, coordinates and tokens."""
    rows, images, coordinates, top_tokens = [], [], [], []
    details = {}
    for suffix in ["no_registers", "registers"]:
        key = f"large_{suffix}"
        data = np.load(results / f"data/{key}.npz")
        a = data["final_norms"]
        mu, sd = float(a.mean()), float(a.std())
        median = float(np.median(a))
        mad = float(np.median(np.abs(a - median)))
        high = a > mu + config["layer_k"] * sd
        normalized = data["normalized_norms"]
        art = normalized > config["threshold"]["value"]
        rescaled = a / median
        assert np.array_equal(high, rescaled > rescaled.mean() + config["layer_k"] * rescaled.std())
        for k in config["tail_diagnostics"]["k_values"]:
            t = mu + k * sd
            mask = a > t
            rows.append(
                dict(
                    model=key,
                    feature_space="raw",
                    rule="mean+k*std",
                    parameter=k,
                    threshold=t,
                    count=int(mask.sum()),
                    percent=float(mask.mean() * 100),
                )
            )
        for multiple in config["tail_diagnostics"]["median_multiples"]:
            t = multiple * median
            mask = a > t
            rows.append(
                dict(
                    model=key,
                    feature_space="raw",
                    rule="multiple*median",
                    parameter=multiple,
                    threshold=t,
                    count=int(mask.sum()),
                    percent=float(mask.mean() * 100),
                )
            )
        t = median + config["tail_diagnostics"]["mad_k"] * 1.4826 * mad
        rows.append(
            dict(
                model=key,
                feature_space="raw",
                rule="median+k*1.4826*MAD",
                parameter=config["tail_diagnostics"]["mad_k"],
                threshold=t,
                count=int((a > t).sum()),
                percent=float((a > t).mean() * 100),
            )
        )
        per_image = high.sum(axis=1)
        freq = high.sum(axis=0)
        center = data["raw_centered_norms"]
        means = data["raw_feature_mean"]
        dc = data["features"].shape[-1] * means**2 / np.maximum(a**2, 1e-12)
        for i, n in enumerate(per_image):
            local = a[i] > a[i].mean() + config["layer_k"] * a[i].std()
            images.append(
                dict(
                    model=key,
                    feature_space="raw",
                    image_index=i,
                    image=sources[i]["path"],
                    pooled_rule_count=int(n),
                    per_image_rule_count=int(local.sum()),
                    normalized_reference_count=int(art[i].sum()),
                    max_norm=float(a[i].max()),
                )
            )
        for j, n in enumerate(freq):
            if n:
                coordinates.append(
                    dict(
                        model=key,
                        feature_space="raw",
                        patch_row=j // 37,
                        patch_col=j % 37,
                        images_above_threshold=int(n),
                    )
                )
        for i, j in zip(*np.where(high)):
            top_tokens.append(
                dict(
                    model=key,
                    feature_space="raw",
                    image_index=int(i),
                    patch_row=int(j // 37),
                    patch_col=int(j % 37),
                    raw_norm=float(a[i, j]),
                    normalized_norm=float(normalized[i, j]),
                    raw_zscore=float((a[i, j] - mu) / sd),
                    mean_offset_energy_fraction=float(dc[i, j]),
                )
            )
        details[key] = {
            "feature_space": "raw",
            "mean": mu,
            "std": sd,
            "median": median,
            "maximum": float(a.max()),
            "max_to_median": float(a.max() / median),
            "pooled_3sigma_count": int(high.sum()),
            "pooled_3sigma_percent": float(high.mean() * 100),
            "images_with_tail": int((per_image > 0).sum()),
            "largest_image_share_percent": float(per_image.max() / high.sum() * 100),
            "top3_image_share_percent": float(np.sort(per_image)[-3:].sum() / high.sum() * 100),
            "unique_patch_coordinates": int((freq > 0).sum()),
            "max_coordinate_image_frequency": int(freq.max()),
            "normalized_reference_overlap_count": int((high & art).sum()),
            "mean_offset_energy_fraction_tail": float(dc[high].mean()),
            "mean_offset_energy_fraction_other": float(dc[~high].mean()),
            "centered_norm_tail_mean": float(center[high].mean()),
            "scale_invariance_verified": True,
        }
    for name, table in [
        ("large_tail_sensitivity", rows),
        ("large_tail_per_image", images),
        ("large_tail_coordinates", coordinates),
        ("large_tail_tokens", top_tokens),
    ]:
        write_csv(results / f"data/{name}.csv", table)
    write_json(results / "data/large_tail_investigation.json", details)


def feature_space_transform(config: Config, results: Path) -> None:
    """Track raw high-gap patches into normalized high and low populations."""
    from .norms import choose_threshold, choose_lower_threshold

    rows = []
    for spec in config["models"]:
        d = np.load(results / f"data/{spec['key']}.npz")
        raw = d["final_norms"]
        normal = d["normalized_norms"]
        upper = choose_threshold(raw, config)
        lower = choose_lower_threshold(normal, config)
        high = raw > upper["threshold"] if upper["clear_gap"] else np.zeros_like(raw, dtype=bool)
        low = (
            normal < lower["threshold"] if lower["clear_gap"] else np.zeros_like(normal, dtype=bool)
        )
        rows.append(
            dict(
                model=spec["key"],
                source_feature_space="raw",
                target_feature_space="normalized",
                raw_high_gap_count=int(high.sum()),
                normalized_low_gap_count=int(low.sum()),
                raw_high_to_normalized_low_overlap=int((high & low).sum()),
                normalized_low_gap_threshold=lower["threshold"],
                normalized_median_for_raw_high=float(np.median(normal[high]))
                if high.any()
                else None,
                normalized_median_for_other_raw=float(np.median(normal[~high])),
            )
        )
    write_csv(results / "data/feature_space_transform.csv", rows)
