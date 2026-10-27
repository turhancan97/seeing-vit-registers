"""Explicit names and array keys for the two measured feature spaces."""

SPACES = {
    "raw": {
        "label": "Raw residual stream · before final LayerNorm",
        "norms": "final_norms",
        "layers": "layer_norms",
        "features": "raw_features",
        "pca": "raw_pca_rgb",
        "cosine": "raw_cosine",
        "prefix": "prefix_norms_raw",
    },
    "normalized": {
        "label": "Normalized output · after final LayerNorm",
        "norms": "normalized_norms",
        "layers": "layer_normalized_norms",
        "features": "features",
        "pca": "pca_rgb",
        "cosine": "cosine",
        "prefix": "prefix_norms_normalized",
    },
}


def reference_threshold(config: dict, space: str) -> float:
    """Use the frozen normalized cutoff or the explicitly descriptive raw cutoff."""
    return config["threshold"]["value"] if space == "normalized" else config["raw_norm_threshold"]
