"""Independent per-image PCA projections of final normalized patch features."""

import numpy as np
from sklearn.decomposition import PCA


def pca_rgb(features: np.ndarray, grid_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit three full-SVD components and min–max scale each RGB channel."""
    pca = PCA(n_components=3, svd_solver="full", random_state=seed)
    projected = pca.fit_transform(features)
    low, high = projected.min(axis=0), projected.max(axis=0)
    rgb = ((projected - low) / np.maximum(high - low, 1e-8)).reshape(grid_size, grid_size, 3)
    return rgb, pca.explained_variance_ratio_
