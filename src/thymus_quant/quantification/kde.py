from __future__ import annotations

"""Exact, faster drop-in for 1-D unweighted ``scipy.stats.gaussian_kde``.

CT attenuation values are integers in HU, so a TRQ with 10^5 voxels usually has
only a few hundred distinct values. ``scipy.stats.gaussian_kde`` sums one kernel
per voxel; this class sums one kernel per distinct value, weighted by its count.
The density and the Scott bandwidth (unbiased variance x n^(-2/5)) are the same
expressions as SciPy's, so results agree up to floating-point summation order.
Cost drops from O(n_voxels x n_grid) to O(n_distinct x n_grid).
"""

import numpy as np


class GroupedGaussianKDE:
    """1-D Gaussian KDE with Scott's rule, equivalent to ``scipy.stats.gaussian_kde(values)``.

    Only the unweighted 1-D case used by the Okamura method is supported.
    """

    def __init__(self, dataset) -> None:
        x = np.asarray(dataset, dtype=float).ravel()
        n = x.size
        if n < 2:
            raise ValueError("`dataset` input should have multiple elements.")
        values, counts = np.unique(x, return_counts=True)
        mean = float(np.sum(counts * values) / n)
        variance = float(np.sum(counts * (values - mean) ** 2) / (n - 1))
        if not np.isfinite(variance) or variance <= 0:
            raise np.linalg.LinAlgError("The data appears to lie in a lower-dimensional subspace.")
        self.n = n
        self.factor = n ** (-1.0 / 5.0)
        self.covariance = variance * self.factor**2
        self._values = values
        self._weights = counts / n

    def __call__(self, points) -> np.ndarray:
        return self.evaluate(points)

    def evaluate(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=float).ravel()
        var = self.covariance
        norm = 1.0 / np.sqrt(2.0 * np.pi * var)
        out = np.empty(p.size, dtype=float)
        # Keep the (points x distinct values) block at <= ~2e7 elements.
        step = max(1, int(2e7 // max(self._values.size, 1)))
        for start in range(0, p.size, step):
            d = p[start:start + step, None] - self._values[None, :]
            out[start:start + step] = (np.exp(-0.5 * d * d / var) @ self._weights) * norm
        return out
