from __future__ import annotations

"""Minimal common utilities shared across quantification methods."""

import nibabel as nib
import numpy as np


def voxvol_ml(mask: np.ndarray, img: nib.spatialimages.SpatialImage | None) -> float:
    """Convert mask voxel count to physical volume in mL.

    If image spacing is unavailable (`img is None`), return ``NaN``.
    """
    if img is None:
        return float("nan")
    zoom = img.header.get_zooms()[:3]
    return float((mask > 0).sum() * (zoom[0] * zoom[1] * zoom[2]) / 1000.0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice similarity coefficient for binary masks."""
    a = a > 0
    b = b > 0
    den = a.sum() + b.sum()
    return 1.0 if den == 0 else float(2.0 * (a & b).sum() / den)
