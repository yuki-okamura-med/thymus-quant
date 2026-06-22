from __future__ import annotations

"""Minimal common utilities shared across quantification methods."""

import nibabel as nib
import numpy as np

from ..exceptions import MissingGeometryError
from ..inputs import validate_spacing


def spacing_voxel_volume_ml(spacing_mm) -> float:
    """Return one-voxel volume in mL from validated spacing in mm."""
    sx, sy, sz = validate_spacing(spacing_mm)
    return float((sx * sy * sz) / 1000.0)


def voxvol_ml(mask: np.ndarray, img: nib.spatialimages.SpatialImage | None = None, spacing_mm=None) -> float:
    """Convert mask voxel count to physical volume in mL.

    Spacing is required; silent NaN volume is not allowed.
    """
    if spacing_mm is None:
        if img is None:
            raise MissingGeometryError("spacing_mm or image header spacing is required for volume/ETV computation")
        spacing_mm = img.header.get_zooms()[:3]
    voxel_ml = spacing_voxel_volume_ml(spacing_mm)
    return float((mask > 0).sum() * voxel_ml)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice similarity coefficient for binary masks."""
    a = a > 0
    b = b > 0
    den = a.sum() + b.sum()
    return 1.0 if den == 0 else float(2.0 * (a & b).sum() / den)
