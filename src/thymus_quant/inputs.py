from __future__ import annotations

"""Input loading and validation for CT images, masks, and geometry."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .exceptions import InputValidationError, MissingGeometryError


@dataclass(slots=True)
class ImageContext:
    """Study-level CT image context shared by segmentation members.

    `ct_hu` is a 3D CT array in Hounsfield units. `spacing_mm` is `(sx, sy, sz)`
    in millimeters on the same grid as the masks.
    """

    ct_hu: Any
    spacing_mm: tuple[float, float, float]
    affine: Any | None = None
    source: str | None = None
    orientation: tuple[str, str, str] | None = None
    geometry: dict[str, Any] | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(np.asarray(self.ct_hu).shape)  # type: ignore[return-value]


def _orientation_from_affine(affine: Any | None) -> tuple[str, str, str] | None:
    if affine is None:
        return None
    try:
        return tuple(str(x) for x in nib.aff2axcodes(np.asarray(affine)))  # type: ignore[return-value]
    except Exception:
        return None


def geometry_report(
    affine: Any | None,
    *,
    spacing_mm: Any | None = None,
    image: Any | None = None,
) -> dict[str, Any] | None:
    """Describe the voxel-to-world geometry of an input for QC.

    Nothing here changes the image. The values are recorded so that oblique,
    sheared or inconsistent inputs can be found later:

    - ``voxel_sizes_affine_mm``: column norms of the affine.
    - ``voxel_size_mismatch_max_mm``: largest difference between ``spacing_mm``
      (header zooms) and the affine column norms.
    - ``obliquity_max_deg``: largest angle between a voxel axis and the nearest
      world axis (``nibabel.affines.obliquity``).
    - ``shear_max``: largest off-diagonal term of the direction-cosine Gram
      matrix (0 for an orthogonal grid).
    - ``qform_code`` / ``sform_code`` and ``qform_sform_max_abs_diff`` when a
      NIfTI image is given and both transforms are set.
    """
    if affine is None:
        return None
    try:
        aff = np.asarray(affine, dtype=float)
        if aff.shape != (4, 4) or not np.all(np.isfinite(aff)):
            return {"valid_affine": False}
        rzs = aff[:3, :3]
        sizes = np.linalg.norm(rzs, axis=0)
        if not np.all(sizes > 0):
            return {"valid_affine": False}
        cosines = rzs / sizes
        report: dict[str, Any] = {
            "valid_affine": True,
            "voxel_sizes_affine_mm": [float(v) for v in sizes],
            "obliquity_max_deg": float(np.degrees(np.max(np.abs(nib.affines.obliquity(aff))))),
            "shear_max": float(np.max(np.abs(cosines.T @ cosines - np.eye(3)))),
        }
        if spacing_mm is not None:
            zooms = np.asarray(spacing_mm, dtype=float)[:3]
            report["voxel_size_mismatch_max_mm"] = float(np.max(np.abs(zooms - sizes)))
        if image is not None and hasattr(image, "get_qform") and hasattr(image, "get_sform"):
            qform, qcode = image.get_qform(coded=True)
            sform, scode = image.get_sform(coded=True)
            report["qform_code"] = int(qcode)
            report["sform_code"] = int(scode)
            if qform is not None and sform is not None and int(qcode) > 0 and int(scode) > 0:
                report["qform_sform_max_abs_diff"] = float(np.max(np.abs(np.asarray(qform) - np.asarray(sform))))
        return report
    except Exception:
        return {"valid_affine": False}


def validate_spacing(spacing_mm: tuple[float, float, float] | list[float] | np.ndarray | None) -> tuple[float, float, float]:
    """Validate and normalize 3-element positive finite spacing in mm."""
    if spacing_mm is None:
        raise MissingGeometryError("spacing_mm is required for volume/ETV computation")
    arr = np.asarray(spacing_mm, dtype=float)
    if arr.shape != (3,):
        raise InputValidationError(f"spacing_mm must contain exactly 3 values; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise InputValidationError(f"spacing_mm must be finite; got {tuple(arr.tolist())}")
    if not np.all(arr > 0):
        raise InputValidationError(f"spacing_mm must be positive; got {tuple(arr.tolist())}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def validate_ct_hu(ct_hu: Any) -> np.ndarray:
    """Validate a 3D CT array and return float copy/view."""
    ct = np.asarray(ct_hu, dtype=np.float32)
    if ct.ndim != 3:
        raise InputValidationError(f"ct_hu must be a 3D array; got {ct.ndim}D with shape {ct.shape}")
    if ct.size == 0:
        raise InputValidationError("ct_hu must not be empty")
    if not np.isfinite(ct).any():
        raise InputValidationError("ct_hu must contain at least one finite HU value")
    return ct


def load_image_context(image: Any, *, source: str | None = None) -> ImageContext:
    """Load a path-like or nibabel image into a validated ImageContext."""
    if isinstance(image, ImageContext):
        return image
    if isinstance(image, nib.spatialimages.SpatialImage):
        img = image
        src = source
    else:
        src = str(image) if source is None else source
        img = nib.load(str(image))

    ct = validate_ct_hu(np.asarray(img.get_fdata(), dtype=np.float32))
    spacing = validate_spacing(img.header.get_zooms()[:3])
    return ImageContext(
        ct_hu=ct,
        spacing_mm=spacing,
        affine=np.asarray(img.affine),
        source=src,
        orientation=_orientation_from_affine(img.affine),
        geometry=geometry_report(img.affine, spacing_mm=spacing, image=img),
    )


def make_image_context(
    *,
    ct_hu: Any,
    spacing_mm: tuple[float, float, float] | list[float] | np.ndarray,
    affine: Any | None = None,
    source: str | None = None,
    orientation: tuple[str, str, str] | None = None,
) -> ImageContext:
    """Build a validated ImageContext from arrays."""
    ct = validate_ct_hu(ct_hu)
    spacing = validate_spacing(spacing_mm)
    orient = orientation if orientation is not None else _orientation_from_affine(affine)
    return ImageContext(
        ct_hu=ct,
        spacing_mm=spacing,
        affine=affine,
        source=source,
        orientation=orient,
        geometry=geometry_report(affine, spacing_mm=spacing),
    )


def validate_binary_mask(mask: Any, *, ct_shape: tuple[int, int, int] | None = None, name: str = "mask") -> np.ndarray:
    """Validate a 3D bool or 0/1 mask and return bool array."""
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise InputValidationError(f"{name} must be a 3D array; got {arr.ndim}D with shape {arr.shape}")
    if ct_shape is not None and tuple(arr.shape) != tuple(ct_shape):
        raise InputValidationError(f"{name} shape {arr.shape} must match ct_hu shape {ct_shape}")
    if arr.dtype == np.bool_:
        out = arr.astype(bool, copy=False)
    else:
        finite = np.isfinite(arr)
        if not np.all(finite):
            raise InputValidationError(f"{name} contains non-finite values")
        # Linear-time 0/1 check; np.unique (a full sort) is only needed for the error message.
        if not np.all((arr == 0) | (arr == 1)):
            uniq = np.unique(arr)
            raise InputValidationError(f"{name} must be bool or binary 0/1 data; got values {uniq[:8].tolist()}")
        out = arr.astype(bool)
    if not out.any():
        raise InputValidationError(f"{name} must not be empty")
    return out


def validate_mask_has_finite_ct(mask: np.ndarray, ct_hu: np.ndarray, *, name: str = "mask") -> None:
    """Ensure a mask contains at least one finite CT value."""
    vals = np.asarray(ct_hu)[np.asarray(mask, dtype=bool)]
    if vals.size == 0 or not np.isfinite(vals).any():
        raise InputValidationError(f"{name} contains no finite CT HU values")


def source_from_input(image: Any) -> str | None:
    """Best-effort public input source string without confusing it with study_id."""
    if isinstance(image, ImageContext):
        return image.source
    if isinstance(image, (str, Path)):
        return str(image)
    return None
