from __future__ import annotations

"""Input loading and validation for CT images, masks, and geometry."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .exceptions import InputValidationError, MissingGeometryError

logger = logging.getLogger(__name__)

# A chest CT in HU contains air and lung, so at least 1% of its voxels are far
# below -500 HU. When the 1st percentile is higher, the values are probably not
# HU (for example, the rescale intercept of -1024 was not applied, or the image
# was windowed or normalized).
HU_CHECK_PERCENTILE = 1.0
HU_CHECK_MAX_LOW_HU = -500.0
# Relative difference allowed between the voxel volume from spacing_mm (header
# voxel sizes) and the voxel volume of the affine before a warning is given.
VOXEL_VOLUME_REL_TOL = 1e-3


@dataclass(slots=True)
class ImageContext:
    """Study-level CT image context shared by segmentation members.

    `ct_hu` is a 3D CT array in Hounsfield units. `spacing_mm` is `(sx, sy, sz)`
    in millimeters on the same grid as the masks. `warnings` holds notes about
    the input (values that do not look like HU, voxel sizes that disagree with
    the affine); they are copied to the analysis result and do not change it.
    """

    ct_hu: Any
    spacing_mm: tuple[float, float, float]
    affine: Any | None = None
    source: str | None = None
    orientation: tuple[str, str, str] | None = None
    geometry: dict[str, Any] | None = None
    intensity: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()

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
    - ``voxel_volume_affine_mm3`` and ``voxel_volume_mismatch_rel``: voxel volume
      of the affine (``|det|``) and its relative difference from the product of
      ``spacing_mm``, which is used for volumes. Shear (gantry tilt) and
      rotation do not change the voxel volume, so they do not count as a
      mismatch here.
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
            vol_affine = float(abs(np.linalg.det(rzs)))
            report["voxel_volume_affine_mm3"] = vol_affine
            if vol_affine > 0:
                report["voxel_volume_mismatch_rel"] = float(abs(float(np.prod(zooms)) - vol_affine) / vol_affine)
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


def intensity_report(ct_hu: Any) -> dict[str, Any] | None:
    """Summarize CT values to find inputs that are probably not in HU.

    Percentiles are taken on every 4th voxel in-plane and every 2nd slice.
    ``looks_like_hu`` is False when the 1st percentile is above -500 HU.
    Nothing here changes the image.
    """
    ct = np.asarray(ct_hu)
    if ct.ndim != 3 or ct.size == 0:
        return None
    sample = ct[::4, ::4, ::2]
    vals = sample[np.isfinite(sample)]
    if vals.size == 0:
        vals = ct[np.isfinite(ct)]
    if vals.size == 0:
        return None
    p_low, p50, p99 = (float(v) for v in np.percentile(vals, [HU_CHECK_PERCENTILE, 50.0, 99.0]))
    return {
        "hu_p01": p_low,
        "hu_p50": p50,
        "hu_p99": p99,
        "looks_like_hu": bool(p_low <= HU_CHECK_MAX_LOW_HU),
    }


def input_warnings(
    *,
    spacing_mm: tuple[float, float, float],
    geometry: dict[str, Any] | None,
    intensity: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Notes about an input that may make the results wrong (results are not changed)."""
    notes: list[str] = []
    if intensity is not None and intensity.get("looks_like_hu") is False:
        notes.append(
            f"CT values do not look like HU: the 1st percentile is {intensity['hu_p01']:.0f} "
            f"(median {intensity['hu_p50']:.0f}), but a chest CT in HU has air and lung below "
            f"{HU_CHECK_MAX_LOW_HU:.0f} HU. Check that the rescale slope/intercept were applied and "
            "that the image was not windowed, normalized or cropped. Results may be wrong."
        )
    rel = None if geometry is None else geometry.get("voxel_volume_mismatch_rel")
    if rel is not None and rel > VOXEL_VOLUME_REL_TOL:
        sizes = ", ".join(f"{v:.4g}" for v in spacing_mm)
        affine_sizes = ", ".join(f"{v:.4g}" for v in geometry.get("voxel_sizes_affine_mm", []))
        notes.append(
            f"Voxel sizes ({sizes} mm, from the header) disagree with the affine (column norms {affine_sizes} mm): "
            f"the voxel volumes differ by {100.0 * rel:.2f}%. Volumes and ETV use the header voxel sizes; "
            "check how the file was written."
        )
    for note in notes:
        logger.warning(note)
    return tuple(notes)


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
    geometry = geometry_report(img.affine, spacing_mm=spacing, image=img)
    intensity = intensity_report(ct)
    return ImageContext(
        ct_hu=ct,
        spacing_mm=spacing,
        affine=np.asarray(img.affine),
        source=src,
        orientation=_orientation_from_affine(img.affine),
        geometry=geometry,
        intensity=intensity,
        warnings=input_warnings(spacing_mm=spacing, geometry=geometry, intensity=intensity),
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
    geometry = geometry_report(affine, spacing_mm=spacing)
    intensity = intensity_report(ct)
    return ImageContext(
        ct_hu=ct,
        spacing_mm=spacing,
        affine=affine,
        source=source,
        orientation=orient,
        geometry=geometry,
        intensity=intensity,
        warnings=input_warnings(spacing_mm=spacing, geometry=geometry, intensity=intensity),
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
