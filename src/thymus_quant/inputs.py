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
# Relative difference allowed between each header voxel size and the matching
# axis length of the affine after removing shear (QR decomposition).
VOXEL_SIZE_REL_TOL = 1e-3
# NIfTI spatial units (xyzt_units) and their size in mm. "unknown" is taken as mm.
_UNIT_TO_MM = {"mm": 1.0, "unknown": 1.0, "meter": 1000.0, "micron": 0.001}


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

    def __post_init__(self) -> None:
        # Built directly (not by load_image_context/make_image_context): read the orientation from the affine,
        # as the loaders do, instead of silently assuming the model orientation.
        if self.affine is not None and self.geometry is None:
            self.geometry = geometry_report(self.affine, spacing_mm=self.spacing_mm)
            if self.orientation is None and self.geometry.get("valid_affine"):
                self.orientation = _orientation_from_affine(self.affine)
                self.geometry["orientation_source"] = "affine"

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
    - ``voxel_sizes_affine_unsheared_mm`` and ``voxel_size_mismatch_rel_max``: axis
      lengths of the affine with shear removed (QR decomposition), and the largest
      relative difference of ``spacing_mm`` from them. This is the voxel size check.
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
      NIfTI image is given and both transforms are set, and
      ``qform_sform_handedness_differs`` (the two disagree on left and right).
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
        vol_affine = float(abs(np.linalg.det(rzs)))
        if not vol_affine > 0:
            return {"valid_affine": False}
        # Axis lengths with shear removed: the voxel edge of each axis perpendicular to the earlier axes.
        unsheared = np.abs(np.diag(np.linalg.qr(rzs)[1]))
        report["voxel_sizes_affine_unsheared_mm"] = [float(v) for v in unsheared]
        report["voxel_volume_affine_mm3"] = vol_affine
        if spacing_mm is not None:
            zooms = np.asarray(spacing_mm, dtype=float)[:3]
            report["voxel_size_mismatch_max_mm"] = float(np.max(np.abs(zooms - sizes)))
            report["voxel_size_mismatch_rel_max"] = float(np.max(np.abs(zooms - unsheared) / unsheared))
            report["voxel_volume_mismatch_rel"] = float(abs(float(np.prod(zooms)) - vol_affine) / vol_affine)
        if image is not None and hasattr(image, "get_qform") and hasattr(image, "get_sform"):
            qform, qcode = image.get_qform(coded=True)
            sform, scode = image.get_sform(coded=True)
            report["qform_code"] = int(qcode)
            report["sform_code"] = int(scode)
            if qform is not None and sform is not None and int(qcode) > 0 and int(scode) > 0:
                report["qform_sform_max_abs_diff"] = float(np.max(np.abs(np.asarray(qform) - np.asarray(sform))))
                q_det = float(np.linalg.det(np.asarray(qform, dtype=float)[:3, :3]))
                s_det = float(np.linalg.det(np.asarray(sform, dtype=float)[:3, :3]))
                report["qform_sform_handedness_differs"] = bool(np.sign(q_det) != np.sign(s_det))
        return report
    except Exception:
        return {"valid_affine": False}


def intensity_report(ct_hu: Any) -> dict[str, Any] | None:
    """Summarize CT values to find inputs that are probably not in HU.

    Percentiles are taken on every 4th voxel in-plane and every 2nd slice.
    ``looks_like_hu`` is False when the 1st percentile is above -500 HU.
    ``n_nonfinite`` counts NaN/inf values in the whole volume.
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
        "n_nonfinite": int(np.count_nonzero(~np.isfinite(ct))),
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
    if intensity is not None and intensity.get("n_nonfinite"):
        notes.append(
            f"CT has {intensity['n_nonfinite']} non-finite values (NaN/inf). They are passed to the segmentation "
            "network as they are, and a member whose TRQ contains any of them is not computed."
        )
    geo = geometry or {}
    rel = geo.get("voxel_size_mismatch_rel_max")
    if rel is not None and rel > VOXEL_SIZE_REL_TOL:
        sizes = ", ".join(f"{v:.4g}" for v in spacing_mm)
        affine_sizes = ", ".join(f"{v:.4g}" for v in geo.get("voxel_sizes_affine_unsheared_mm", []))
        notes.append(
            f"Voxel sizes ({sizes} mm, from the header) disagree with the affine (axis lengths without shear "
            f"{affine_sizes} mm; largest difference {100.0 * rel:.2f}%, voxel volume {100.0 * geo.get('voxel_volume_mismatch_rel', 0.0):.2f}%). "
            "Volumes and ETV use the header voxel sizes; check how the file was written."
        )
    if geo.get("valid_affine") is False:
        notes.append(
            "The affine is not usable (non-finite or singular), so the voxel order (orientation) is unknown; "
            "the voxel array is used as stored (assumed LAS for TRQseg-v1)."
        )
    if geo.get("orientation_source") == "fallback":
        notes.append(
            "NIfTI orientation is unspecified (qform_code = sform_code = 0). nibabel then assumes LAS, which is not "
            "information from the file; the voxel array is used as stored (assumed LAS for TRQseg-v1). Check the "
            "source image geometry."
        )
    if geo.get("qform_sform_handedness_differs"):
        notes.append(
            "The qform and sform of this NIfTI disagree on left and right (opposite handedness). nibabel uses the "
            "sform; other tools may use the qform. Check the file's orientation."
        )
    if geo.get("unit_scale_to_mm", 1.0) != 1.0:
        notes.append(
            f"NIfTI spatial unit is {geo.get('spatial_unit')}; voxel sizes and the affine were converted to mm "
            f"(x {geo['unit_scale_to_mm']:g})."
        )
    if geo.get("default_like_header"):
        notes.append(
            "The NIfTI geometry looks like library defaults (1 mm voxels, axis-aligned unit affine, origin 0). This is "
            "valid for data resampled that way, but also appears when spatial information is lost while saving an "
            "array (for example nibabel.Nifti1Image(arr, np.eye(4)) or SimpleITK.GetImageFromArray without copying "
            "spacing and direction). Check that the voxel sizes and orientation are real."
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
    unit = "unknown"
    if hasattr(img.header, "get_xyzt_units"):
        unit = str(img.header.get_xyzt_units()[0] or "unknown")
    scale = _UNIT_TO_MM.get(unit, 1.0)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    spacing = validate_spacing(zooms * scale)
    affine = np.asarray(img.affine, dtype=float).copy()
    affine[:3, :] *= scale
    geometry = geometry_report(affine, spacing_mm=spacing, image=img)
    geometry["spatial_unit"] = unit
    geometry["unit_scale_to_mm"] = scale
    orientation = _orientation_from_affine(affine) if geometry.get("valid_affine") else None
    geometry["orientation_source"] = "affine" if orientation is not None else None
    if geometry.get("qform_code") == 0 and geometry.get("sform_code") == 0:
        # No orientation is given in the file; nibabel's fallback affine assumes LAS.
        orientation = None
        geometry["orientation_source"] = "fallback"
    geometry["default_like_header"] = bool(
        geometry.get("valid_affine")
        and np.allclose(np.abs(affine[:3, :3]), np.eye(3), atol=1e-6)
        and np.allclose(affine[:3, 3], 0.0, atol=1e-6)
        and np.allclose(zooms * scale, 1.0, atol=1e-6)
    )
    intensity = intensity_report(ct)
    return ImageContext(
        ct_hu=ct,
        spacing_mm=spacing,
        affine=affine,
        source=src,
        orientation=orientation,
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
    geometry = geometry_report(affine, spacing_mm=spacing)
    from_affine = _orientation_from_affine(affine) if geometry is not None and geometry.get("valid_affine") else None
    if orientation is not None:
        orient = tuple(str(c) for c in orientation)
        if from_affine is not None and orient != from_affine:
            raise InputValidationError(
                f"orientation {orient} contradicts the affine, whose voxel order is {from_affine}; give one of them"
            )
    else:
        orient = from_affine
    if geometry is not None:
        geometry["orientation_source"] = "explicit" if orientation is not None else ("affine" if from_affine else None)
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
