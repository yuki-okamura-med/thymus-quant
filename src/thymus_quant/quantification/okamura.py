from __future__ import annotations

"""Okamura-method quantification implementation."""

from dataclasses import dataclass
import warnings

import nibabel as nib
import numpy as np

from ..exceptions import InputValidationError, MissingGeometryError, NumericalError
from ..inputs import validate_binary_mask, validate_mask_has_finite_ct
from ..results import AnalysisResultOkamura, DetailLevel, OkamuraMemberResult, OkamuraQC, ResultMeta
from ..segmentors import SegmentationResult
from .common import voxvol_ml
# Same density as scipy.stats.gaussian_kde(values), computed per distinct HU value (see kde.py).
# Kept under the name `gaussian_kde` so tests and callers can still patch it here.
from .kde import GroupedGaussianKDE as gaussian_kde


@dataclass(slots=True)
class OkamuraOptions:
    """Configuration for the Okamura quantification method."""

    aadipose_hu: float = -110.0
    athymic_hu: float = 80.0
    second_peak_ratio_threshold: float = 0.5
    js_divergence_threshold: float = 0.1
    pairwise_dsc_threshold: float = 0.7
    hu_variance_threshold: float = 20.0
    delta_hu_margin: float = 1.0
    invalidate_if_any_member_invalid: bool = True
    apply_qc: bool = True


# Definitions follow the analysis code of the paper (Okamura YT et al., Ann Biomed Eng 2025):
# the KDE is fitted on every TRQ voxel and evaluated on fixed HU grids inside [-300, 300] HU.
# Peaks are strict interior local maxima on the grid (scipy.signal.argrelextrema(d, np.greater)).
KDE_WINDOW_HU: tuple[float, float] = (-300.0, 300.0)
# Grid for the QC quantities: second-peak ratio, mode used for the HU value variance, and the
# Jensen-Shannon distance (as in the paper).
QC_GRID_PER_HU = 1
# Grid for A_TRQ (the reported mode). The paper used the 1 HU grid; 0.1 HU removes up to
# 0.5 HU of rounding, which matters for fatty TRQs because ETV is proportional to A_TRQ + 110.
MODE_GRID_PER_HU = 10
# Members with fewer TRQ voxels are not computed (the paper's volume_lower_limit).
MIN_MEMBER_VOXELS = 2


def _window_grid(per_hu: int) -> np.ndarray:
    lo, hi = KDE_WINDOW_HU
    return np.linspace(lo, hi, int(round((hi - lo) * per_hu)) + 1)


def _interior_peak_indices(y: np.ndarray) -> np.ndarray:
    """Strict interior local maxima, as scipy.signal.argrelextrema(y, np.greater)."""
    if y.size < 3:
        return np.array([], dtype=int)
    return (np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])) + 1).astype(int)


@dataclass(slots=True)
class MemberKDE:
    """KDE summary of one member's TRQ HU values."""

    mode: float  # A_TRQ: highest peak on the 0.1 HU grid
    mode_qc: float  # highest peak on the 1 HU grid; its variance across members is a QC criterion
    second_peak_ratio: float  # second-highest / highest peak on the 1 HU grid; 0 with one peak
    density_qc: np.ndarray  # KDE on the 1 HU grid, for the Jensen-Shannon distance


def _kde_mode_and_second_ratio(values: np.ndarray) -> MemberKDE:
    """Mode, second-peak ratio and QC density of one member's HU values (paper definitions)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < MIN_MEMBER_VOXELS:
        raise NumericalError(f"KDE needs at least {MIN_MEMBER_VOXELS} finite HU values; got {vals.size}")
    lo, hi = KDE_WINDOW_HU
    grid_qc = _window_grid(QC_GRID_PER_HU)

    uniq = np.unique(vals)
    if uniq.size == 1:
        # gaussian_kde cannot be fitted to a constant region; its single value is the mode.
        v = float(uniq[0])
        if not lo < v < hi:
            raise NumericalError(f"constant TRQ value {v} HU is outside {KDE_WINDOW_HU} HU")
        density = np.exp(-0.5 * (grid_qc - v) ** 2)  # narrow kernel, used only for the JS distance
        return MemberKDE(mode=v, mode_qc=v, second_peak_ratio=0.0, density_qc=density)

    try:
        kde = gaussian_kde(vals)
        d_qc = np.asarray(kde(grid_qc), dtype=float)
        d_mode = np.asarray(kde(_window_grid(MODE_GRID_PER_HU)), dtype=float)
    except Exception as e:
        raise NumericalError(f"KDE failed: {type(e).__name__}: {e}") from e
    if not (np.all(np.isfinite(d_qc)) and np.all(np.isfinite(d_mode))):
        raise NumericalError("KDE produced non-finite density values")

    peaks_qc = _interior_peak_indices(d_qc)
    peaks_mode = _interior_peak_indices(d_mode)
    if peaks_qc.size == 0 or peaks_mode.size == 0:
        raise NumericalError(f"KDE has no peak inside {KDE_WINDOW_HU} HU")

    heights = np.sort(d_qc[peaks_qc])[::-1]
    ratio = float(heights[1] / heights[0]) if heights.size >= 2 and heights[0] > 0 else 0.0
    mode_qc = lo + int(peaks_qc[np.argmax(d_qc[peaks_qc])]) / QC_GRID_PER_HU
    mode = lo + int(peaks_mode[np.argmax(d_mode[peaks_mode])]) / MODE_GRID_PER_HU
    return MemberKDE(mode=float(mode), mode_qc=float(mode_qc), second_peak_ratio=ratio, density_qc=d_qc)


# Grid points where both curves are below this are left out of the JS distance (see _js_distance).
_JS_NEGLIGIBLE = 1e-300


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance of two KDE curves, as scipy.spatial.distance.jensenshannon(p, q).

    This is the square root of the JS divergence with the natural logarithm, after scaling each
    curve to sum to 1. The paper's "JS divergence" and its threshold 0.1 refer to this value.

    Grid points where both scaled curves are below 1e-300 are left out. Far in the tail of a
    narrow KDE the values are subnormal, their mean underflows to 0, and SciPy then returns inf;
    the left-out points contribute less than 1e-297 otherwise.
    """
    from scipy.special import rel_entr

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    keep = (p > _JS_NEGLIGIBLE) | (q > _JS_NEGLIGIBLE)
    p, q = p[keep], q[keep]
    m = (p + q) / 2.0
    js = (np.sum(rel_entr(p, m)) + np.sum(rel_entr(q, m))) / 2.0
    return float(np.sqrt(max(js, 0.0)))


def _jsd_from_values(v1: np.ndarray, v2: np.ndarray) -> float:
    """Jensen-Shannon distance between the HU distributions of two members (paper definition)."""
    return _js_distance(_kde_mode_and_second_ratio(v1).density_qc, _kde_mode_and_second_ratio(v2).density_qc)


def _union_bbox(masks: list[np.ndarray]) -> tuple[slice, ...]:
    """Bounding box (as slices) of the union of same-shape masks; the full extent if all are empty."""
    shape = np.asarray(masks[0]).shape
    union = np.zeros(shape, dtype=bool)
    for m in masks:
        union |= np.asarray(m) > 0
    if not union.any():
        return tuple(slice(0, s) for s in shape)
    box = []
    for axis in range(union.ndim):
        other = tuple(k for k in range(union.ndim) if k != axis)
        idx = np.flatnonzero(union.any(axis=other))
        box.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(box)


def _ensemble_qc_values(
    masks: list[np.ndarray | None], kdes: list[MemberKDE | None]
) -> tuple[float | None, float | None, float | None]:
    """Mean pairwise JS distance, mean pairwise DSC and HU value variance over all supplied members.

    As in the paper's analysis code, members that could not be computed take part: their mask
    (empty if it could not be read) counts in the DSC, pairs of two empty masks are skipped, JS
    pairs need a KDE on both sides, and the variance of the 1 HU modes is None when any member
    has no mode.
    """
    present = [x for x in masks if x is not None]
    if not present:
        return None, None, None
    # DSC only depends on voxels inside the union of the masks; crop to its bounding box.
    box = _union_bbox(present)
    cropped = [None if x is None else np.asarray(x)[box] > 0 for x in masks]
    dscs: list[float] = []
    jsds: list[float] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = cropped[i], cropped[j]
            size = (0 if a is None else int(a.sum())) + (0 if b is None else int(b.sum()))
            if size > 0:
                overlap = 0 if a is None or b is None else int((a & b).sum())
                dscs.append(2.0 * overlap / size)
            if kdes[i] is not None and kdes[j] is not None:
                d = _js_distance(kdes[i].density_qc, kdes[j].density_qc)
                if np.isfinite(d):
                    jsds.append(d)
    qc_modes = [None if k is None else k.mode_qc for k in kdes]
    hu_var = float(np.var(qc_modes, ddof=1)) if all(v is not None for v in qc_modes) else None
    return (
        float(np.mean(jsds)) if jsds else None,
        float(np.mean(dscs)) if dscs else None,
        hu_var,
    )


def _extract_member_ct_and_img(member) -> tuple[np.ndarray, nib.spatialimages.SpatialImage | None]:
    """Extract CT array and image context from member payload.

    Raises
    ------
    ValueError
        If CT array is not attached to member raw_output.
    """
    if not isinstance(member.raw_output, dict):
        raise ValueError("CT context is missing in segmentation member raw_output")

    ct = member.raw_output.get("ct_array")
    img = member.raw_output.get("ct_image")
    if ct is None:
        raise ValueError("CT array is missing in segmentation member raw_output['ct_array']")

    return np.asarray(ct), img


def _extract_ct_and_spacing(segmentation: SegmentationResult, member) -> tuple[np.ndarray, tuple[float, float, float], nib.spatialimages.SpatialImage | None]:
    """Extract public ImageContext, with deprecated raw_output fallback."""
    if segmentation.image is not None:
        return np.asarray(segmentation.image.ct_hu), segmentation.image.spacing_mm, None
    if isinstance(member.raw_output, dict) and "ct_array" in member.raw_output:
        warnings.warn(
            "Reading CT from SegmentationMember.raw_output['ct_array'] is deprecated; use SegmentationResult.image.",
            DeprecationWarning,
            stacklevel=2,
        )
        ct, img = _extract_member_ct_and_img(member)
        if img is None:
            raise MissingGeometryError("spacing_mm is required; deprecated raw_output fallback has no ct_image")
        return ct, img.header.get_zooms()[:3], img
    raise MissingGeometryError("SegmentationResult.image with ct_hu and spacing_mm is required for Okamura quantification")


def _as_summary_member(member: OkamuraMemberResult) -> OkamuraMemberResult:
    return OkamuraMemberResult(
        member_id=member.member_id,
        valid=member.valid,
        trq_hu_mode=member.trq_hu_mode,
        trq_volume_ml=member.trq_volume_ml,
        etv_ml=member.etv_ml,
        thymic_tissue_fraction=member.thymic_tissue_fraction,
        etv_fraction_adjusted=member.etv_fraction_adjusted,
        atrq_below_aadipose=member.atrq_below_aadipose,
        second_peak_ratio=member.second_peak_ratio,
        flags=member.flags,
        failure_reason=member.failure_reason,
    )


def quantify_okamura(
    segmentation: SegmentationResult,
    *,
    options: OkamuraOptions,
    detail: DetailLevel,
    meta: ResultMeta | None,
) -> AnalysisResultOkamura:
    """Run Okamura quantification from member-level segmentation outputs."""
    members = list(segmentation.members) if segmentation.members else []
    if not members and segmentation.trq_mask is not None:
        from ..segmentors import SegmentationMember

        members = [SegmentationMember(member_id="single", trq_mask=segmentation.trq_mask)]

    member_results: list[OkamuraMemberResult] = []
    # One entry per member, in member order; None for members whose values could not be computed.
    modes: list[float | None] = []
    vols: list[float | None] = []
    etvs: list[float | None] = []
    thymic_tissue_fractions: list[float | None] = []
    atrq_below_flags: list[bool | None] = []
    valid_mask: list[bool] = []
    computed_mask: list[bool] = []
    kdes: list[MemberKDE | None] = []
    masks: list[np.ndarray | None] = []

    for m in members:
        member_flags: list[str] = []
        failure_reason = None
        mask = None
        try:
            ct, spacing, img = _extract_ct_and_spacing(segmentation, m)
            mask = validate_binary_mask(m.trq_mask, ct_shape=tuple(ct.shape), name=f"member {m.member_id} trq_mask")
            validate_mask_has_finite_ct(mask, ct, name=f"member {m.member_id} trq_mask")
            n_voxels = int(mask.sum())
            if n_voxels < MIN_MEMBER_VOXELS:
                member_flags.append("too_few_voxels")
                raise InputValidationError(f"member {m.member_id} TRQ has {n_voxels} voxel(s); at least {MIN_MEMBER_VOXELS} are needed")
            # Index first, then cast: avoids a float64 copy of the whole CT volume per member.
            vals = np.asarray(ct)[mask].astype(float)
            n_nonfinite = int(np.count_nonzero(~np.isfinite(vals)))
            if n_nonfinite:
                # Not computed rather than measured on the finite part: the volume would still count
                # the missing voxels, and the paper's KDE cannot be fitted with them.
                member_flags.append("nonfinite_hu")
                raise InputValidationError(
                    f"member {m.member_id} TRQ has {n_nonfinite} of {n_voxels} voxels with non-finite CT values"
                )
            kde = _kde_mode_and_second_ratio(vals)
            vol = voxvol_ml(mask, img, spacing_mm=spacing)
            computed = True
        except Exception as e:
            kde = None
            vol = None
            computed = False
            failure_reason = f"{type(e).__name__}: {e}"
            member_flags.insert(0, "computation_failed")
            if isinstance(e, NumericalError):
                member_flags.append("kde_failed")

        masks.append(mask)
        kdes.append(kde)

        if not computed:
            member_results.append(
                OkamuraMemberResult(
                    member_id=m.member_id,
                    valid=False,
                    flags=tuple(member_flags),
                    failure_reason=failure_reason,
                    trq_mask=(m.trq_mask if detail == "full" else None),
                    airway_mask=(m.airway_mask if detail == "full" else None),
                )
            )
            for values in (modes, vols, etvs, thymic_tissue_fractions, atrq_below_flags):
                values.append(None)
            computed_mask.append(False)
            valid_mask.append(False)
            continue

        mode = kde.mode
        second_ratio = kde.second_peak_ratio
        thymic_tissue_fraction = float(
            np.clip((mode - options.aadipose_hu) / (options.athymic_hu - options.aadipose_hu), 0.0, 1.0)
        )

        mode_for_etv = float(max(mode, options.aadipose_hu + options.delta_hu_margin))
        etv_frac = float(
            np.clip((mode_for_etv - options.aadipose_hu) / (options.athymic_hu - options.aadipose_hu), 0.0, 1.0)
        )
        etv = float(etv_frac * vol)

        atrq_below = bool(mode < options.aadipose_hu)
        valid = bool(second_ratio <= options.second_peak_ratio_threshold)
        if not valid:
            member_flags.append("multimodal_or_invalid")

        member_results.append(
            OkamuraMemberResult(
                member_id=m.member_id,
                valid=valid,
                trq_hu_mode=float(mode),
                trq_volume_ml=vol,
                etv_ml=etv,
                thymic_tissue_fraction=thymic_tissue_fraction,
                etv_fraction_adjusted=etv_frac,
                atrq_below_aadipose=atrq_below,
                second_peak_ratio=float(second_ratio),
                flags=tuple(member_flags),
                failure_reason=failure_reason,
                trq_mask=(m.trq_mask if detail == "full" else None),
                airway_mask=(m.airway_mask if detail == "full" else None),
            )
        )

        modes.append(float(mode))
        vols.append(vol)
        etvs.append(etv)
        thymic_tissue_fractions.append(thymic_tissue_fraction)
        atrq_below_flags.append(atrq_below)
        valid_mask.append(valid)
        computed_mask.append(True)

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    computed_indices = [i for i, v in enumerate(computed_mask) if v]
    use_indices = valid_indices if valid_indices else computed_indices

    trq_hu = float(np.median([modes[i] for i in use_indices])) if use_indices else None
    trq_vol = float(np.median([vols[i] for i in use_indices])) if use_indices else None
    etv = float(np.median([etvs[i] for i in use_indices])) if use_indices else None
    thymic_tissue_fraction = float(np.median([thymic_tissue_fractions[i] for i in use_indices])) if use_indices else None

    n_members = len(member_results)
    mean_jsd, mean_dsc, hu_var = (_ensemble_qc_values(masks, kdes) if n_members >= 2 and computed_indices else (None, None, None))
    flags: list[str] = []
    warnings_out: list[str] = []

    if options.invalidate_if_any_member_invalid and member_results and (not all(valid_mask)):
        flags.append("invalid_member")
    if any("kde_failed" in m.flags for m in member_results):
        flags.append("kde_failed")
    if len(member_results) > 0 and len(valid_indices) == 0 and len(computed_indices) > 0:
        flags.extend(["all_members_invalid", "used_invalid_members"])
        warnings_out.append("All members failed QC but had computable numeric values; summary uses invalid members and status is check.")
    elif len(valid_indices) < len(computed_indices):
        flags.append("excluded_invalid_members")
    if options.apply_qc and n_members >= 2 and computed_indices:
        # Pass when JS distance <= 0.1, DSC >= 0.7 and HU variance <= 20, as in the paper's analysis code.
        if mean_jsd is None:
            flags.append("jsd_unavailable")
        elif mean_jsd > options.js_divergence_threshold:
            flags.append("high_jsd")
        if mean_dsc is None or mean_dsc < options.pairwise_dsc_threshold:
            flags.append("low_dsc")
        if hu_var is None:
            flags.append("hu_variance_unavailable")
        elif hu_var > options.hu_variance_threshold:
            flags.append("high_hu_variance")

    if len(member_results) == 0 or not computed_indices:
        qc_status = "failed" if member_results else "not_available"
        met = None
    elif n_members < 2:
        flags.append("ensemble_qc_unavailable")
        qc_status = "check"
        met = None
    elif not options.apply_qc:
        # The ensemble criteria were not evaluated, so they cannot be reported as met.
        flags.append("ensemble_qc_not_applied")
        qc_status = "check"
        met = None
    else:
        met = len(flags) == 0
        qc_status = "ok" if met else "check"
    if "used_invalid_members" in flags or "invalid_member" in flags:
        met = False if met is True else met
        qc_status = "check" if qc_status != "failed" else qc_status

    # Informational only: does not change flags, QC status or paper_criteria_met.
    if segmentation.image is not None:
        warnings_out.extend(segmentation.image.warnings)
    prep = segmentation.preprocessing or {}
    if prep.get("inplane_is_512") is False:
        shape = prep.get("inplane_shape") or ["?", "?"]
        warnings_out.append(
            f"In-plane size {shape[0]}x{shape[1]} is not 512x512 (the size of the TRQseg-v1 training images); "
            "the image was segmented as given. Results may be less reliable."
        )

    qc = OkamuraQC(
        status=qc_status,
        paper_criteria_met=met,
        flags=tuple(dict.fromkeys(flags)),
        mean_pairwise_jsd=mean_jsd,
        mean_pairwise_dsc=mean_dsc,
        hu_variance=hu_var,
        expected_member_count=None if segmentation.segmentor is None else len(segmentation.segmentor.selected_members),
        supplied_member_count=len(member_results),
        computed_member_count=len(computed_indices),
        valid_member_count=len(valid_indices),
    )

    members_out = tuple(member_results) if detail == "full" else tuple(_as_summary_member(m) for m in member_results)

    return AnalysisResultOkamura(
        study_id=segmentation.study_id,
        method="okamura",
        meta=meta,
        segmentation=(segmentation if detail == "full" else None),
        status=qc_status,
        flags=tuple(dict.fromkeys(flags)),
        warnings=tuple(warnings_out),
        failure_reason=None if computed_indices else "No member had computable Okamura values",
        trq_hu_mode=trq_hu,
        trq_volume_ml=trq_vol,
        etv_ml=etv,
        thymic_tissue_fraction=thymic_tissue_fraction,
        atrq_below_aadipose_any=(any(atrq_below_flags[i] for i in computed_indices) if computed_indices else None),
        trq_hu_mode_members=tuple(modes),
        trq_volume_ml_members=tuple(vols),
        etv_ml_members=tuple(etvs),
        thymic_tissue_fraction_members=tuple(thymic_tissue_fractions),
        atrq_below_aadipose_members=tuple(atrq_below_flags),
        qc=qc,
        members=members_out,
    )
