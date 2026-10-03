from __future__ import annotations

"""Okamura-method quantification implementation."""

from dataclasses import dataclass
import warnings

import nibabel as nib
import numpy as np

from ..exceptions import MissingGeometryError, NumericalError
from ..inputs import validate_binary_mask, validate_mask_has_finite_ct
from ..results import AnalysisResultOkamura, DetailLevel, OkamuraMemberResult, OkamuraQC, ResultMeta
from ..segmentors import SegmentationResult
from .common import dice, voxvol_ml
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


def _local_peak_indices(y: np.ndarray) -> np.ndarray:
    """Return indices of local maxima in a 1D array."""
    if y.size < 3:
        return np.array([], dtype=int)
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    return idx.astype(int)


def _kde_mode_and_second_ratio(values: np.ndarray, grid_points: int = 2048) -> tuple[float, float]:
    """Estimate representative HU (mode) and second-peak ratio via KDE."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise NumericalError("KDE input has no finite values")

    uniq = np.unique(vals)
    if uniq.size == 1:
        return float(uniq[0]), 0.0

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if lo == hi:
        return lo, 0.0

    grid = np.linspace(lo, hi, grid_points)
    try:
        kde = gaussian_kde(vals)
        dens = np.asarray(kde(grid), dtype=float)
    except Exception as e:
        raise NumericalError(f"KDE failed: {type(e).__name__}: {e}") from e

    if dens.size == 0 or not np.isfinite(dens).any():
        raise NumericalError("KDE produced no finite density values")

    mode = float(grid[int(np.nanargmax(dens))])

    peaks = _local_peak_indices(dens)
    if peaks.size == 0:
        first = float(np.nanmax(dens))
        second = 0.0
    else:
        peak_heights = np.sort(dens[peaks])[::-1]
        first = float(peak_heights[0])
        second = float(peak_heights[1]) if peak_heights.size >= 2 else 0.0

    ratio = 0.0 if first <= 0 else second / first
    return mode, float(ratio)


def _kde_pdf_for_jsd(values: np.ndarray, hu_min: float = -300.0, hu_max: float = 300.0, grid_points: int = 1201) -> tuple[np.ndarray, np.ndarray]:
    """Estimate KDE-smoothed HU distribution for JSD computation.

    HU values are first limited to [-300, 300] as described in the paper.
    """
    grid = np.linspace(hu_min, hu_max, grid_points)

    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = np.clip(vals, hu_min, hu_max)

    if vals.size == 0:
        pdf = np.full(grid_points, 1.0 / grid_points, dtype=float)
        return grid, pdf

    uniq = np.unique(vals)
    if uniq.size == 1:
        # approximate delta distribution by narrow Gaussian on the same grid
        sigma = 1.0
        dens = np.exp(-0.5 * ((grid - float(uniq[0])) / sigma) ** 2)
    else:
        try:
            kde = gaussian_kde(vals)
            dens = np.asarray(kde(grid), dtype=float)
        except Exception:
            # fallback to histogram density if KDE fails
            hist, edges = np.histogram(vals, bins=grid_points - 1, range=(hu_min, hu_max), density=False)
            dens = np.interp(grid, (edges[:-1] + edges[1:]) / 2, hist, left=0.0, right=0.0)

    dens = np.clip(dens, 1e-12, None)
    dens = dens / dens.sum()
    return grid, dens


def _jsd_from_values(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute JSD between two KDE-smoothed HU distributions."""
    _, p = _kde_pdf_for_jsd(v1)
    _, q = _kde_pdf_for_jsd(v2)

    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


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
    values_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for m in members:
        member_flags: list[str] = []
        failure_reason = None
        try:
            ct, spacing, img = _extract_ct_and_spacing(segmentation, m)
            mask = validate_binary_mask(m.trq_mask, ct_shape=tuple(ct.shape), name=f"member {m.member_id} trq_mask")
            validate_mask_has_finite_ct(mask, ct, name=f"member {m.member_id} trq_mask")
            # Index first, then cast: avoids a float64 copy of the whole CT volume per member.
            vals = np.asarray(ct)[mask].astype(float)
            vals = vals[np.isfinite(vals)]
            mode, second_ratio = _kde_mode_and_second_ratio(vals)
            vol = voxvol_ml(mask, img, spacing_mm=spacing)
            computed = True
        except Exception as e:
            mask = np.asarray(m.trq_mask)
            vals = np.array([], dtype=float)
            mode = None
            second_ratio = None
            vol = None
            computed = False
            failure_reason = f"{type(e).__name__}: {e}"
            member_flags.append("computation_failed")
            if isinstance(e, NumericalError):
                member_flags.append("kde_failed")

        masks.append(mask)
        values_list.append(vals)

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

        thymic_tissue_fraction = float(
            np.clip((mode - options.aadipose_hu) / (options.athymic_hu - options.aadipose_hu), 0.0, 1.0)
        )

        mode_for_etv = float(max(mode, options.aadipose_hu + options.delta_hu_margin))
        etv_frac = float(
            np.clip((mode_for_etv - options.aadipose_hu) / (options.athymic_hu - options.aadipose_hu), 0.0, 1.0)
        )
        etv = float(etv_frac * vol)

        atrq_below = bool(np.isfinite(mode) and (mode < options.aadipose_hu))
        valid = bool(np.isfinite(mode) and (second_ratio <= options.second_peak_ratio_threshold))
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

    mean_jsd = None
    mean_dsc = None
    hu_var = None
    flags: list[str] = []
    warnings_out: list[str] = []

    if len(use_indices) >= 2:
        jsds: list[float] = []
        dscs: list[float] = []
        # DSC only depends on voxels inside the union of the masks; crop to its bounding box.
        box = _union_bbox([masks[i] for i in use_indices])
        for a in range(len(use_indices)):
            for b in range(a + 1, len(use_indices)):
                i = use_indices[a]
                j = use_indices[b]
                if values_list[i].size and values_list[j].size:
                    jsds.append(_jsd_from_values(values_list[i], values_list[j]))
                dscs.append(dice(masks[i][box], masks[j][box]))
        mean_jsd = float(np.mean(jsds)) if jsds else None
        mean_dsc = float(np.mean(dscs)) if dscs else None
        hu_var = float(np.var([modes[i] for i in use_indices], ddof=1)) if len(use_indices) >= 2 else None

    if options.invalidate_if_any_member_invalid and member_results and (not all(valid_mask)):
        flags.append("invalid_member")
    if any("kde_failed" in m.flags for m in member_results):
        flags.append("kde_failed")
    if len(member_results) > 0 and len(valid_indices) == 0 and len(computed_indices) > 0:
        flags.extend(["all_members_invalid", "used_invalid_members"])
        warnings_out.append("All members failed QC but had computable numeric values; summary uses invalid members and status is check.")
    elif len(valid_indices) < len(computed_indices):
        flags.append("excluded_invalid_members")
    if options.apply_qc:
        if mean_jsd is not None and mean_jsd >= options.js_divergence_threshold:
            flags.append("high_jsd")
        if mean_dsc is not None and mean_dsc <= options.pairwise_dsc_threshold:
            flags.append("low_dsc")
        if hu_var is not None and hu_var >= options.hu_variance_threshold:
            flags.append("high_hu_variance")

    if len(member_results) == 0 or not computed_indices:
        qc_status = "failed" if member_results else "not_available"
        met = None
    elif len(use_indices) < 2:
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
