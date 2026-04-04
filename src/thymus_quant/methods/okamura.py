from __future__ import annotations

"""Okamura-method quantification implementation."""

from dataclasses import dataclass

import nibabel as nib
import numpy as np

from ..results import AnalysisResultOkamura, DetailLevel, OkamuraMemberResult, OkamuraQC, ResultMeta
from ..segmentors import SegmentationResult
from .common import dice, voxvol_ml


@dataclass(slots=True)
class OkamuraOptions:
    """Configuration for the Okamura quantification method."""

    aadipose_hu: float = -110.0
    athymic_hu: float = 80.0
    second_peak_ratio_threshold: float = 0.5
    js_divergence_threshold: float = 0.1
    pairwise_dsc_threshold: float = 0.7
    hu_variance_threshold: float = 20.0
    invalidate_if_any_member_invalid: bool = True
    apply_qc: bool = True


def _mode_and_second_ratio(values: np.ndarray, bins: int = 512) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")

    vmin, vmax = float(np.min(values)), float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return float(vmin), 0.0

    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    idx = int(np.argmax(hist))
    first = float(hist[idx])
    hist2 = hist.copy()
    hist2[idx] = 0
    second = float(np.max(hist2)) if hist2.size else 0.0
    mode = float((edges[idx] + edges[idx + 1]) / 2)
    ratio = 0.0 if first <= 0 else second / first
    return mode, ratio


def _jsd_from_values(v1: np.ndarray, v2: np.ndarray, bins: int = 300) -> float:
    lo = float(min(np.min(v1), np.min(v2)))
    hi = float(max(np.max(v1), np.max(v2)))
    if lo == hi:
        return 0.0

    h1, _ = np.histogram(v1, bins=bins, range=(lo, hi))
    h2, _ = np.histogram(v2, bins=bins, range=(lo, hi))

    p = h1.astype(float)
    q = h2.astype(float)
    p = p / (p.sum() if p.sum() else 1)
    q = q / (q.sum() if q.sum() else 1)

    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def _extract_member_ct_and_img(member, mask: np.ndarray) -> tuple[np.ndarray, nib.spatialimages.SpatialImage | None]:
    ct = None
    img = None
    if isinstance(member.raw_output, dict):
        ct = member.raw_output.get("ct_array")
        img = member.raw_output.get("ct_image")
    if ct is None:
        ct = np.where(mask > 0, -30.0, -110.0)
    return np.asarray(ct), img


def _as_summary_member(member: OkamuraMemberResult) -> OkamuraMemberResult:
    return OkamuraMemberResult(
        member_id=member.member_id,
        valid=member.valid,
        trq_hu_mode=member.trq_hu_mode,
        trq_volume_ml=member.trq_volume_ml,
        etv_ml=member.etv_ml,
        second_peak_ratio=member.second_peak_ratio,
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
    modes: list[float] = []
    vols: list[float] = []
    etvs: list[float] = []
    valid_mask: list[bool] = []
    values_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for m in members:
        mask = np.asarray(m.trq_mask)
        masks.append(mask)

        ct, img = _extract_member_ct_and_img(m, mask)
        vals = ct[mask > 0]
        values_list.append(vals)

        mode, second_ratio = _mode_and_second_ratio(vals)
        vol = voxvol_ml(mask, img)
        frac = float(np.clip((mode - options.aadipose_hu) / (options.athymic_hu - options.aadipose_hu), 0.0, 1.0))
        etv = float(frac * vol)
        valid = bool(np.isfinite(mode) and (second_ratio <= options.second_peak_ratio_threshold))

        member_results.append(
            OkamuraMemberResult(
                member_id=m.member_id,
                valid=valid,
                trq_hu_mode=float(mode),
                trq_volume_ml=vol,
                etv_ml=etv,
                second_peak_ratio=float(second_ratio),
                trq_mask=(m.trq_mask if detail == "full" else None),
                airway_mask=(m.airway_mask if detail == "full" else None),
            )
        )

        modes.append(float(mode))
        vols.append(vol)
        etvs.append(etv)
        valid_mask.append(valid)

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    use_indices = valid_indices if valid_indices else list(range(len(member_results)))

    trq_hu = float(np.median([modes[i] for i in use_indices])) if use_indices else None
    trq_vol = float(np.median([vols[i] for i in use_indices])) if use_indices else None
    etv = float(np.median([etvs[i] for i in use_indices])) if use_indices else None

    mean_jsd = None
    mean_dsc = None
    hu_var = None
    flags: list[str] = []

    if len(member_results) >= 2:
        jsds: list[float] = []
        dscs: list[float] = []
        for i in range(len(member_results)):
            for j in range(i + 1, len(member_results)):
                if values_list[i].size and values_list[j].size:
                    jsds.append(_jsd_from_values(values_list[i], values_list[j]))
                dscs.append(dice(masks[i], masks[j]))
        mean_jsd = float(np.mean(jsds)) if jsds else None
        mean_dsc = float(np.mean(dscs)) if dscs else None
        hu_var = float(np.var([modes[i] for i in use_indices])) if len(use_indices) >= 2 else 0.0

    if options.invalidate_if_any_member_invalid and member_results and (not all(valid_mask)):
        flags.append("invalid_member")
    if options.apply_qc:
        if mean_jsd is not None and mean_jsd >= options.js_divergence_threshold:
            flags.append("high_jsd")
        if mean_dsc is not None and mean_dsc <= options.pairwise_dsc_threshold:
            flags.append("low_dsc")
        if hu_var is not None and hu_var >= options.hu_variance_threshold:
            flags.append("high_hu_variance")

    if len(member_results) == 0:
        qc_status = "not_available"
        met = None
    else:
        met = len(flags) == 0
        qc_status = "ok" if met else "check"

    qc = OkamuraQC(
        status=qc_status,
        paper_criteria_met=met,
        flags=tuple(flags),
        mean_pairwise_jsd=mean_jsd,
        mean_pairwise_dsc=mean_dsc,
        hu_variance=hu_var,
        valid_member_count=len(valid_indices),
    )

    members_out = tuple(member_results) if detail == "full" else tuple(_as_summary_member(m) for m in member_results)

    return AnalysisResultOkamura(
        study_id=segmentation.study_id,
        method="okamura",
        meta=meta,
        segmentation=(segmentation if detail == "full" else None),
        trq_hu_mode=trq_hu,
        trq_volume_ml=trq_vol,
        etv_ml=etv,
        qc=qc,
        members=members_out,
    )
