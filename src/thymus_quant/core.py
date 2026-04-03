from __future__ import annotations

from itertools import combinations

import numpy as np

from .models import AnalysisResult, MemberResult, QCReport, QuantSummary, ResultMeta
from .io import load_array_and_spacing

LIB_VERSION = "0.1.0"


def _binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def _mode_hu(values: np.ndarray, hu_min: int = -300, hu_max: int = 200, bin_width: int = 1) -> float:
    bins = np.arange(hu_min, hu_max + bin_width, bin_width)
    hist, edges = np.histogram(values, bins=bins)
    if hist.sum() == 0:
        raise ValueError("mode HU cannot be computed")
    idx = int(np.argmax(hist))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = _binary(a)
    b = _binary(b)
    inter = int((a & b).sum())
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def _hist_pdf(values: np.ndarray, hu_min: int = -300, hu_max: int = 200, bin_width: int = 1) -> np.ndarray:
    bins = np.arange(hu_min, hu_max + bin_width, bin_width)
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s <= 0:
        return np.full_like(hist, 1.0 / len(hist), dtype=np.float64)
    return hist / s


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _member_metrics(
    ct_hu: np.ndarray,
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    a_thymic: float,
    a_adipose: float,
):
    bm = _binary(mask)
    values = ct_hu[bm > 0]
    if values.size == 0:
        return None
    mode = _mode_hu(values)
    voxel_ml = (spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0
    vol_ml = float(values.size * voxel_ml)
    frac = float(np.clip((mode - a_adipose) / (a_thymic - a_adipose), 0.0, 1.0))
    etv = float(frac * vol_ml)
    return {
        "trq_hu_mode": mode,
        "trq_volume_ml": vol_ml,
        "etv_ml": etv,
        "ptt_fraction": frac,
        "ptt_percent": frac * 100.0,
    }


def _qc_status(passed: bool, fail_reasons: list[str], any_valid: bool) -> str:
    if not any_valid:
        return "fail"
    if passed:
        return "pass"
    return "review" if fail_reasons else "pass"


def quantify(
    ct_image,
    trq_masks,
    *,
    study_id: str | None = None,
    segmenter: str = "provided_mask",
    protocol: str = "okamura2025",
    a_thymic: float = 80.0,
    a_adipose: float = -110.0,
    jsd_threshold: float = 0.1,
    dsc_threshold: float = 0.7,
    hu_var_threshold: float = 20.0,
) -> AnalysisResult:
    if protocol not in {"okamura2025", "okamura2025_plus_ptt"}:
        raise ValueError(f"Unsupported protocol: {protocol}")

    ct_hu, spacing, _ = load_array_and_spacing(ct_image)

    if not isinstance(trq_masks, (list, tuple)):
        trq_masks = [trq_masks]

    masks_arr = [load_array_and_spacing(m)[0] for m in trq_masks]

    members: list[MemberResult] = []
    valid_metrics: list[dict] = []
    hists: list[np.ndarray | None] = []

    for i, mask in enumerate(masks_arr):
        mm = _member_metrics(ct_hu, mask, spacing, a_thymic, a_adipose)
        if mm is None:
            members.append(MemberResult(member_id=f"member{i}", checkpoint_id=None, metrics={}, valid=False))
            hists.append(None)
            continue
        members.append(MemberResult(member_id=f"member{i}", checkpoint_id=None, metrics=mm, valid=True))
        valid_metrics.append(mm)
        hists.append(_hist_pdf(ct_hu[_binary(mask) > 0]))

    if not valid_metrics:
        summary = QuantSummary(
            trq_hu_mode=float("nan"),
            trq_volume_ml=float("nan"),
            etv_ml=float("nan"),
            ptt_fraction=float("nan"),
            ptt_percent=float("nan"),
            n_members=len(members),
            n_valid_members=0,
        )
        qc = QCReport(status="fail", passed=False, mean_pairwise_jsd=None, mean_pairwise_dsc=None, hu_variance=None, fail_reasons=["no_valid_member"])
        return AnalysisResult(summary=summary, qc=qc, members=members, meta=ResultMeta(study_id=study_id, segmenter=segmenter, protocol=protocol, library_version=LIB_VERSION))

    def med(key):
        return float(np.median([m[key] for m in valid_metrics]))

    summary = QuantSummary(
        trq_hu_mode=med("trq_hu_mode"),
        trq_volume_ml=med("trq_volume_ml"),
        etv_ml=med("etv_ml"),
        ptt_fraction=med("ptt_fraction"),
        ptt_percent=med("ptt_percent"),
        n_members=len(members),
        n_valid_members=len(valid_metrics),
    )

    pair_dsc = []
    pair_jsd = []
    if len(masks_arr) >= 2:
        for i, j in combinations(range(len(masks_arr)), 2):
            pair_dsc.append(_dice(masks_arr[i], masks_arr[j]))
            if hists[i] is not None and hists[j] is not None:
                pair_jsd.append(_jsd(hists[i], hists[j]))

    mean_dsc = float(np.mean(pair_dsc)) if pair_dsc else None
    mean_jsd = float(np.mean(pair_jsd)) if pair_jsd else None
    hu_var = float(np.var([m["trq_hu_mode"] for m in valid_metrics])) if len(valid_metrics) >= 2 else 0.0

    fail_reasons = []
    if mean_jsd is not None and not (mean_jsd < jsd_threshold):
        fail_reasons.append("jsd")
    if mean_dsc is not None and not (mean_dsc > dsc_threshold):
        fail_reasons.append("dsc")
    if hu_var is not None and not (hu_var < hu_var_threshold):
        fail_reasons.append("hu_var")

    qc_passed = len(fail_reasons) == 0
    qc = QCReport(
        status=_qc_status(qc_passed, fail_reasons, any_valid=True),
        passed=qc_passed,
        mean_pairwise_jsd=mean_jsd,
        mean_pairwise_dsc=mean_dsc,
        hu_variance=hu_var,
        fail_reasons=fail_reasons,
    )

    return AnalysisResult(
        summary=summary,
        qc=qc,
        members=members,
        meta=ResultMeta(study_id=study_id, segmenter=segmenter, protocol=protocol, library_version=LIB_VERSION),
    )


def analyze(
    ct_image,
    *,
    trq_mask,
    study_id: str | None = None,
    segmenter: str = "provided_mask",
    protocol: str = "okamura2025",
):
    if isinstance(trq_mask, (list, tuple)):
        masks = list(trq_mask)
    else:
        masks = [trq_mask]
    return quantify(ct_image, masks, study_id=study_id, segmenter=segmenter, protocol=protocol)
