from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Literal, Sequence, overload

import nibabel as nib
import numpy as np

from .methods import ChaunzwaOptions, OkamuraOptions
from .results import (
    AnalysisResultBase,
    AnalysisResultChaunzwa,
    AnalysisResultOkamura,
    BatchAnalysisResultBase,
    BatchAnalysisResultChaunzwa,
    BatchAnalysisResultOkamura,
    BatchErrorRecord,
    DetailLevel,
    GMMComponent,
    GaussianMixtureFit,
    MethodName,
    OkamuraMemberResult,
    OkamuraQC,
    OnError,
    PosteriorMaps,
    PosteriorSummary,
    ResultMeta,
)
from .segmentors import ImageInput, LoadedSegmentor, SegmentationResult, SegmentorInfo, SegmentorMember


_SEGMENTOR_CACHE: dict[tuple, LoadedSegmentor] = {}
_SEGMENTOR_CACHE_LOCK = threading.Lock()


def list_methods() -> tuple[MethodName, ...]:
    return ("okamura", "chaunzwa")


def list_segmentors() -> dict[str, SegmentorInfo]:
    available = tuple(f"fold-{i}" for i in range(5))
    return {
        "okamura_trq_v1": SegmentorInfo(
            name="okamura_trq_v1",
            repo_id="yuki-okamura-hf/TRQseg-v1",
            architecture="deeplabv3_resnet50",
            is_ensemble=True,
            available_members=available,
            selected_members=available,
        ),
        "heuristic_trq": SegmentorInfo(
            name="heuristic_trq",
            repo_id=None,
            architecture="heuristic",
            is_ensemble=False,
            available_members=("single",),
            selected_members=("single",),
        ),
    }


def _resolve_members(members: Sequence[int | str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if members is None:
        return tuple(default)
    out = []
    for m in members:
        if isinstance(m, int):
            out.append(f"fold-{m}")
        else:
            s = str(m)
            out.append(s if s.startswith("fold-") else f"fold-{s}")
    return tuple(out)


def _find_local_trqseg_v1_repo() -> str | None:
    env_path = os.environ.get("THYQ_TRQSEG_V1_LOCAL_REPO")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/mnt/w/repos/TRQseg-v1"),
            Path("/mnt/w/repos/trqseg-v1"),
        ]
    )
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)
    return None


def _segmentor_cache_key(
    info: SegmentorInfo,
    requested_members: tuple[str, ...],
    local_files_only: bool,
    device: str,
) -> tuple:
    return (
        info.name,
        info.repo_id,
        info.resolved_revision,
        requested_members,
        bool(local_files_only),
        info.cache_dir,
        device,
    )


def load_segmentor(
    segmentor: str = "okamura_trq_v1",
    *,
    revision: str | None = None,
    ensemble: bool | Literal["auto"] = "auto",
    members: Sequence[int | str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
    device: str = "auto",
) -> LoadedSegmentor:
    reg = list_segmentors()
    info = reg.get(segmentor)
    if info is None:
        info = SegmentorInfo(name=segmentor, repo_id=segmentor, architecture="unknown", is_ensemble=True)

    default_members = info.available_members or tuple(f"fold-{i}" for i in range(5))
    requested = _resolve_members(members, default_members)
    if ensemble is False and requested:
        requested = (requested[0],)

    info = SegmentorInfo(
        name=info.name,
        repo_id=info.repo_id,
        requested_revision=revision,
        resolved_revision=revision,
        architecture=info.architecture,
        is_ensemble=(len(requested) > 1),
        available_members=info.available_members,
        selected_members=requested,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )

    cache_key = _segmentor_cache_key(info, requested, local_files_only, device)
    with _SEGMENTOR_CACHE_LOCK:
        cached = _SEGMENTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    local_repo = _find_local_trqseg_v1_repo() if info.repo_id == "yuki-okamura-hf/TRQseg-v1" else None

    mems = []
    for m in requested:
        rel = f"weights/{m}/model.safetensors"
        local_path = None
        if local_repo is not None:
            candidate = Path(local_repo) / rel
            if candidate.exists():
                local_path = str(candidate)
        mems.append(
            SegmentorMember(
                member_id=m,
                relative_path=rel,
                local_path=local_path,
                revision=revision,
            )
        )

    loaded = LoadedSegmentor(
        info=info,
        members=tuple(mems),
        device=device,
        local_files_only=local_files_only,
    )

    with _SEGMENTOR_CACHE_LOCK:
        _SEGMENTOR_CACHE[cache_key] = loaded
    return loaded


def segment_trq(
    image: ImageInput,
    *,
    study_id: str | None = None,
    segmentor: str | LoadedSegmentor = "okamura_trq_v1",
    revision: str | None = None,
    device: str = "auto",
) -> SegmentationResult:
    seg = load_segmentor(segmentor, revision=revision, device=device) if isinstance(segmentor, str) else segmentor
    return seg.segment_trq(image, study_id=study_id)


def _voxvol_ml(mask: np.ndarray, img: nib.spatialimages.SpatialImage) -> float:
    zoom = img.header.get_zooms()[:3]
    return float((mask > 0).sum() * (zoom[0] * zoom[1] * zoom[2]) / 1000.0)


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


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    den = a.sum() + b.sum()
    return 1.0 if den == 0 else float(2.0 * (a & b).sum() / den)


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


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _fit_gmm_1d(x: np.ndarray, k: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
    x = x.astype(float)
    n = x.size
    qs = np.linspace(0.1, 0.9, k)
    mus = np.quantile(x, qs)
    sigmas = np.full(k, np.std(x) if np.std(x) > 1e-6 else 1.0)
    pis = np.full(k, 1.0 / k)

    prev_ll = -np.inf
    resp = np.zeros((n, k), dtype=float)
    converged = False
    for it in range(1, max_iter + 1):
        for j in range(k):
            resp[:, j] = pis[j] * _gaussian_pdf(x, mus[j], sigmas[j])
        denom = np.sum(resp, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        resp /= denom

        nk = np.sum(resp, axis=0)
        pis = nk / n
        mus = np.sum(resp * x[:, None], axis=0) / np.clip(nk, 1e-12, None)
        var = np.sum(resp * (x[:, None] - mus[None, :]) ** 2, axis=0) / np.clip(nk, 1e-12, None)
        sigmas = np.sqrt(np.clip(var, 1e-6, None))

        mix = np.zeros(n)
        for j in range(k):
            mix += pis[j] * _gaussian_pdf(x, mus[j], sigmas[j])
        ll = float(np.sum(np.log(np.clip(mix, 1e-12, None))))
        if abs(ll - prev_ll) < 1e-5:
            converged = True
            prev_ll = ll
            break
        prev_ll = ll
    return pis, mus, sigmas, resp, converged, it, prev_ll


def _bic_aic(loglik: float, n: int, k: int) -> tuple[float, float]:
    p = 3 * k - 1
    bic = -2 * loglik + p * math.log(max(n, 1))
    aic = -2 * loglik + 2 * p
    return bic, aic


def _meta(segmentation: SegmentationResult, method: MethodName, detail: DetailLevel) -> ResultMeta:
    sinfo = segmentation.segmentor
    return ResultMeta(
        study_id=segmentation.study_id,
        method=method,
        detail=detail,
        input_source=segmentation.study_id,
        segmentor_name=None if sinfo is None else sinfo.name,
        segmentor_repo_id=None if sinfo is None else sinfo.repo_id,
        segmentor_revision=None if sinfo is None else sinfo.resolved_revision,
        segmentor_members=() if sinfo is None else tuple(sinfo.selected_members),
        library_version="0.1.0a0",
    )


@overload
def quantify(segmentation: SegmentationResult, *, method: Literal["okamura"], options: OkamuraOptions | None = None, detail: DetailLevel = "summary") -> AnalysisResultOkamura: ...


@overload
def quantify(segmentation: SegmentationResult, *, method: Literal["chaunzwa"], options: ChaunzwaOptions | None = None, detail: DetailLevel = "summary") -> AnalysisResultChaunzwa: ...


def quantify(segmentation: SegmentationResult, *, method: MethodName, options: OkamuraOptions | ChaunzwaOptions | None = None, detail: DetailLevel = "summary") -> AnalysisResultBase:
    if method == "okamura":
        opt = options if isinstance(options, OkamuraOptions) else OkamuraOptions()

        members = segmentation.members if segmentation.members else []
        if not members and segmentation.trq_mask is not None:
            from .segmentors import SegmentationMember
            members = [SegmentationMember(member_id="single", trq_mask=segmentation.trq_mask)]

        # no original CT on SegmentationResult by schema; assume raw_output may carry CT image
        # fallback: use mask values as proxy if CT unavailable
        member_results = []
        modes, vols, etvs, valid_mask = [], [], [], []
        values_list = []
        masks = []
        for m in members:
            ct = None
            img = None
            if isinstance(m.raw_output, dict):
                ct = m.raw_output.get("ct_array")
                img = m.raw_output.get("ct_image")
            mask = np.asarray(m.trq_mask)
            masks.append(mask)
            if ct is None:
                ct = np.where(mask > 0, -30.0, -110.0)
            vals = np.asarray(ct)[mask > 0]
            values_list.append(vals)
            mode, second_ratio = _mode_and_second_ratio(vals)
            vol = float(mask.sum()) if img is None else _voxvol_ml(mask, img)
            frac = (mode - opt.aadipose_hu) / (opt.athymic_hu - opt.aadipose_hu)
            frac = float(np.clip(frac, 0.0, 1.0))
            etv = frac * vol
            valid = np.isfinite(mode) and (second_ratio <= opt.second_peak_ratio_threshold)
            member_results.append(OkamuraMemberResult(member_id=m.member_id, valid=bool(valid), trq_hu_mode=float(mode), trq_volume_ml=float(vol), etv_ml=float(etv), second_peak_ratio=float(second_ratio), trq_mask=(m.trq_mask if detail == "full" else None), airway_mask=(m.airway_mask if detail == "full" else None)))
            modes.append(float(mode)); vols.append(float(vol)); etvs.append(float(etv)); valid_mask.append(bool(valid))

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
            jsds, dscs = [], []
            for i in range(len(member_results)):
                for j in range(i + 1, len(member_results)):
                    if values_list[i].size and values_list[j].size:
                        jsds.append(_jsd_from_values(values_list[i], values_list[j]))
                    dscs.append(_dice(masks[i], masks[j]))
            mean_jsd = float(np.mean(jsds)) if jsds else None
            mean_dsc = float(np.mean(dscs)) if dscs else None
            hu_var = float(np.var([modes[i] for i in use_indices])) if len(use_indices) >= 2 else 0.0

        if opt.invalidate_if_any_member_invalid and (not all(valid_mask)):
            flags.append("invalid_member")
        if opt.apply_qc:
            if mean_jsd is not None and mean_jsd >= opt.js_divergence_threshold:
                flags.append("high_jsd")
            if mean_dsc is not None and mean_dsc <= opt.pairwise_dsc_threshold:
                flags.append("low_dsc")
            if hu_var is not None and hu_var >= opt.hu_variance_threshold:
                flags.append("high_hu_variance")

        if len(member_results) == 0:
            status = "not_available"
            met = None
        else:
            met = len(flags) == 0
            status = "ok" if met else "check"

        qc = OkamuraQC(status=status, paper_criteria_met=met, flags=tuple(flags), mean_pairwise_jsd=mean_jsd, mean_pairwise_dsc=mean_dsc, hu_variance=hu_var, valid_member_count=len(valid_indices))
        return AnalysisResultOkamura(
            study_id=segmentation.study_id,
            method="okamura",
            meta=_meta(segmentation, "okamura", detail),
            segmentation=(segmentation if detail == "full" else None),
            trq_hu_mode=trq_hu,
            trq_volume_ml=trq_vol,
            etv_ml=etv,
            qc=qc,
            members=tuple(member_results) if detail == "full" else tuple([OkamuraMemberResult(member_id=m.member_id, valid=m.valid, trq_hu_mode=m.trq_hu_mode, trq_volume_ml=m.trq_volume_ml, etv_ml=m.etv_ml, second_peak_ratio=m.second_peak_ratio) for m in member_results]),
        )

    # chaunzwa
    opt = options if isinstance(options, ChaunzwaOptions) else ChaunzwaOptions()
    if segmentation.trq_mask is None:
        raise ValueError("segmentation.trq_mask is required")
    mask = np.asarray(segmentation.trq_mask)
    # obtain ct from first member raw_output if present, else synthetic default
    ct = None
    img = None
    if segmentation.members and isinstance(segmentation.members[0].raw_output, dict):
        ct = segmentation.members[0].raw_output.get("ct_array")
        img = segmentation.members[0].raw_output.get("ct_image")
    if ct is None:
        ct = np.where(mask > 0, -30.0, -110.0)

    vals = np.asarray(ct)[mask > 0]
    if vals.size == 0:
        raise ValueError("TRQ mask is empty")

    if opt.model_selection == "fixed" and opt.gmm_n_components is not None:
        k_list = [int(opt.gmm_n_components)]
    elif opt.gmm_n_components is not None:
        k_list = [int(opt.gmm_n_components)]
    else:
        kmax = int(opt.gmm_max_components or 4)
        k_list = list(range(1, max(1, kmax) + 1))

    fits = []
    for k in k_list:
        pis, mus, sigmas, resp, conv, n_iter, ll = _fit_gmm_1d(vals, k)
        bic, aic = _bic_aic(ll, vals.size, k)
        fits.append((k, pis, mus, sigmas, resp, conv, n_iter, ll, bic, aic))

    if opt.model_selection in {"bic", "auto"}:
        best = min(fits, key=lambda t: t[8])
    elif opt.model_selection == "aic":
        best = min(fits, key=lambda t: t[9])
    else:
        best = fits[0]

    k, pis, mus, sigmas, resp, conv, n_iter, ll, bic, aic = best
    atrq = float(np.sum(pis * mus))
    trq_vol = float(mask.sum()) if img is None else _voxvol_ml(mask, img)
    ptt = float(np.clip((atrq - opt.aadipose_hu) / (opt.athymic_hu - opt.aadipose_hu), 0.0, 1.0))
    etv = float(ptt * trq_vol)

    comps = []
    post_mass = np.sum(resp, axis=0)
    voxel_ml = (1.0 if img is None else (img.header.get_zooms()[0] * img.header.get_zooms()[1] * img.header.get_zooms()[2]) / 1000.0)
    for i in range(k):
        comps.append(GMMComponent(component_id=i, weight=float(pis[i]), mu_hu=float(mus[i]), sigma_hu=float(sigmas[i]), posterior_mass_vox=float(post_mass[i]), posterior_volume_ml=float(post_mass[i] * voxel_ml)))

    gmm = GaussianMixtureFit(n_components=k, converged=bool(conv), n_iter=int(n_iter), model_selection=opt.model_selection, aic=float(aic), bic=float(bic), components=tuple(comps))

    posterior = PosteriorSummary(
        component_posterior_mass_vox={i: float(post_mass[i]) for i in range(k)},
        component_posterior_volume_ml={i: float(post_mass[i] * voxel_ml) for i in range(k)},
        tissue_posterior_mass_vox={"thymic": float(np.sum(resp * (mus[None, :] >= 0), axis=None)), "adipose": float(np.sum(resp * (mus[None, :] < 0), axis=None))},
        tissue_posterior_volume_ml={"thymic": float(np.sum(resp * (mus[None, :] >= 0), axis=None) * voxel_ml), "adipose": float(np.sum(resp * (mus[None, :] < 0), axis=None) * voxel_ml)},
    )

    posterior_maps = None
    if opt.include_posterior_maps:
        hard = np.argmax(resp, axis=1)
        arr = np.zeros(mask.shape, dtype=np.int16)
        arr[mask > 0] = hard + 1
        posterior_maps = PosteriorMaps(component_posteriors={}, tissue_posteriors={}, hard_component_labels=arr)

    return AnalysisResultChaunzwa(
        study_id=segmentation.study_id,
        method="chaunzwa",
        meta=_meta(segmentation, "chaunzwa", detail),
        segmentation=(segmentation if detail == "full" else None),
        atrq_hu=atrq,
        trq_volume_ml=trq_vol,
        etv_ml=etv,
        ptt=ptt,
        gmm=gmm,
        posterior=posterior,
        posterior_maps=posterior_maps,
    )


@overload
def analyze(image: ImageInput, *, method: Literal["okamura"], study_id: str | None = None, segmentor: str | LoadedSegmentor = "okamura_trq_v1", revision: str | None = None, options: OkamuraOptions | None = None, detail: DetailLevel = "summary", device: str = "auto") -> AnalysisResultOkamura: ...


@overload
def analyze(image: ImageInput, *, method: Literal["chaunzwa"], study_id: str | None = None, segmentor: str | LoadedSegmentor = "okamura_trq_v1", revision: str | None = None, options: ChaunzwaOptions | None = None, detail: DetailLevel = "summary", device: str = "auto") -> AnalysisResultChaunzwa: ...


def analyze(image: ImageInput, *, method: MethodName, study_id: str | None = None, segmentor: str | LoadedSegmentor = "okamura_trq_v1", revision: str | None = None, options: OkamuraOptions | ChaunzwaOptions | None = None, detail: DetailLevel = "summary", device: str = "auto") -> AnalysisResultBase:
    seg = segment_trq(image, study_id=study_id, segmentor=segmentor, revision=revision, device=device)

    # attach CT to member raw_output for quantification routines
    img = image if isinstance(image, nib.spatialimages.SpatialImage) else nib.load(str(image))
    ct = np.asarray(img.get_fdata(), dtype=float)
    for m in seg.members:
        base = m.raw_output if isinstance(m.raw_output, dict) else {}
        base["ct_array"] = ct
        base["ct_image"] = img
        m.raw_output = base

    return quantify(seg, method=method, options=options, detail=detail)


@overload
def analyze_many(images: Sequence[ImageInput], *, method: Literal["okamura"], study_ids: Sequence[str | None] | None = None, segmentor: str | LoadedSegmentor = "okamura_trq_v1", revision: str | None = None, options: OkamuraOptions | None = None, detail: DetailLevel = "summary", on_error: OnError = "record", device: str = "auto") -> BatchAnalysisResultOkamura: ...


@overload
def analyze_many(images: Sequence[ImageInput], *, method: Literal["chaunzwa"], study_ids: Sequence[str | None] | None = None, segmentor: str | LoadedSegmentor = "okamura_trq_v1", revision: str | None = None, options: ChaunzwaOptions | None = None, detail: DetailLevel = "summary", on_error: OnError = "record", device: str = "auto") -> BatchAnalysisResultChaunzwa: ...


def analyze_many(images: Sequence[ImageInput], *, method: MethodName, study_ids: Sequence[str | None] | None = None, segmentor: str | LoadedSegmentor = "okamura_trq_v1", revision: str | None = None, options: OkamuraOptions | ChaunzwaOptions | None = None, detail: DetailLevel = "summary", on_error: OnError = "record", device: str = "auto") -> BatchAnalysisResultBase:
    if study_ids is not None and len(study_ids) != len(images):
        raise ValueError("study_ids must match images length")

    seg_loaded = load_segmentor(segmentor, revision=revision, device=device) if isinstance(segmentor, str) else segmentor

    results = []
    errors: list[BatchErrorRecord] = []
    for i, image in enumerate(images):
        sid = None if study_ids is None else study_ids[i]
        try:
            r = analyze(image, method=method, study_id=sid, segmentor=seg_loaded, options=options, detail=detail, device=device)
            results.append(r)
        except Exception as e:
            if on_error == "raise":
                raise
            if on_error == "record":
                errors.append(BatchErrorRecord(input_source=str(image), study_id=sid, error_type=type(e).__name__, message=str(e)))
            # skip -> do nothing

    if method == "okamura":
        return BatchAnalysisResultOkamura(method="okamura", results=tuple(results), errors=tuple(errors))
    return BatchAnalysisResultChaunzwa(method="chaunzwa", results=tuple(results), errors=tuple(errors))
