from __future__ import annotations

"""Chaunzwa-method quantification implementation."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..results import (
    AnalysisResultChaunzwa,
    DetailLevel,
    GMMComponent,
    GaussianMixtureFit,
    PosteriorMaps,
    PosteriorSummary,
    ResultMeta,
)
from ..segmentors import SegmentationResult
from .common import voxvol_ml


@dataclass(slots=True)
class ChaunzwaOptions:
    """Configuration for the Chaunzwa quantification method."""

    aadipose_hu: float = -110.0
    athymic_hu: float = 80.0
    histogram_bins: int = 50
    gmm_n_components: int | None = None
    gmm_max_components: int | None = None
    model_selection: Literal["auto", "fixed", "bic", "aic"] = "auto"
    include_posterior_maps: bool = False
    include_histogram: bool = False


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _fit_gmm_1d(
    x: np.ndarray,
    k: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
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


def quantify_chaunzwa(
    segmentation: SegmentationResult,
    *,
    options: ChaunzwaOptions,
    detail: DetailLevel,
    meta: ResultMeta | None,
) -> AnalysisResultChaunzwa:
    """Run Chaunzwa quantification with a 1D Gaussian mixture model."""
    if segmentation.trq_mask is None:
        raise ValueError("segmentation.trq_mask is required")

    mask = np.asarray(segmentation.trq_mask)

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

    if options.model_selection == "fixed" and options.gmm_n_components is not None:
        k_list = [int(options.gmm_n_components)]
    elif options.gmm_n_components is not None:
        k_list = [int(options.gmm_n_components)]
    else:
        kmax = int(options.gmm_max_components or 4)
        k_list = list(range(1, max(1, kmax) + 1))

    fits = []
    for k in k_list:
        pis, mus, sigmas, resp, conv, n_iter, ll = _fit_gmm_1d(vals, k)
        bic, aic = _bic_aic(ll, vals.size, k)
        fits.append((k, pis, mus, sigmas, resp, conv, n_iter, ll, bic, aic))

    if options.model_selection in {"bic", "auto"}:
        best = min(fits, key=lambda t: t[8])
    elif options.model_selection == "aic":
        best = min(fits, key=lambda t: t[9])
    else:
        best = fits[0]

    k, pis, mus, sigmas, resp, conv, n_iter, ll, bic, aic = best

    atrq = float(np.sum(pis * mus))
    trq_vol = voxvol_ml(mask, img)
    ptt = float(np.clip((atrq - options.aadipose_hu) / (options.athymic_hu - options.aadipose_hu), 0.0, 1.0))
    etv = float(ptt * trq_vol)

    post_mass = np.sum(resp, axis=0)
    voxel_ml = 1.0 if img is None else (img.header.get_zooms()[0] * img.header.get_zooms()[1] * img.header.get_zooms()[2]) / 1000.0

    components = [
        GMMComponent(
            component_id=i,
            weight=float(pis[i]),
            mu_hu=float(mus[i]),
            sigma_hu=float(sigmas[i]),
            posterior_mass_vox=float(post_mass[i]),
            posterior_volume_ml=float(post_mass[i] * voxel_ml),
        )
        for i in range(k)
    ]

    gmm = GaussianMixtureFit(
        n_components=k,
        converged=bool(conv),
        n_iter=int(n_iter),
        model_selection=options.model_selection,
        aic=float(aic),
        bic=float(bic),
        components=tuple(components),
    )

    posterior = PosteriorSummary(
        component_posterior_mass_vox={i: float(post_mass[i]) for i in range(k)},
        component_posterior_volume_ml={i: float(post_mass[i] * voxel_ml) for i in range(k)},
        tissue_posterior_mass_vox={
            "thymic": float(np.sum(resp * (mus[None, :] >= 0), axis=None)),
            "adipose": float(np.sum(resp * (mus[None, :] < 0), axis=None)),
        },
        tissue_posterior_volume_ml={
            "thymic": float(np.sum(resp * (mus[None, :] >= 0), axis=None) * voxel_ml),
            "adipose": float(np.sum(resp * (mus[None, :] < 0), axis=None) * voxel_ml),
        },
    )

    posterior_maps = None
    if options.include_posterior_maps:
        hard = np.argmax(resp, axis=1)
        arr = np.zeros(mask.shape, dtype=np.int16)
        arr[mask > 0] = hard + 1
        posterior_maps = PosteriorMaps(component_posteriors={}, tissue_posteriors={}, hard_component_labels=arr)

    return AnalysisResultChaunzwa(
        study_id=segmentation.study_id,
        method="chaunzwa",
        meta=meta,
        segmentation=(segmentation if detail == "full" else None),
        atrq_hu=atrq,
        trq_volume_ml=trq_vol,
        etv_ml=etv,
        ptt=ptt,
        gmm=gmm,
        posterior=posterior,
        posterior_maps=posterior_maps,
    )
