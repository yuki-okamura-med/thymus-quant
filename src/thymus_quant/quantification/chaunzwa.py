from __future__ import annotations

"""Chaunzwa-method quantification implementation.

Paper-faithful primary definition:
- fit GMM on histogramized TRQ HU values
- evaluate voxel-wise posterior responsibilities
- define pTT as mean posterior of non-adipose components (percent, 0-100)
"""

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
    model_selection: Literal["auto", "fixed", "bic", "aic"] = "bic"
    adipose_policy: Literal["lowest_mean_only", "lowest_plus_subfat"] = "lowest_mean_only"
    ptt_definition: Literal["posterior_nonadipose", "atrq_linear"] = "posterior_nonadipose"
    bayesian_delta_hu: float = 1.0
    include_posterior_maps: bool = False
    include_histogram: bool = False


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _weighted_quantiles(x: np.ndarray, w: np.ndarray, qs: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws)
    if cdf[-1] <= 0:
        return np.full_like(qs, float(np.mean(x)), dtype=float)
    cdf = cdf / cdf[-1]
    return np.interp(qs, cdf, xs)


def _fit_gmm_histogram_1d(
    values: np.ndarray,
    *,
    k: int,
    bins: int,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
    """Fit 1D GMM using weighted EM on histogram bin centers.

    Parameters
    ----------
    values:
        Finite HU values inside TRQ mask.
    k:
        Number of components.
    bins:
        Histogram bins over full dynamic range.
    """
    vals = np.asarray(values, dtype=float)
    counts, edges = np.histogram(vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    use = counts > 0
    x = centers[use]
    w = counts[use].astype(float)
    n_vox = float(vals.size)

    if x.size == 0:
        raise ValueError("No histogram bins with positive counts")

    qs = np.linspace(0.1, 0.9, k)
    mus = _weighted_quantiles(x, w, qs)

    w_sum = np.sum(w)
    w_mean = float(np.sum(w * x) / max(w_sum, 1.0))
    w_var = float(np.sum(w * (x - w_mean) ** 2) / max(w_sum, 1.0))
    init_sigma = math.sqrt(max(w_var, 1e-6))
    sigmas = np.full(k, init_sigma, dtype=float)
    pis = np.full(k, 1.0 / k, dtype=float)

    prev_ll = -np.inf
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        # E-step on histogram centers
        prob = np.zeros((x.size, k), dtype=float)
        for j in range(k):
            prob[:, j] = pis[j] * _gaussian_pdf(x, mus[j], sigmas[j])

        denom = np.sum(prob, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        r = prob / denom

        # M-step with histogram counts as weights
        n_i = np.sum(w[:, None] * r, axis=0)
        pis = n_i / max(n_vox, 1.0)

        mus = np.sum(w[:, None] * r * x[:, None], axis=0) / np.clip(n_i, 1e-12, None)
        var = np.sum(w[:, None] * r * (x[:, None] - mus[None, :]) ** 2, axis=0) / np.clip(n_i, 1e-12, None)
        sigmas = np.sqrt(np.clip(var, 1e-6, None))

        ll = float(np.sum(w * np.log(np.clip(np.sum(prob, axis=1), 1e-12, None))))
        if abs(ll - prev_ll) < tol:
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    return pis, mus, sigmas, converged, it, float(prev_ll)


def _bic_aic(loglik: float, n: int, k: int) -> tuple[float, float]:
    p = 3 * k - 1
    bic = -2 * loglik + p * math.log(max(n, 1))
    aic = -2 * loglik + 2 * p
    return bic, aic


def _voxel_responsibilities(values: np.ndarray, pis: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    k = pis.size
    prob = np.zeros((vals.size, k), dtype=float)
    for j in range(k):
        prob[:, j] = pis[j] * _gaussian_pdf(vals, mus[j], sigmas[j])
    denom = np.sum(prob, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return prob / denom


def _adipose_component_ids(
    mus: np.ndarray,
    *,
    policy: Literal["lowest_mean_only", "lowest_plus_subfat"],
    aadipose_hu: float,
) -> set[int]:
    """Choose adipose component set F from fitted component means.

    Default policy follows the paper-oriented conservative choice:
    `lowest_mean_only`.

    To avoid forcing adipose when no fat-like component exists, we require
    the lowest-mean component to be <= `aadipose_hu`.
    """
    order = np.argsort(mus)
    if order.size == 0:
        return set()

    lowest = int(order[0])
    if float(mus[lowest]) > float(aadipose_hu):
        # no adipose-like mode present
        return set()

    ids = {lowest}

    if policy == "lowest_plus_subfat":
        for i in order[1:]:
            if float(mus[i]) <= float(aadipose_hu):
                ids.add(int(i))

    return ids


def quantify_chaunzwa(
    segmentation: SegmentationResult,
    *,
    options: ChaunzwaOptions,
    detail: DetailLevel,
    meta: ResultMeta | None,
) -> AnalysisResultChaunzwa:
    """Run Chaunzwa quantification using posterior-based pTT definition."""
    if segmentation.trq_mask is None:
        raise ValueError("segmentation.trq_mask is required")

    mask = np.asarray(segmentation.trq_mask)

    ct = None
    img = None
    if segmentation.members and isinstance(segmentation.members[0].raw_output, dict):
        ct = segmentation.members[0].raw_output.get("ct_array")
        img = segmentation.members[0].raw_output.get("ct_image")
    if ct is None:
        raise ValueError("CT array is missing for Chaunzwa quantification")

    vals = np.asarray(ct, dtype=float)[mask > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("TRQ mask has no finite HU voxels")

    # model candidates
    if options.model_selection == "fixed" and options.gmm_n_components is not None:
        k_list = [int(options.gmm_n_components)]
    elif options.gmm_n_components is not None:
        k_list = [int(options.gmm_n_components)]
    else:
        kmax = int(options.gmm_max_components or 4)
        k_list = list(range(1, max(1, kmax) + 1))

    fits = []
    for k in k_list:
        pis, mus, sigmas, conv, n_iter, ll = _fit_gmm_histogram_1d(
            vals,
            k=k,
            bins=int(options.histogram_bins),
        )
        bic, aic = _bic_aic(ll, vals.size, k)
        fits.append((k, pis, mus, sigmas, conv, n_iter, ll, bic, aic))

    model_sel = "bic" if options.model_selection == "auto" else options.model_selection
    if model_sel == "bic":
        best = min(fits, key=lambda t: t[7])
    elif model_sel == "aic":
        best = min(fits, key=lambda t: t[8])
    else:  # fixed
        best = fits[0]

    k, pis, mus, sigmas, conv, n_iter, ll, bic, aic = best

    # auxiliary ATRQ metric
    atrq = float(np.sum(pis * mus))
    atrq_adjusted = float(max(atrq, options.aadipose_hu + options.bayesian_delta_hu))

    # voxel-wise posterior responsibilities
    resp = _voxel_responsibilities(vals, pis, mus, sigmas)

    adipose_ids = _adipose_component_ids(
        mus,
        policy=options.adipose_policy,
        aadipose_hu=options.aadipose_hu,
    )
    nonadipose_ids = set(range(k)) - adipose_ids

    gamma_adipose = np.sum(resp[:, sorted(adipose_ids)], axis=1) if adipose_ids else np.zeros(vals.size, dtype=float)
    gamma_nonadipose = (
        np.sum(resp[:, sorted(nonadipose_ids)], axis=1) if nonadipose_ids else np.zeros(vals.size, dtype=float)
    )

    # primary pTT definition (percent)
    ptt_posterior_percent = float(np.mean(gamma_nonadipose) * 100.0)
    ptt_posterior_percent = float(np.clip(ptt_posterior_percent, 0.0, 100.0))

    # legacy pTT definition for backward compatibility
    ptt_linear_percent = float(
        np.clip((atrq - options.aadipose_hu) / (options.athymic_hu - options.aadipose_hu), 0.0, 1.0) * 100.0
    )

    if options.ptt_definition == "atrq_linear":
        ptt = ptt_linear_percent
    else:
        ptt = ptt_posterior_percent

    trq_vol = voxvol_ml(mask, img)
    etv = float(trq_vol * (ptt / 100.0))

    post_mass = np.sum(resp, axis=0)
    voxel_ml = (
        float("nan")
        if img is None
        else (img.header.get_zooms()[0] * img.header.get_zooms()[1] * img.header.get_zooms()[2]) / 1000.0
    )

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
        model_selection=model_sel,
        aic=float(aic),
        bic=float(bic),
        components=tuple(components),
    )

    adipose_mass = float(np.sum(gamma_adipose))
    nonadipose_mass = float(np.sum(gamma_nonadipose))

    posterior = PosteriorSummary(
        component_posterior_mass_vox={i: float(post_mass[i]) for i in range(k)},
        component_posterior_volume_ml={i: float(post_mass[i] * voxel_ml) for i in range(k)},
        tissue_posterior_mass_vox={
            "adipose": adipose_mass,
            "thymic": nonadipose_mass,
        },
        tissue_posterior_volume_ml={
            "adipose": float(adipose_mass * voxel_ml),
            "thymic": float(nonadipose_mass * voxel_ml),
        },
    )

    posterior_maps = None
    if options.include_posterior_maps:
        # map each finite TRQ voxel back to volume space
        trq_idx = np.argwhere(mask > 0)
        ct_vals = np.asarray(ct, dtype=float)[mask > 0]
        finite_sel = np.isfinite(ct_vals)
        finite_idx = trq_idx[finite_sel]

        component_maps: dict[int, np.ndarray] = {}
        for i in range(k):
            arr = np.zeros(mask.shape, dtype=np.float32)
            arr[tuple(finite_idx.T)] = resp[:, i].astype(np.float32)
            component_maps[i] = arr

        adipose_map = np.zeros(mask.shape, dtype=np.float32)
        if adipose_ids:
            adipose_map[tuple(finite_idx.T)] = gamma_adipose.astype(np.float32)

        thymic_map = np.zeros(mask.shape, dtype=np.float32)
        if nonadipose_ids:
            thymic_map[tuple(finite_idx.T)] = gamma_nonadipose.astype(np.float32)

        hard = np.zeros(mask.shape, dtype=np.int16)
        hard_labels = np.argmax(resp, axis=1) + 1
        hard[tuple(finite_idx.T)] = hard_labels.astype(np.int16)

        posterior_maps = PosteriorMaps(
            component_posteriors=component_maps,
            tissue_posteriors={"adipose": adipose_map, "thymic": thymic_map},
            hard_component_labels=hard,
        )

    return AnalysisResultChaunzwa(
        study_id=segmentation.study_id,
        method="chaunzwa",
        meta=meta,
        segmentation=(segmentation if detail == "full" else None),
        atrq_hu=atrq,
        atrq_hu_adjusted=atrq_adjusted,
        trq_volume_ml=trq_vol,
        etv_ml=etv,
        ptt=ptt,
        ptt_definition=options.ptt_definition,
        ptt_posterior_primary=ptt_posterior_percent,
        ptt_linear_legacy=ptt_linear_percent,
        adipose_component_ids=tuple(sorted(adipose_ids)),
        nonadipose_component_ids=tuple(sorted(nonadipose_ids)),
        gmm=gmm,
        posterior=posterior,
        posterior_maps=posterior_maps,
    )
