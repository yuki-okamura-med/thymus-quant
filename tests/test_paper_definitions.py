"""The Okamura quantities follow the paper's analysis code.

The reference below re-implements that code directly with SciPy (gaussian_kde on every TRQ
voxel, evaluated on -300..300 HU; peaks from argrelextrema(d, np.greater); JS distance from
scipy.spatial.distance.jensenshannon; DSC over all members; unbiased variance of the 1 HU modes).
The only intended difference is A_TRQ, taken on the 0.1 HU grid (the paper used 1 HU).
"""
import numpy as np
import pytest
from scipy.signal import argrelextrema
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from scipy.stats import gaussian_kde

import thymus_quant as tq
from thymus_quant.inputs import make_image_context
from thymus_quant.quantification import okamura as okm

G1 = np.linspace(-300, 300, 601)
G01 = np.linspace(-300, 300, 6001)


def _paper_member(values):
    kde = gaussian_kde(values)
    d1, d01 = kde(G1), kde(G01)
    p1 = np.ravel(argrelextrema(d1, np.greater))
    p01 = np.ravel(argrelextrema(d01, np.greater))
    h = np.sort(d1[p1])[::-1]
    return {
        "mode_1hu": p1[np.argmax(d1[p1])] - 300.0,
        "mode_01hu": p01[np.argmax(d01[p01])] / 10 - 300.0,
        "ratio": h[1] / h[0] if h.size > 1 else 0.0,
        "density": d1,
    }


def _ct():
    rng = np.random.default_rng(7)
    ct = rng.normal(-1000.0, 5.0, size=(50, 50, 16)).astype(np.float32)
    ct[8:42, 8:42, 2:14] = rng.normal(-60.0, 40.0, size=(34, 34, 12))
    ct[8:42, 30:42, 2:14] = rng.normal(-105.0, 35.0, size=(34, 12, 12))
    return ct


def _mask(shape, x0, x1, y0, y1):
    m = np.zeros(shape, dtype=np.uint8)
    m[x0:x1, y0:y1, 3:13] = 1
    return m


def _members(ct, masks):
    return tq.SegmentationResult(
        study_id="case",
        image=make_image_context(ct_hu=ct, spacing_mm=(0.7, 0.7, 2.0)),
        members=tuple(tq.SegmentationMember(f"fold-{i}", m) for i, m in enumerate(masks)),
    )


def test_member_values_match_the_paper_code():
    ct = _ct()
    masks = [_mask(ct.shape, 10, 40, 10, 30 + k) for k in range(5)]
    res = tq.quantify(_members(ct, masks), method="okamura", detail="full")
    for k, m in enumerate(masks):
        ref = _paper_member(ct[m > 0].astype(float))
        got = okm._kde_mode_and_second_ratio(ct[m > 0].astype(float))
        assert got.mode == pytest.approx(ref["mode_01hu"], abs=1e-9)
        assert got.mode_qc == pytest.approx(ref["mode_1hu"], abs=1e-9)
        assert got.second_peak_ratio == pytest.approx(ref["ratio"], rel=1e-7, abs=1e-12)
        assert res.members[k].trq_hu_mode == pytest.approx(ref["mode_01hu"], abs=1e-9)


def test_ensemble_qc_matches_the_paper_code():
    ct = _ct()
    masks = [_mask(ct.shape, 10 + k, 40, 10, 32 - k) for k in range(5)]
    res = tq.quantify(_members(ct, masks), method="okamura")
    refs = [_paper_member(ct[m > 0].astype(float)) for m in masks]
    js, dsc = [], []
    for i in range(5):
        for j in range(i + 1, 5):
            js.append(jensenshannon(refs[i]["density"], refs[j]["density"]))
            a, b = masks[i] > 0, masks[j] > 0
            dsc.append(2 * (a & b).sum() / (a.sum() + b.sum()))
    assert res.qc.mean_pairwise_jsd == pytest.approx(np.mean(js), rel=1e-7)
    assert res.qc.mean_pairwise_dsc == pytest.approx(np.mean(dsc), rel=1e-12)
    assert res.qc.hu_variance == pytest.approx(np.var([r["mode_1hu"] for r in refs], ddof=1), abs=1e-9)


def test_js_value_is_the_scipy_distance_not_the_log2_divergence():
    rng = np.random.default_rng(3)
    a = rng.normal(-60, 15, 4000)
    b = rng.normal(-45, 15, 4000)
    d = okm._jsd_from_values(a, b)
    p, q = gaussian_kde(a)(G1), gaussian_kde(b)(G1)
    assert d == pytest.approx(jensenshannon(p, q), rel=1e-7)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    log2_divergence = (np.sum(rel_entr(p, m)) + np.sum(rel_entr(q, m))) / 2 / np.log(2)
    assert d == pytest.approx(np.sqrt(log2_divergence * np.log(2)), rel=1e-6)


def test_peaks_outside_the_window_are_not_the_mode():
    # Mostly air with a smaller soft-tissue peak: the mode is the peak inside -300..300 HU.
    rng = np.random.default_rng(5)
    v = np.concatenate([rng.normal(-950, 20, 6000), rng.normal(-40, 15, 2000)])
    got = okm._kde_mode_and_second_ratio(v)
    assert got.mode == pytest.approx(_paper_member(v)["mode_01hu"], abs=1e-9)
    assert -60 < got.mode < -20


def test_two_peaks_at_the_data_extremes_are_detected():
    # Two equal spikes: before, the grid ran from min to max and the peaks sat on its ends.
    v = np.repeat([-100.0, 50.0], 1000)
    assert okm._kde_mode_and_second_ratio(v).second_peak_ratio > 0.5


def test_failed_and_tiny_members_take_part_in_the_ensemble_qc():
    ct = _ct()
    good = _mask(ct.shape, 10, 40, 10, 30)
    tiny = np.zeros(ct.shape, dtype=np.uint8)
    tiny[20, 20, 5] = 1
    res = tq.quantify(_members(ct, [good, good, good, good, tiny]), method="okamura", detail="full")
    assert "too_few_voxels" in res.members[4].flags
    assert "computation_failed" in res.members[4].flags
    # The 1-voxel mask still counts in the DSC (4 of 10 pairs score about 0), as in the paper.
    assert res.qc.mean_pairwise_dsc == pytest.approx(0.6, abs=0.01)
    assert "low_dsc" in res.flags
    assert res.qc.hu_variance is None
    assert "hu_variance_unavailable" in res.flags
    assert res.status == "check"


def test_js_distance_is_finite_where_scipy_underflows_to_inf():
    # Far in a narrow KDE tail a value can be the smallest subnormal; facing a 0 in the other
    # curve, the mean (p + q) / 2 rounds to 0 and SciPy returns inf.
    p = np.array([5e-324, 0.25, 0.5, 0.25])
    q = np.array([0.0, 0.25, 0.5, 0.25])
    with np.errstate(all="ignore"):
        assert np.isinf(jensenshannon(p, q))
    assert okm._js_distance(p, q) == 0.0
    x = np.linspace(-300, 300, 601)
    rng = np.random.default_rng(1)
    a, b = gaussian_kde(rng.normal(-60, 30, 3000))(x), gaussian_kde(rng.normal(-40, 30, 3000))(x)
    assert okm._js_distance(a, b) == pytest.approx(jensenshannon(a, b), rel=1e-12)


def test_method_version_is_okamura_v2():
    ct = _ct()
    res = tq.quantify(_members(ct, [_mask(ct.shape, 10, 40, 10, 30)] * 5), method="okamura")
    assert res.meta.method_version == "okamura_v2"
