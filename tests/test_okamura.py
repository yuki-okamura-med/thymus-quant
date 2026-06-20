import numpy as np
import pytest

import thymus_quant as tq
import thymus_quant.quantification.okamura as okm


def _mask(shape=(6, 6, 6)):
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    return mask


def _seg(ct, masks=None):
    masks = [_mask(ct.shape)] if masks is None else masks
    return tq.SegmentationResult.from_member_masks(
        member_masks=masks,
        ct_hu=ct,
        spacing_mm=(1.0, 2.0, 3.0),
        study_id="case",
        member_ids=[f"fold-{i}" for i in range(len(masks))],
    )


def test_constant_hu_region_ptt_and_anisotropic_volume():
    ct = np.full((6, 6, 6), -15.0)
    result = tq.quantify(_seg(ct), method="okamura")
    assert result.ptt == pytest.approx(50.0)
    assert result.trq_volume_ml == pytest.approx(64 * 6 / 1000)
    assert result.etv_ml == pytest.approx(result.trq_volume_ml * 0.5)
    assert "ensemble_qc_unavailable" in result.qc.flags
    assert result.qc.paper_criteria_met is None


def test_kde_exception_is_not_median_mode(monkeypatch):
    class BadKde:
        def __init__(self, values):
            raise RuntimeError("singular")

    monkeypatch.setattr(okm, "gaussian_kde", BadKde)
    ct = np.linspace(-100, 20, 216).reshape(6, 6, 6)
    result = tq.quantify(_seg(ct), method="okamura", detail="full")

    assert result.ptt is None
    assert result.status == "failed"
    assert "kde_failed" in result.flags
    assert result.members[0].trq_hu_mode is None
    assert result.members[0].failure_reason is not None


def test_valid_and_invalid_member_mixture_uses_valid_summary():
    ct1 = np.full((6, 6, 6), -15.0)
    ct2 = ct1.copy()
    ct2[1:-1, 1:-1, 1:-1] = np.tile(np.array([-100.0, 50.0]), 32).reshape(4, 4, 4)
    masks = [_mask(), _mask()]
    seg = _seg(ct2, masks=masks)
    result = tq.quantify(seg, method="okamura", options=tq.OkamuraOptions(second_peak_ratio_threshold=-1.0))
    assert result.status == "check"
    assert "invalid_member" in result.flags
    assert result.qc.paper_criteria_met is False


def test_all_members_qc_invalid_but_computable_returns_check_with_flags():
    ct = np.full((6, 6, 6), -15.0)
    result = tq.quantify(
        _seg(ct, masks=[_mask(), _mask()]),
        method="okamura",
        options=tq.OkamuraOptions(second_peak_ratio_threshold=-1.0),
    )
    assert result.ptt is not None
    assert result.status == "check"
    assert "all_members_invalid" in result.flags
    assert "used_invalid_members" in result.flags
    assert result.qc.paper_criteria_met is False


def test_all_members_computation_failed_returns_failed(monkeypatch):
    monkeypatch.setattr(okm, "_kde_mode_and_second_ratio", lambda values: (_ for _ in ()).throw(okm.NumericalError("boom")))
    result = tq.quantify(_seg(np.full((6, 6, 6), -15.0)), method="okamura")
    assert result.status == "failed"
    assert result.ptt is None


def test_missing_spacing_does_not_return_nan():
    seg = tq.SegmentationResult(trq_mask=np.ones((2, 2, 2)), members=(tq.SegmentationMember("m0", np.ones((2, 2, 2))),))
    result = tq.quantify(seg, method="okamura")
    assert result.status == "failed"
    assert result.trq_volume_ml is None


def test_jsd_identity_symmetry_and_dice_edges():
    vals = np.array([-10, 0, 10, 20], dtype=float)
    assert okm._jsd_from_values(vals, vals) == pytest.approx(0.0)
    assert okm._jsd_from_values(vals, vals + 1) == pytest.approx(okm._jsd_from_values(vals + 1, vals))
    from thymus_quant.quantification.common import dice

    assert dice(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
    assert dice(np.ones((2, 2)), np.zeros((2, 2))) == 0.0
