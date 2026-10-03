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


def test_constant_hu_region_thymic_tissue_fraction_and_anisotropic_volume():
    ct = np.full((6, 6, 6), -15.0)
    result = tq.quantify(_seg(ct), method="okamura")
    assert result.thymic_tissue_fraction == pytest.approx(0.5)
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

    assert result.thymic_tissue_fraction is None
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
    assert result.thymic_tissue_fraction is not None
    assert result.status == "check"
    assert "all_members_invalid" in result.flags
    assert "used_invalid_members" in result.flags
    assert result.qc.paper_criteria_met is False


def test_all_members_computation_failed_returns_failed(monkeypatch):
    monkeypatch.setattr(okm, "_kde_mode_and_second_ratio", lambda values: (_ for _ in ()).throw(okm.NumericalError("boom")))
    result = tq.quantify(_seg(np.full((6, 6, 6), -15.0)), method="okamura")
    assert result.status == "failed"
    assert result.thymic_tissue_fraction is None


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


def _two_tissue_ct():
    """Air, with a soft-tissue block whose upper half in axis 1 is fat."""
    rng = np.random.default_rng(0)
    ct = rng.normal(-1000, 5, (64, 64, 20)).astype(np.float32)
    ct[10:50, 10:50, 2:18] = rng.normal(-40, 12, (40, 40, 16))
    ct[10:50, 30:50, 2:18] = rng.normal(-110, 12, (40, 20, 16))
    return ct


def _box(shape, y0, y1):
    m = np.zeros(shape, dtype=np.uint8)
    m[15:45, y0:y1, 4:16] = 1
    return m


def _nn_like_seg(ct, masks):
    """Members as TRQseg-v1 returns them: a member mask may be empty."""
    from thymus_quant.inputs import make_image_context

    return tq.SegmentationResult(
        study_id="case",
        image=make_image_context(ct_hu=ct, spacing_mm=(1.0, 1.0, 1.0)),
        members=tuple(tq.SegmentationMember(f"fold-{i}", m) for i, m in enumerate(masks)),
    )


def _alone(ct, mask):
    return tq.quantify(tq.SegmentationResult.from_mask(trq_mask=mask, ct_hu=ct, spacing_mm=(1.0, 1.0, 1.0)), method="okamura")


def test_failed_member_does_not_shift_summary_or_member_values():
    # fold-1 is empty (computation fails); fold-4 spans fat and soft tissue (multimodal, invalid).
    ct = _two_tissue_ct()
    masks = [_box(ct.shape, 12, 28), np.zeros(ct.shape, np.uint8), _box(ct.shape, 13, 29), _box(ct.shape, 14, 28), _box(ct.shape, 18, 42)]
    alone = {i: _alone(ct, m) for i, m in enumerate(masks) if m.any()}
    assert [alone[i].members[0].valid for i in (0, 2, 3, 4)] == [True, True, True, False]

    result = tq.quantify(_nn_like_seg(ct, masks), method="okamura", detail="full")

    valid = (0, 2, 3)
    assert result.etv_ml == pytest.approx(np.median([alone[i].etv_ml for i in valid]))
    assert result.trq_hu_mode == pytest.approx(np.median([alone[i].trq_hu_mode for i in valid]))
    assert result.trq_volume_ml == pytest.approx(np.median([alone[i].trq_volume_ml for i in valid]))
    assert result.thymic_tissue_fraction == pytest.approx(np.median([alone[i].thymic_tissue_fraction for i in valid]))
    assert result.qc.hu_variance == pytest.approx(np.var([alone[i].trq_hu_mode for i in valid], ddof=1))
    assert "high_hu_variance" not in result.flags
    assert result.status == "check"
    assert "computation_failed" in result.members[1].flags

    for name in ("trq_hu_mode", "trq_volume_ml", "etv_ml", "thymic_tissue_fraction"):
        per_member = getattr(result, f"{name}_members")
        assert len(per_member) == len(result.members) == 5
        assert per_member[1] is None
        for i in (0, 2, 3, 4):
            assert per_member[i] == pytest.approx(getattr(alone[i], name))
            assert per_member[i] == pytest.approx(getattr(result.members[i], name))
    assert result.atrq_below_aadipose_members[1] is None
    assert result.to_dict()["member_values"]["etv_ml_members"][1] is None


def test_failed_member_before_the_last_does_not_raise():
    ct = _two_tissue_ct()
    masks = [_box(ct.shape, 12, 28), np.zeros(ct.shape, np.uint8), _box(ct.shape, 13, 29), _box(ct.shape, 14, 28), _box(ct.shape, 12, 27)]
    result = tq.quantify(_nn_like_seg(ct, masks), method="okamura")
    computed = (0, 2, 3, 4)
    assert result.qc.valid_member_count == 4
    assert result.etv_ml == pytest.approx(np.median([_alone(ct, masks[i]).etv_ml for i in computed]))
    assert result.etv_ml_members[1] is None


def test_apply_qc_false_does_not_report_paper_criteria_met():
    ct = _two_tissue_ct()
    wide, thin = _box(ct.shape, 10, 30), _box(ct.shape, 12, 16)
    seg = _nn_like_seg(ct, [wide, wide, thin, thin, thin])  # mean pairwise DSC 0.6
    checked = tq.quantify(seg, method="okamura")
    assert "low_dsc" in checked.flags
    assert checked.qc.paper_criteria_met is False

    unchecked = tq.quantify(seg, method="okamura", options=tq.OkamuraOptions(apply_qc=False))
    assert unchecked.status == "check"
    assert unchecked.qc.status == "check"
    assert unchecked.qc.paper_criteria_met is None
    assert "ensemble_qc_not_applied" in unchecked.flags
    assert "low_dsc" not in unchecked.flags
    assert unchecked.qc.mean_pairwise_dsc == pytest.approx(checked.qc.mean_pairwise_dsc)
    assert unchecked.etv_ml == pytest.approx(checked.etv_ml)

    clean = tq.quantify(_nn_like_seg(ct, [wide] * 5), method="okamura", options=tq.OkamuraOptions(apply_qc=False))
    assert clean.status == "check"
    assert clean.qc.paper_criteria_met is None
    assert tq.quantify(_nn_like_seg(ct, [wide] * 5), method="okamura").status == "ok"
