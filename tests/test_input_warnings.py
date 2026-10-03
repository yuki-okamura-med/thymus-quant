"""Warnings for inputs that are probably not in HU, or whose voxel sizes disagree with the affine.

The warnings are notes only: they must not change values, flags, QC status or
paper_criteria_met.
"""
import logging

import nibabel as nib
import numpy as np
import pytest

import thymus_quant as tq
from thymus_quant.inputs import geometry_report, intensity_report, load_image_context

AFFINE = np.diag([-0.7, 0.7, 2.5, 1.0])


def _chest_like_ct(shape=(40, 40, 12)):
    """Air around a soft-tissue block, in HU."""
    rng = np.random.default_rng(0)
    ct = rng.normal(-1000.0, 5.0, size=shape).astype(np.float32)
    ct[8:32, 8:32, :] = rng.normal(-20.0, 15.0, size=(24, 24, shape[2]))
    return ct


def _mask(shape=(40, 40, 12)):
    mask = np.zeros(shape, dtype=bool)
    mask[12:28, 12:28, 2:10] = True
    return mask


def _quantify(ct, affine=AFFINE, spacing=(0.7, 0.7, 2.5)):
    seg = tq.SegmentationResult.from_mask(trq_mask=_mask(ct.shape), ct_hu=ct, spacing_mm=spacing, affine=affine)
    return seg, tq.quantify(seg, method="okamura")


def _same_result_without_warnings(seg, res):
    seg.image.warnings = ()
    plain = tq.quantify(seg, method="okamura")
    assert res.flags == plain.flags
    assert res.status == plain.status
    assert res.qc.paper_criteria_met == plain.qc.paper_criteria_met
    assert res.etv_ml == plain.etv_ml
    assert res.trq_hu_mode == plain.trq_hu_mode
    assert res.trq_volume_ml == plain.trq_volume_ml


def test_hu_input_has_no_warning():
    seg, res = _quantify(_chest_like_ct())
    assert seg.image.intensity["looks_like_hu"] is True
    assert seg.image.intensity["hu_p01"] < -900
    assert not res.warnings
    assert res.to_record()["warnings"] == ()
    assert res.meta.intensity == seg.image.intensity


@pytest.mark.parametrize(
    "transform",
    [
        lambda ct: ct + 1024.0,  # rescale intercept not applied
        lambda ct: np.clip(ct, -160.0, 240.0),  # mediastinal window
        lambda ct: (ct + 1024.0) / 4095.0,  # normalized to 0-1
    ],
)
def test_values_that_do_not_look_like_hu_are_warned_without_changing_results(transform, caplog):
    ct = transform(_chest_like_ct())
    with caplog.at_level(logging.WARNING, logger="thymus_quant.inputs"):
        seg, res = _quantify(ct)
    assert seg.image.intensity["looks_like_hu"] is False
    notes = [w for w in res.warnings if "do not look like HU" in w]
    assert len(notes) == 1
    assert notes[0] in res.to_record()["warnings"]
    assert any("do not look like HU" in r.getMessage() for r in caplog.records)
    assert not any("HU" in f for f in res.flags)
    _same_result_without_warnings(seg, res)


def test_nifti_input_gets_the_same_check():
    img = nib.Nifti1Image(_chest_like_ct() + 1024.0, AFFINE)
    ctx = load_image_context(img)
    assert ctx.intensity["looks_like_hu"] is False
    assert any("do not look like HU" in w for w in ctx.warnings)
    assert not load_image_context(nib.Nifti1Image(_chest_like_ct(), AFFINE)).warnings


def test_intensity_report_ignores_nonfinite_values():
    ct = _chest_like_ct()
    ct[0, :, :] = np.nan
    rep = intensity_report(ct)
    assert np.isfinite(rep["hu_p01"]) and rep["looks_like_hu"] is True


def test_header_voxel_sizes_disagreeing_with_affine_are_warned(caplog):
    img = nib.Nifti1Image(_chest_like_ct(), AFFINE)
    img.header.set_zooms((1.0, 1.0, 1.0))  # header says 1 mm, affine says 0.7 x 0.7 x 2.5 mm
    with caplog.at_level(logging.WARNING, logger="thymus_quant.inputs"):
        ctx = load_image_context(img)
    assert ctx.spacing_mm == (1.0, 1.0, 1.0)  # volumes still use the header voxel sizes
    assert ctx.geometry["voxel_volume_mismatch_rel"] == pytest.approx(abs(1.0 - 1.225) / 1.225)
    assert any("disagree with the affine" in w for w in ctx.warnings)
    assert any("disagree with the affine" in r.getMessage() for r in caplog.records)

    seg, res = _quantify(_chest_like_ct(), affine=AFFINE, spacing=(1.0, 1.0, 1.0))
    assert any("disagree with the affine" in w for w in res.warnings)
    assert res.trq_volume_ml == pytest.approx(_mask().sum() * 1.0 / 1000)
    _same_result_without_warnings(seg, res)


def test_matching_sheared_or_float32_rounded_voxel_sizes_are_not_warned():
    sheared = AFFINE.copy()
    sheared[1, 2] = 0.5  # gantry tilt: column norm changes, voxel volume does not
    rep = geometry_report(sheared, spacing_mm=(0.7, 0.7, 2.5))
    assert rep["voxel_size_mismatch_max_mm"] > 0.0
    assert rep["voxel_volume_mismatch_rel"] == pytest.approx(0.0, abs=1e-9)
    _, res = _quantify(_chest_like_ct(), affine=sheared)
    assert not any("disagree with the affine" in w for w in res.warnings)

    img = nib.Nifti1Image(_chest_like_ct(), AFFINE)  # header zooms stored as float32
    assert not load_image_context(img).warnings


def test_array_inputs_without_affine_get_no_geometry_warning():
    seg, res = _quantify(_chest_like_ct(), affine=None)
    assert seg.image.geometry is None
    assert not res.warnings
