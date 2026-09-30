"""TRQseg-v1 with real weights: the same CT in different voxel orders gives the same masks.

Opt-in integration test. Runs only when THYQ_RUN_MODEL_TESTS=1, the TRQseg-v1
weights are available (for example THYQ_TRQSEG_V1_LOCAL_REPO), and the public
COVID-19 sample used by tests/run_covid_inference_smoke.py is present in
tests/test_images. One member (fold-0) is used to keep the run short on CPU.
"""
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import thymus_quant as tq

SAMPLE = Path(__file__).parent / "test_images" / "volume-covid19-A-0700_day000_0000.nii.gz"
LAS = ("L", "A", "S")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("THYQ_RUN_MODEL_TESTS") != "1", reason="set THYQ_RUN_MODEL_TESTS=1 to run"),
    pytest.mark.skipif(not SAMPLE.exists(), reason=f"sample not found: {SAMPLE}"),
]


def _reorient(img, codes):
    o = nib.orientations
    return img.as_reoriented(o.ornt_transform(o.io_orientation(img.affine), o.axcodes2ornt(codes)))


def _to_las(arr, codes):
    o = nib.orientations
    return o.apply_orientation(np.asarray(arr), o.ornt_transform(o.axcodes2ornt(codes), o.axcodes2ornt(LAS)))


@pytest.fixture(scope="module")
def segmentor():
    return tq.load_segmentor("trqseg_v1", members=["fold-0"], device=os.environ.get("THYQ_TEST_DEVICE", "auto"))


@pytest.fixture(scope="module")
def reference(segmentor):
    img_las = _reorient(nib.load(str(SAMPLE)), LAS)
    seg = segmentor.segment_trq(img_las, study_id="las")
    assert seg.preprocessing["orientation_status"] == "model_orientation"
    assert seg.trq_mask.sum() > 0
    return img_las, seg


@pytest.mark.parametrize("codes", [("R", "A", "S"), ("L", "P", "S"), ("S", "A", "R")], ids=lambda c: "".join(c))
def test_same_masks_in_any_voxel_order(segmentor, reference, codes):
    img_las, ref = reference
    img_v = _reorient(img_las, codes)
    seg = segmentor.segment_trq(img_v, study_id="".join(codes))
    assert seg.preprocessing["orientation_status"] == "reoriented"
    assert seg.trq_mask.shape == img_v.shape
    np.testing.assert_array_equal(_to_las(seg.trq_mask, codes), ref.trq_mask)
    np.testing.assert_array_equal(_to_las(seg.members[0].airway_mask, codes), ref.members[0].airway_mask)

    r_ref = tq.quantify(ref, method="okamura")
    r_v = tq.quantify(seg, method="okamura")
    assert r_v.trq_volume_ml == pytest.approx(r_ref.trq_volume_ml, rel=1e-9)
    assert r_v.trq_hu_mode == pytest.approx(r_ref.trq_hu_mode, rel=1e-9)
    assert r_v.etv_ml == pytest.approx(r_ref.etv_ml, rel=1e-9)
