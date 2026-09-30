"""Input voxel order handling for TRQseg-v1 segmentation.

The network is replaced by a position-dependent fake so these tests run without
model weights: the fake labels bright voxels as TRQ only in one half of the
model-space slice and dark voxels as airway only in the other half. A flipped or
permuted array therefore gives a different answer unless the input is brought
back to LAS before the network, which is what these tests check.
"""
import logging
import types

import nibabel as nib
import numpy as np
import pytest

import thymus_quant as tq
from thymus_quant.inputs import geometry_report, make_image_context
from thymus_quant.segmentors import LoadedSegmentor, SegmentorInfo, SegmentorMember

LAS = ("L", "A", "S")
ORDERS = [
    ("L", "P", "S"),
    ("R", "A", "S"),
    ("R", "P", "I"),
    ("L", "A", "I"),
    ("S", "A", "R"),
    ("A", "S", "L"),
    ("P", "I", "R"),
]


def _fake_predict(self, model, x):
    """Position-dependent fake network on the model-space input (Z, 3, H, W)."""
    v = x[:, 0]
    z, h, w = v.shape
    rows = np.arange(h)[None, :, None]
    cols = np.arange(w)[None, None, :]
    trq = (v > 0.05 + 0.02 * model.k) & (rows < h // 2) & (cols >= w // 3)  # members differ slightly
    airway = (v < -0.9) & (rows >= h // 2)
    logits = np.zeros((z, 3, h, w), dtype=np.float32)
    logits[:, 0] = 1.0
    logits[:, 1] = np.where(airway, 2.0, 0.0)
    logits[:, 2] = np.where(trq, 3.0, 0.0)
    return logits


@pytest.fixture
def fake_segmentor(monkeypatch):
    info = SegmentorInfo(name="trqseg_v1", repo_id="fake/fake", preprocessing_version="trqseg_v1_preprocess_v1")
    seg = LoadedSegmentor(info=info, members=tuple(SegmentorMember(member_id=f"fold-{i}") for i in range(3)))
    seg._models = {f"fold-{i}": types.SimpleNamespace(k=i) for i in range(3)}
    monkeypatch.setattr(LoadedSegmentor, "_predict_logits", _fake_predict)
    return seg


def _las_image(shape=(15, 11, 4)):
    rng = np.random.default_rng(0)
    ct = rng.uniform(-1000, 300, size=shape).astype(np.float32)
    ct[2:9, 1:5, :] = 200.0  # bright block
    ct[3:12, 7:10, :] = -1000.0  # air block
    affine = np.diag([-0.7, 0.7, 2.5, 1.0])
    affine[:3, 3] = [100.0, -80.0, -300.0]
    img = nib.Nifti1Image(ct, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    assert nib.aff2axcodes(img.affine) == LAS
    return img


def _reorient(img, codes):
    o = nib.orientations
    return img.as_reoriented(o.ornt_transform(o.io_orientation(img.affine), o.axcodes2ornt(codes)))


def _to_las(arr, codes):
    o = nib.orientations
    return o.apply_orientation(np.asarray(arr), o.ornt_transform(o.axcodes2ornt(codes), o.axcodes2ornt(LAS)))


def test_las_input_is_used_as_is(fake_segmentor):
    img = _las_image()
    seg = fake_segmentor.segment_trq(img, study_id="las")
    assert seg.preprocessing["orientation_status"] == "model_orientation"
    assert seg.preprocessing["reoriented_for_model"] is False
    assert seg.preprocessing["input_orientation"] == "LAS"
    assert seg.trq_mask.shape == img.shape
    assert seg.trq_mask.any()
    for m in seg.members:
        assert m.raw_output["reoriented_for_model"] is False


@pytest.mark.parametrize("codes", ORDERS, ids=lambda c: "".join(c))
def test_non_las_input_gives_the_same_masks(fake_segmentor, codes):
    img = _las_image()
    ref = fake_segmentor.segment_trq(img, study_id="ref")
    img_v = _reorient(img, codes)
    assert nib.aff2axcodes(img_v.affine) == codes
    seg = fake_segmentor.segment_trq(img_v, study_id="v")

    assert seg.preprocessing["orientation_status"] == "reoriented"
    assert seg.preprocessing["reoriented_for_model"] is True
    assert seg.preprocessing["input_orientation"] == "".join(codes)
    # returned on the input grid, in the input voxel order
    assert seg.trq_mask.shape == img_v.shape
    assert seg.trq_mask.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(_to_las(seg.trq_mask, codes), ref.trq_mask)
    for m_v, m_ref in zip(seg.members, ref.members):
        assert m_v.trq_mask.shape == img_v.shape
        np.testing.assert_array_equal(_to_las(m_v.trq_mask, codes), m_ref.trq_mask)
        np.testing.assert_array_equal(_to_las(m_v.airway_mask, codes), m_ref.airway_mask)


def test_fake_network_is_orientation_sensitive(fake_segmentor):
    """Without reorientation the fake gives another answer, so the test above is meaningful."""
    img = _las_image()
    ref = fake_segmentor.segment_trq(img, study_id="ref")
    codes = ("L", "P", "S")
    img_v = _reorient(img, codes)
    ctx = make_image_context(ct_hu=np.asarray(img_v.dataobj), spacing_mm=img_v.header.get_zooms()[:3])
    seg = fake_segmentor.segment_trq(ctx, study_id="as_given")
    assert seg.preprocessing["orientation_status"] == "assumed_model_orientation"
    assert not np.array_equal(_to_las(seg.trq_mask, codes), ref.trq_mask)


def test_array_input_without_orientation_is_used_as_given(fake_segmentor):
    img = _las_image()
    ref = fake_segmentor.segment_trq(img, study_id="ref")
    ctx = make_image_context(ct_hu=np.asarray(img.dataobj), spacing_mm=(0.7, 0.7, 2.5))
    seg = fake_segmentor.segment_trq(ctx, study_id="arr")
    assert seg.preprocessing["orientation_status"] == "assumed_model_orientation"
    assert seg.preprocessing["reoriented_for_model"] is False
    assert seg.preprocessing["input_orientation"] is None
    np.testing.assert_array_equal(seg.trq_mask, ref.trq_mask)


def test_array_input_with_explicit_orientation_is_reoriented(fake_segmentor):
    img = _las_image()
    ref = fake_segmentor.segment_trq(img, study_id="ref")
    codes = ("L", "P", "S")
    img_v = _reorient(img, codes)
    ctx = make_image_context(ct_hu=np.asarray(img_v.dataobj), spacing_mm=img_v.header.get_zooms()[:3], orientation=codes)
    seg = fake_segmentor.segment_trq(ctx, study_id="arr")
    assert seg.preprocessing["orientation_status"] == "reoriented"
    np.testing.assert_array_equal(_to_las(seg.trq_mask, codes), ref.trq_mask)


def test_unresolved_orientation_is_used_as_given(fake_segmentor, caplog):
    img = _las_image()
    ref = fake_segmentor.segment_trq(img, study_id="ref")
    ctx = make_image_context(ct_hu=np.asarray(img.dataobj), spacing_mm=(0.7, 0.7, 2.5), orientation=("X", "Y", "Z"))
    with caplog.at_level(logging.WARNING, logger="thymus_quant.segmentors"):
        seg = fake_segmentor.segment_trq(ctx, study_id="bad")
    assert seg.preprocessing["orientation_status"] == "unresolved"
    assert seg.preprocessing["reoriented_for_model"] is False
    np.testing.assert_array_equal(seg.trq_mask, ref.trq_mask)
    assert any("could not be interpreted" in r.getMessage() for r in caplog.records)


def test_reorientation_is_logged_at_info(fake_segmentor, caplog):
    img_v = _reorient(_las_image(), ("L", "P", "S"))
    with caplog.at_level(logging.INFO, logger="thymus_quant.segmentors"):
        fake_segmentor.segment_trq(img_v, study_id="v")
    recs = [r for r in caplog.records if "reorienting voxel array" in r.getMessage()]
    assert recs and all(r.levelno == logging.INFO for r in recs)


def test_even_and_odd_shapes(fake_segmentor):
    for shape in [(16, 12, 4), (15, 11, 3), (9, 14, 5)]:
        img = _las_image(shape)
        ref = fake_segmentor.segment_trq(img, study_id="ref")
        for codes in [("R", "P", "S"), ("S", "A", "R")]:
            img_v = _reorient(img, codes)
            seg = fake_segmentor.segment_trq(img_v, study_id="v")
            assert seg.trq_mask.shape == img_v.shape
            np.testing.assert_array_equal(_to_las(seg.trq_mask, codes), ref.trq_mask)


def test_quantify_meta_and_to_dict_record_orientation(fake_segmentor):
    img_v = _reorient(_las_image(), ("L", "P", "S"))
    seg = fake_segmentor.segment_trq(img_v, study_id="v")
    res = tq.quantify(seg, method="okamura")
    assert res.meta.orientation == ("L", "P", "S")
    assert res.meta.model_orientation == "LAS"
    assert res.meta.orientation_status == "reoriented"
    assert res.meta.reoriented_for_model is True
    assert res.meta.geometry["valid_affine"] is True
    d = seg.to_dict()
    assert d["preprocessing"]["reoriented_for_model"] is True
    assert d["image"]["geometry"]["obliquity_max_deg"] == pytest.approx(0.0, abs=1e-6)
    assert res.to_dict()["meta"]["reoriented_for_model"] is True


def test_external_mask_results_have_no_model_preprocessing():
    ct = np.full((5, 5, 5), -20.0)
    mask = np.zeros_like(ct, dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    seg = tq.SegmentationResult.from_mask(trq_mask=mask, ct_hu=ct, spacing_mm=(1, 1, 1))
    res = tq.quantify(seg, method="okamura")
    assert seg.preprocessing is None
    assert res.meta.reoriented_for_model is None
    assert res.meta.geometry is None


def test_geometry_report_orthogonal_oblique_sheared_and_forms():
    aff = np.diag([-0.7, 0.7, 2.5, 1.0])
    rep = geometry_report(aff, spacing_mm=(0.7, 0.7, 2.5))
    assert rep["valid_affine"] is True
    assert rep["obliquity_max_deg"] == pytest.approx(0.0, abs=1e-9)
    assert rep["shear_max"] == pytest.approx(0.0, abs=1e-12)
    assert rep["voxel_size_mismatch_max_mm"] == pytest.approx(0.0, abs=1e-9)
    assert rep["voxel_sizes_affine_mm"] == pytest.approx([0.7, 0.7, 2.5])

    t = np.deg2rad(10.0)
    rot = np.eye(4)
    rot[1:3, 1:3] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    rep = geometry_report(rot @ aff)
    assert rep["obliquity_max_deg"] == pytest.approx(10.0, abs=1e-6)
    assert rep["shear_max"] == pytest.approx(0.0, abs=1e-9)

    sheared = aff.copy()
    sheared[1, 2] = 0.5  # slice-to-slice shift, as with gantry tilt
    rep = geometry_report(sheared, spacing_mm=(0.7, 0.7, 2.5))
    assert rep["shear_max"] > 0.1
    assert rep["voxel_size_mismatch_max_mm"] > 0.0

    assert geometry_report(np.full((4, 4), np.nan)) == {"valid_affine": False}
    assert geometry_report(np.zeros((4, 4))) == {"valid_affine": False}
    assert geometry_report(None) is None

    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), aff)
    img.set_qform(aff, code=1)
    moved = aff.copy()
    moved[0, 3] = 5.0
    img.set_sform(moved, code=2)
    rep = geometry_report(img.affine, image=img)
    assert rep["qform_code"] == 1
    assert rep["sform_code"] == 2
    assert rep["qform_sform_max_abs_diff"] == pytest.approx(5.0)


def test_heuristic_segmentor_is_unchanged():
    ct = np.full((10, 10, 6), 50.0)
    img = nib.Nifti1Image(ct, np.diag([1.0, -1.0, 1.0, 1.0]))
    seg_h = tq.load_segmentor("heuristic_trq")
    with pytest.warns(UserWarning):
        out = seg_h.segment_trq(img, study_id="h")
    assert out.preprocessing is None
