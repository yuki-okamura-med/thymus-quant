"""NIfTI header checks: spatial units, unspecified or contradictory orientation, unusable affines,
voxel sizes per axis, and headers that look like library defaults."""
import nibabel as nib
import numpy as np
import pytest

import thymus_quant as tq
from thymus_quant.exceptions import InputValidationError
from thymus_quant.inputs import ImageContext, geometry_report, load_image_context, make_image_context
from thymus_quant.segmentors import _model_orientation_transforms

LAS = np.diag([-0.7, 0.7, 2.5, 1.0])


def _ct(shape=(30, 30, 8)):
    rng = np.random.default_rng(0)
    ct = rng.normal(-1000.0, 5.0, size=shape).astype(np.float32)
    ct[6:24, 6:24, :] = rng.normal(-30.0, 15.0, size=(18, 18, shape[2]))
    return ct


def _notes(ctx, text):
    return [w for w in ctx.warnings if text in w]


@pytest.mark.parametrize("unit, scale", [("meter", 1000.0), ("micron", 0.001)])
def test_spatial_units_are_converted_to_mm(unit, scale):
    img = nib.Nifti1Image(_ct(), np.diag([-0.7 / scale, 0.7 / scale, 2.5 / scale, 1.0]))
    img.header.set_xyzt_units(xyz=unit)
    ctx = load_image_context(img)
    assert ctx.spacing_mm == pytest.approx((0.7, 0.7, 2.5))
    assert np.allclose(ctx.affine[:3, :3], LAS[:3, :3])
    assert ctx.geometry["spatial_unit"] == unit
    assert _notes(ctx, f"spatial unit is {unit}")
    mask = np.zeros(ctx.shape, dtype=bool)
    mask[8:22, 8:22, 2:6] = True
    seg = tq.SegmentationResult.from_mask(trq_mask=mask, ct_hu=ctx.ct_hu, spacing_mm=ctx.spacing_mm)
    assert tq.quantify(seg, method="okamura").trq_volume_ml == pytest.approx(mask.sum() * 0.7 * 0.7 * 2.5 / 1000)


def test_mm_and_unknown_units_are_left_as_they_are():
    for unit in ("mm", "unknown"):
        img = nib.Nifti1Image(_ct(), LAS)
        img.header.set_xyzt_units(xyz=unit)
        ctx = load_image_context(img)
        assert ctx.spacing_mm == pytest.approx((0.7, 0.7, 2.5))
        assert not ctx.warnings


def test_unspecified_orientation_is_not_reported_as_las():
    img = nib.Nifti1Image(_ct(), LAS)
    img.set_qform(None, code=0)
    img.set_sform(None, code=0)
    img = nib.Nifti1Image.from_bytes(img.to_bytes())  # as read from a file: nibabel falls back to LAS
    assert nib.aff2axcodes(img.affine) == ("L", "A", "S")
    ctx = load_image_context(img)
    assert ctx.orientation is None
    assert ctx.geometry["orientation_source"] == "fallback"
    assert _notes(ctx, "orientation is unspecified")
    assert _model_orientation_transforms(ctx.orientation)[2] == "assumed_model_orientation"


def test_opposite_qform_and_sform_handedness_is_warned():
    img = nib.Nifti1Image(_ct(), LAS)
    img.set_sform(LAS, code=1)
    img.set_qform(np.diag([0.7, 0.7, 2.5, 1.0]), code=1)  # RAS: left and right swapped
    ctx = load_image_context(img)
    assert ctx.geometry["qform_sform_handedness_differs"] is True
    assert _notes(ctx, "disagree on left and right")


def test_singular_affine_gives_unknown_orientation():
    aff = LAS.copy()
    aff[:3, 2] = aff[:3, 1]  # third axis parallel to the second
    ctx = make_image_context(ct_hu=_ct(), spacing_mm=(0.7, 0.7, 2.5), affine=aff)
    assert ctx.geometry["valid_affine"] is False
    assert ctx.orientation is None
    assert _notes(ctx, "affine is not usable")


def test_voxel_sizes_are_compared_per_axis_without_shear():
    # Sizes that cancel out in the voxel volume are caught per axis.
    rep = geometry_report(np.diag([2.0, 0.5, 1.0, 1.0]), spacing_mm=(1.0, 1.0, 1.0))
    assert rep["voxel_volume_mismatch_rel"] == pytest.approx(0.0)
    assert rep["voxel_size_mismatch_rel_max"] == pytest.approx(1.0)
    ctx = make_image_context(ct_hu=_ct(), spacing_mm=(1.0, 1.0, 1.0), affine=np.diag([2.0, 0.5, 1.0, 1.0]))
    assert _notes(ctx, "disagree with the affine")
    # Gantry tilt: the slice axis is sheared; its length without shear still equals the header size.
    tilt = LAS.copy()
    tilt[1, 2] = 0.5
    rep = geometry_report(tilt, spacing_mm=(0.7, 0.7, 2.5))
    assert rep["voxel_sizes_affine_unsheared_mm"] == pytest.approx([0.7, 0.7, 2.5])
    assert rep["voxel_size_mismatch_rel_max"] == pytest.approx(0.0, abs=1e-12)
    # Header slice spacing measured along the tilted axis (longer than the perpendicular spacing).
    rep = geometry_report(tilt, spacing_mm=(0.7, 0.7, float(np.hypot(2.5, 0.5))))
    assert rep["voxel_size_mismatch_rel_max"] > 0.01


@pytest.mark.parametrize("affine", [np.eye(4), np.diag([-1.0, -1.0, 1.0, 1.0])])
def test_library_default_like_headers_get_a_note(affine):
    # nibabel.Nifti1Image(arr, np.eye(4)), or SimpleITK's default direction written as NIfTI.
    ctx = load_image_context(nib.Nifti1Image(_ct(), affine))
    assert ctx.geometry["default_like_header"] is True
    assert _notes(ctx, "looks like library defaults")
    moved = affine.copy()
    moved[:3, 3] = [-150.0, -150.0, 300.0]
    assert not load_image_context(nib.Nifti1Image(_ct(), moved)).warnings


def test_image_context_built_directly_reads_orientation_from_affine():
    ras = np.diag([0.7, 0.7, 2.5, 1.0])
    ctx = ImageContext(ct_hu=_ct(), spacing_mm=(0.7, 0.7, 2.5), affine=ras)
    assert ctx.orientation == ("R", "A", "S")
    assert _model_orientation_transforms(ctx.orientation)[2] == "reoriented"
    assert ImageContext(ct_hu=_ct(), spacing_mm=(0.7, 0.7, 2.5)).orientation is None


def test_explicit_orientation_must_agree_with_the_affine():
    with pytest.raises(InputValidationError, match="contradicts the affine"):
        make_image_context(ct_hu=_ct(), spacing_mm=(0.7, 0.7, 2.5), affine=LAS, orientation=("R", "A", "S"))
    ctx = make_image_context(ct_hu=_ct(), spacing_mm=(0.7, 0.7, 2.5), affine=LAS, orientation=("L", "A", "S"))
    assert ctx.orientation == ("L", "A", "S")
    assert make_image_context(ct_hu=_ct(), spacing_mm=(0.7, 0.7, 2.5), orientation=("R", "A", "S")).orientation == ("R", "A", "S")
