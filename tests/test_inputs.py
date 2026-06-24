import numpy as np
import pytest

import thymus_quant as tq
from thymus_quant.inputs import make_image_context, validate_binary_mask


def test_from_mask_accepts_bool_mask_and_spacing():
    ct = np.zeros((4, 4, 4), dtype=float)
    mask = np.zeros_like(ct, dtype=bool)
    mask[1:3, 1:3, 1:3] = True

    seg = tq.SegmentationResult.from_mask(
        trq_mask=mask,
        ct_hu=ct,
        spacing_mm=(0.7, 0.8, 1.5),
        study_id="case-001",
    )

    assert seg.image.spacing_mm == (0.7, 0.8, 1.5)
    assert seg.image.shape == (4, 4, 4)
    assert seg.trq_mask.dtype == bool


def test_shape_mismatch_rejected():
    ct = np.zeros((4, 4, 4))
    mask = np.ones((4, 4, 3))
    with pytest.raises(ValueError, match="shape"):
        tq.SegmentationResult.from_mask(trq_mask=mask, ct_hu=ct, spacing_mm=(1, 1, 1))


@pytest.mark.parametrize("ct", [np.zeros((4, 4)), np.zeros((2, 2, 2, 2))])
def test_ct_must_be_3d(ct):
    with pytest.raises(ValueError, match="3D"):
        make_image_context(ct_hu=ct, spacing_mm=(1, 1, 1))


def test_integer_binary_mask_ok_and_invalid_values_rejected():
    assert validate_binary_mask(np.array([[[0, 1]]], dtype=np.uint8)).dtype == bool
    with pytest.raises(ValueError, match="binary"):
        validate_binary_mask(np.array([[[0, 2]]], dtype=np.uint8))


@pytest.mark.parametrize("spacing", [(0, 1, 1), (-1, 1, 1), (np.nan, 1, 1), (np.inf, 1, 1), (1, 1)])
def test_invalid_spacing_rejected(spacing):
    with pytest.raises(ValueError, match="spacing"):
        make_image_context(ct_hu=np.zeros((2, 2, 2)), spacing_mm=spacing)


def test_empty_mask_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        tq.SegmentationResult.from_mask(
            trq_mask=np.zeros((2, 2, 2), dtype=bool),
            ct_hu=np.zeros((2, 2, 2)),
            spacing_mm=(1, 1, 1),
        )


def test_orientation_is_recorded_from_affine_flip():
    affine = np.diag([-1, 1, 1, 1])
    seg = tq.SegmentationResult.from_mask(
        trq_mask=np.ones((2, 2, 2), dtype=bool),
        ct_hu=np.zeros((2, 2, 2)),
        spacing_mm=(1, 1, 1),
        affine=affine,
    )
    assert seg.image.orientation[0] == "L"
