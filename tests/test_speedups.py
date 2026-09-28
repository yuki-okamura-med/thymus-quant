"""The speed-ups must not change any number.

Each test compares the faster code path with the original computation it replaced.
"""

import numpy as np
import pytest
from scipy.stats import gaussian_kde as scipy_gaussian_kde

import thymus_quant as tq
from thymus_quant.inputs import validate_binary_mask
from thymus_quant.quantification import okamura as okm
from thymus_quant.quantification.common import dice
from thymus_quant.quantification.kde import GroupedGaussianKDE
from thymus_quant.segmentors import _majority_vote


def _hu_like(rng, n):
    return np.round(np.concatenate([rng.normal(-100, 25, int(n * 0.8)), rng.normal(40, 30, n - int(n * 0.8))]))


@pytest.mark.parametrize("n", [2, 50, 5_000, 60_000])
def test_grouped_kde_matches_scipy_on_integer_hu(n):
    rng = np.random.default_rng(n)
    v = _hu_like(rng, n) if n > 2 else np.array([-100.0, 30.0])
    grid = np.linspace(v.min() - 20, v.max() + 20, 777)
    ref = scipy_gaussian_kde(v)
    new = GroupedGaussianKDE(v)
    assert new.factor == pytest.approx(ref.factor, rel=1e-12)
    assert new.covariance == pytest.approx(float(ref.covariance[0, 0]), rel=1e-12)
    np.testing.assert_allclose(new(grid), ref(grid), rtol=1e-9, atol=1e-15)


def test_grouped_kde_matches_scipy_on_non_integer_values():
    v = np.random.default_rng(1).normal(0.3, 2.0, 3_000)
    grid = np.linspace(-8, 8, 301)
    np.testing.assert_allclose(GroupedGaussianKDE(v)(grid), scipy_gaussian_kde(v)(grid), rtol=1e-9, atol=1e-15)


def test_grouped_kde_rejects_what_scipy_rejects():
    with pytest.raises(ValueError):
        GroupedGaussianKDE(np.array([1.0]))
    with pytest.raises(np.linalg.LinAlgError):
        GroupedGaussianKDE(np.full(10, -50.0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mode_ratio_and_jsd_pdf_unchanged(monkeypatch, seed):
    v = _hu_like(np.random.default_rng(seed), 20_000)
    new_mode = okm._kde_mode_and_second_ratio(v)
    new_pdf = okm._kde_pdf_for_jsd(v)[1]
    monkeypatch.setattr(okm, "gaussian_kde", scipy_gaussian_kde)
    ref_mode = okm._kde_mode_and_second_ratio(v)
    ref_pdf = okm._kde_pdf_for_jsd(v)[1]
    assert new_mode[0] == ref_mode[0]
    assert new_mode[1] == pytest.approx(ref_mode[1], rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(new_pdf, ref_pdf, rtol=1e-9, atol=1e-18)


def _synthetic_study(seed=0, shape=(48, 44, 30), n_members=5):
    rng = np.random.default_rng(seed)
    ct = np.round(rng.normal(-100, 40, size=shape)).astype(np.float32)
    masks = []
    for k in range(n_members):
        m = np.zeros(shape, dtype=np.uint8)
        m[10 + k:30, 8:26 - k, 5:20 + k] = 1
        masks.append(m)
    return ct, masks


def test_quantify_results_unchanged_against_scipy_kde(monkeypatch):
    ct, masks = _synthetic_study()
    ids = [f"fold-{k}" for k in range(len(masks))]

    def run():
        seg = tq.SegmentationResult.from_member_masks(
            member_masks=masks, ct_hu=ct, spacing_mm=(0.7, 0.7, 1.0), study_id="s", member_ids=ids)
        return tq.quantify(seg, method="okamura", detail="summary").to_dict()

    new = run()
    monkeypatch.setattr(okm, "gaussian_kde", scipy_gaussian_kde)
    ref = run()
    assert new["status"] == ref["status"]
    assert new["flags"] == ref["flags"]
    for key in ("trq_hu_mode", "trq_volume_ml", "etv_ml", "thymic_tissue_fraction"):
        assert new["summary"][key] == pytest.approx(ref["summary"][key], rel=1e-12)
    for key in ("mean_pairwise_dsc", "mean_pairwise_jsd", "hu_variance"):
        assert new["qc"][key] == pytest.approx(ref["qc"][key], rel=1e-9, abs=1e-15)


def test_pairwise_dsc_on_cropped_box_equals_full_arrays():
    _, masks = _synthetic_study(seed=3, shape=(64, 64, 40))
    bool_masks = [m.astype(bool) for m in masks]
    box = okm._union_bbox(bool_masks)
    for i in range(len(bool_masks)):
        for j in range(i + 1, len(bool_masks)):
            assert dice(bool_masks[i][box], bool_masks[j][box]) == dice(bool_masks[i], bool_masks[j])


def test_union_bbox_of_empty_masks_is_full_extent():
    box = okm._union_bbox([np.zeros((3, 4, 5), dtype=bool)])
    assert box == (slice(0, 3), slice(0, 4), slice(0, 5))


@pytest.mark.parametrize("n_members", [1, 2, 3, 4, 5, 6])
def test_majority_vote_equals_float_mean(n_members):
    rng = np.random.default_rng(n_members)
    masks = [(rng.random((9, 8, 7)) > 0.5).astype(np.uint8) for _ in range(n_members)]
    expected = np.mean(np.stack(masks, axis=0), axis=0) >= 0.5
    np.testing.assert_array_equal(_majority_vote(masks), expected)


@pytest.mark.parametrize(
    "arr",
    [
        np.array([[[0, 1]]], dtype=np.uint8),
        np.array([[[0, 1]]], dtype=np.int16),
        np.array([[[0.0, 1.0]]], dtype=np.float32),
        np.array([[[False, True]]]),
    ],
)
def test_validate_binary_mask_accepts_binary(arr):
    out = validate_binary_mask(arr)
    assert out.dtype == bool
    np.testing.assert_array_equal(out, arr.astype(bool))


@pytest.mark.parametrize(
    "arr, message",
    [
        (np.array([[[0, 2]]], dtype=np.uint8), "got values [0, 2]"),
        (np.array([[[-1, 1]]], dtype=np.int8), "got values [-1, 1]"),
        (np.array([[[0.0, 0.5]]], dtype=np.float32), "got values [0.0, 0.5]"),
        (np.array([[[0.0, np.nan]]], dtype=np.float32), "non-finite"),
    ],
)
def test_validate_binary_mask_rejects_non_binary_with_same_message(arr, message):
    with pytest.raises(tq.InputValidationError, match=message.replace("[", r"\[").replace("]", r"\]")):
        validate_binary_mask(arr)
