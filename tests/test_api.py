import numpy as np
import pytest

import thymus_quant as tq


def _seg():
    ct = np.full((5, 5, 5), -20.0)
    mask = np.zeros_like(ct, dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    return tq.SegmentationResult.from_mask(trq_mask=mask, ct_hu=ct, spacing_mm=(1, 1, 1), study_id="s1")


def test_public_quantify_basic_okamura():
    result = tq.quantify(_seg(), method="okamura")
    assert result.method == "okamura"
    assert result.ptt is not None
    assert result.qc.status == "check"  # single-member ensemble QC unavailable


def test_unknown_method_detail_and_options_type_fail():
    seg = _seg()
    with pytest.raises(ValueError, match="unknown method"):
        tq.quantify(seg, method="bad")
    with pytest.raises(ValueError, match="unknown detail"):
        tq.quantify(seg, method="okamura", detail="verbose")
    with pytest.raises(TypeError, match="requires ChaunzwaOptions"):
        tq.quantify(seg, method="chaunzwa", options=tq.OkamuraOptions())


def test_analyze_requires_explicit_segmentor(tmp_path):
    with pytest.raises(Exception, match="segmentor"):
        tq.analyze(tmp_path / "missing.nii.gz", method="okamura")


def test_analyze_many_validates_common_options_before_processing():
    with pytest.raises(ValueError, match="on_error"):
        tq.analyze_many([], method="okamura", segmentor=tq.load_segmentor("heuristic_trq"), on_error="bad")


def test_batch_ordering_and_errors_preserved():
    seg = tq.load_segmentor("heuristic_trq")
    imgs = []
    # invalid path errors are recorded in input order without repeating common config validation.
    batch = tq.analyze_many(["missing-a.nii.gz", "missing-b.nii.gz"], method="okamura", segmentor=seg, on_error="record")
    assert batch.n_requested == 2
    assert batch.n_succeeded == 0
    assert batch.n_failed == 2
    assert [e.input_index for e in batch.errors] == [0, 1]
