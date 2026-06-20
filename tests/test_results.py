import json

import numpy as np

import thymus_quant as tq


def _result():
    ct = np.full((4, 4, 4), -15.0)
    mask = np.ones_like(ct, dtype=bool)
    seg = tq.SegmentationResult.from_mask(
        trq_mask=mask,
        ct_hu=ct,
        spacing_mm=(1, 1, 1),
        study_id="case-001",
        source="case.nii.gz",
    )
    return tq.quantify(seg, method="okamura")


def test_result_status_metadata_and_ptt_units():
    result = _result()
    d = result.to_dict()
    assert d["status"] == "check"
    assert d["meta"]["library_version"] == tq.__version__
    assert d["meta"]["spacing_mm"] == [1.0, 1.0, 1.0]
    assert result.ptt == 50.0


def test_json_serialization_has_no_nan_or_infinity():
    result = _result()
    text = json.dumps(result.to_dict(), allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text


def test_to_record_and_frame():
    result = _result()
    rec = result.to_record()
    assert rec["ptt"] == 50.0
    frame = result.to_frame()
    assert list(frame["study_id"]) == ["case-001"]


def test_batch_errors_to_frame():
    batch = tq.analyze_many(["missing.nii.gz"], method="okamura", segmentor=tq.load_segmentor("heuristic_trq"), on_error="record")
    assert batch.n_requested == 1
    assert batch.n_failed == 1
    assert list(batch.errors_to_frame()["input_index"]) == [0]
