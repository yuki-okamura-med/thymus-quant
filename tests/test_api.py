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
    assert result.thymic_tissue_fraction is not None
    assert result.qc.status == "check"  # single-member ensemble QC unavailable


def test_unknown_method_detail_and_options_type_fail():
    seg = _seg()
    with pytest.raises(ValueError, match="unknown method"):
        tq.quantify(seg, method="bad")
    with pytest.raises(ValueError, match="unknown detail"):
        tq.quantify(seg, method="okamura", detail="verbose")
    with pytest.raises(ValueError, match="unknown method"):
        tq.quantify(seg, method="bad", options=tq.OkamuraOptions())


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


def _write_ct(path, offset):
    import nibabel as nib

    rng = np.random.default_rng(int(offset) + 100)
    ct = rng.normal(-1000.0, 5.0, size=(24, 24, 12)).astype(np.float32)
    ct[4:20, 2:12, 3:9] = rng.normal(-40.0 + offset, 10.0, size=(16, 10, 6))
    nib.save(nib.Nifti1Image(ct, np.diag([-0.7, 0.7, 2.5, 1.0])), str(path))
    return str(path)


@pytest.mark.filterwarnings("ignore:heuristic_trq")
def test_batch_frame_rows_can_be_matched_to_inputs_when_one_fails(tmp_path):
    seg = tq.load_segmentor("heuristic_trq")
    paths = [_write_ct(tmp_path / f"case{i}.nii.gz", 20 * i) for i in range(4)]
    paths.insert(1, str(tmp_path / "missing.nii.gz"))

    batch = tq.analyze_many(paths, method="okamura", segmentor=seg, on_error="record")
    frame = batch.to_frame()

    assert list(frame["input_index"]) == [0, 2, 3, 4]
    assert list(frame["input_source"]) == [paths[i] for i in (0, 2, 3, 4)]
    assert list(batch.errors_to_frame()["input_index"]) == [1]
    assert sorted(set(batch.members_to_frame()["input_index"])) == [0, 2, 3, 4]
    for _, row in frame.iterrows():
        alone = tq.analyze(paths[row["input_index"]], method="okamura", segmentor=seg)
        assert row["etv_ml"] == pytest.approx(alone.etv_ml)
        assert alone.to_record()["input_index"] is None
        assert alone.to_record()["input_source"] == paths[row["input_index"]]
