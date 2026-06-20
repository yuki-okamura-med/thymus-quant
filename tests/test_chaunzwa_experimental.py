import pathlib

import numpy as np
import pytest

import thymus_quant as tq


def test_chaunzwa_api_warns_and_marks_experimental():
    ct = np.linspace(-120, 40, 125).reshape(5, 5, 5)
    mask = np.ones_like(ct, dtype=bool)
    seg = tq.SegmentationResult.from_mask(trq_mask=mask, ct_hu=ct, spacing_mm=(1, 1, 1), study_id="c")

    with pytest.warns(tq.ExperimentalWarning):
        result = tq.quantify(seg, method="chaunzwa", options=tq.ChaunzwaOptions(gmm_n_components=1))

    assert result.status == "check"
    assert "experimental" in result.flags
    assert result.meta.experimental is True
    assert result.to_dict()["summary"]["ptt"] is not None


def test_readme_mentions_chaunzwa_as_experimental():
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "Experimental: Chaunzwa" in readme
    assert "primary endpoint" in readme
