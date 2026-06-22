import pathlib

import numpy as np
import pytest

import thymus_quant as tq


def test_chaunzwa_method_is_not_released_but_options_remain():
    ct = np.zeros((3, 3, 3), dtype=float)
    mask = np.ones_like(ct, dtype=bool)
    seg = tq.SegmentationResult.from_mask(trq_mask=mask, ct_hu=ct, spacing_mm=(1, 1, 1), study_id="c")

    options = tq.ChaunzwaOptions(gmm_n_components=1)
    assert options.gmm_n_components == 1
    assert tq.list_methods() == ("okamura",)
    with pytest.raises(ValueError, match="unknown method"):
        tq.quantify(seg, method="chaunzwa", options=options)


def test_readme_mentions_chaunzwa_as_coming_soon_only():
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "Coming soon: Methods of Chaunzwa et al." in readme
    assert "planned for a future release" in readme
