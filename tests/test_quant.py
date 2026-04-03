import numpy as np
import nibabel as nib

import thymus_quant as tq


def _nii(arr):
    return nib.Nifti1Image(arr.astype(np.float32), np.eye(4))


def test_analyze_basic():
    ct = np.full((8, 8, 8), -110.0, dtype=np.float32)
    m1 = np.zeros((8, 8, 8), dtype=np.uint8)
    m2 = np.zeros((8, 8, 8), dtype=np.uint8)
    m1[2:6, 2:6, 2:6] = 1
    m2[2:6, 2:6, 2:6] = 1
    ct[m1 > 0] = -10.0

    result = tq.analyze(_nii(ct), trq_mask=[_nii(m1), _nii(m2)], protocol="okamura2025_plus_ptt")

    assert result.summary.trq_volume_ml > 0
    assert 0 <= result.summary.ptt_fraction <= 1
    assert result.summary.etv_ml > 0
    assert result.qc.status in {"pass", "review", "fail"}
