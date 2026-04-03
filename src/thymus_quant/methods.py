from __future__ import annotations

"""Method-specific option objects for thymus quantification.

The goal is to keep the public function signatures small while still allowing
method-specific settings to be explicit and typed.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class OkamuraOptions:
    """Configuration for the Okamura quantification method.

    Defaults are chosen to mirror the published method as closely as possible:

    - five-member ensemble outputs are quantified separately;
    - representative TRQ HU is the KDE mode;
    - a segmentation member is marked invalid when the second KDE peak exceeds
      half the height of the first peak;
    - ETV is computed using ``aadipose_hu=-110`` and ``athymic_hu=80``;
    - study-level QC uses mean pairwise JS divergence, mean pairwise DSC, and
      HU variance thresholds.

    Notes
    -----
    `invalidate_if_any_member_invalid=True` matches the paper-level analysis
    flow, where a study was excluded if any of the five member segmentations was
    invalid.
    """

    aadipose_hu: float = -110.0
    athymic_hu: float = 80.0
    second_peak_ratio_threshold: float = 0.5
    js_divergence_threshold: float = 0.1
    pairwise_dsc_threshold: float = 0.7
    hu_variance_threshold: float = 20.0
    invalidate_if_any_member_invalid: bool = True
    apply_qc: bool = True


@dataclass(slots=True)
class ChaunzwaOptions:
    """Configuration for the Chaunzwa quantification method.

    Defaults are chosen to reflect the preprint description:

    - HU values inside the contoured thymic region are binned into 50 equal
      width histogram bins over the full finite dynamic range;
    - a Gaussian mixture model (GMM) is fit to the empirical distribution;
    - ``A_TRQ`` is estimated as the weighted sum of GMM component means;
    - ``ETV`` and ``pTT`` are derived from ``A_TRQ``, ``A_adipose`` and
      ``A_thymic``.

    Parameters such as ``gmm_n_components`` and ``model_selection`` are exposed
    because the paper excerpt specifies a K-component GMM but does not fully pin
    down the selection strategy in the way the Okamura method pins down QC.
    """

    aadipose_hu: float = -110.0
    athymic_hu: float = 80.0
    histogram_bins: int = 50
    gmm_n_components: int | None = None
    gmm_max_components: int | None = None
    model_selection: Literal["auto", "fixed", "bic", "aic"] = "auto"
    include_posterior_maps: bool = False
    include_histogram: bool = False
