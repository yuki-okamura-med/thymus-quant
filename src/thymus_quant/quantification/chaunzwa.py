from __future__ import annotations

"""Reserved configuration surface for future Chaunzwa-method support."""

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class ChaunzwaOptions:
    """Reserved options for a future Chaunzwa quantification method.

    The Chaunzwa method is not part of the current release. The option names are
    retained to make a future implementation easier to add without redesigning
    the public configuration surface.
    """

    gmm_n_components: int | None = None
    gmm_max_components: int = 4
    model_selection: Literal["bic", "aic", "fixed", "auto"] = "auto"
    histogram_bins: int = 256
    max_iter: int = 200
    tol: float = 1e-4
    random_state: int | None = 0
    aadipose_hu: float = -110.0
    athymic_hu: float = 80.0
    bayesian_delta_hu: float = 1.0
    tissue_fraction_definition: Literal["posterior_nonadipose", "atrq_linear"] = "posterior_nonadipose"
    adipose_policy: Literal["below_aadipose", "lowest_component"] = "below_aadipose"
    exclude_mu_hu_below: float | None = -300.0
    exclude_mu_hu_above: float | None = 300.0
    include_posterior_maps: bool = False
