from __future__ import annotations

"""Result container classes for :mod:`thymus_quant`.

This module defines:
- study-level result objects
- batch-level result objects
- lightweight metadata and QC records
- convenience exporters (`to_dict`, `to_record`, `to_frame`)
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import math
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeAlias

if TYPE_CHECKING:
    import pandas as pd

    from .segmentors import SegmentationResult

MethodName: TypeAlias = Literal["okamura"]
DetailLevel: TypeAlias = Literal["summary", "full"]
ResultStatus: TypeAlias = Literal["ok", "check", "failed", "not_available"]
QCStatus: TypeAlias = ResultStatus
OnError: TypeAlias = Literal["raise", "record", "skip"]


def _json_safe(x: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON-safe serialization."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _df(records: list[dict[str, Any]]):
    """Build a pandas DataFrame from flat records.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover - optional dependency path
        raise ImportError("pandas is required for to_frame()") from e
    return pd.DataFrame.from_records(records)


@dataclass(slots=True)
class ResultMeta:
    """Common metadata attached to a study-level analysis result."""

    study_id: str | None = None
    method: MethodName | None = None
    detail: DetailLevel = "summary"
    input_source: str | None = None
    segmentor_name: str | None = None
    segmentor_repo_id: str | None = None
    segmentor_requested_revision: str | None = None
    segmentor_revision: str | None = None
    segmentor_members: tuple[str, ...] = ()
    segmentor_weight_source: str | None = None
    segmentor_local_source: str | None = None
    preprocessing_version: str | None = None
    image_shape: tuple[int, int, int] | None = None
    spacing_mm: tuple[float, float, float] | None = None
    orientation: tuple[str, str, str] | None = None
    method_version: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    experimental: bool = False
    library_version: str | None = None
    model_orientation: str | None = None
    orientation_status: str | None = None
    reoriented_for_model: bool | None = None
    geometry: dict[str, Any] | None = None
    inplane_shape: tuple[int, int] | None = None
    inplane_is_512: bool | None = None
    network_pixel_mm: tuple[float, float] | None = None


@dataclass(slots=True)
class BatchErrorRecord:
    """Per-item error record returned by ``analyze_many(..., on_error='record')``."""

    input_source: str | None = None
    study_id: str | None = None
    error_type: str | None = None
    message: str | None = None
    input_index: int | None = None


@dataclass(slots=True)
class AnalysisResultBase(ABC):
    """Abstract base class shared by all study-level result objects."""

    study_id: str | None = None
    method: MethodName | None = None
    meta: ResultMeta | None = None
    segmentation: SegmentationResult | None = None
    status: ResultStatus = "not_available"
    flags: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    failure_reason: str | None = None

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return nested JSON-friendly representation."""
        raise NotImplementedError

    @abstractmethod
    def to_record(self) -> dict[str, Any]:
        """Return flat one-study record for tabular export."""
        raise NotImplementedError

    @abstractmethod
    def to_frame(self) -> "pd.DataFrame":
        """Return one-row pandas DataFrame summary."""
        raise NotImplementedError


@dataclass(slots=True)
class OkamuraQC:
    """Quality-control summary for the Okamura method."""

    status: QCStatus = "not_available"
    paper_criteria_met: bool | None = None
    flags: Sequence[str] = field(default_factory=tuple)
    mean_pairwise_jsd: float | None = None
    mean_pairwise_dsc: float | None = None
    hu_variance: float | None = None
    expected_member_count: int | None = None
    supplied_member_count: int = 0
    computed_member_count: int = 0
    valid_member_count: int = 0


@dataclass(slots=True)
class OkamuraMemberResult:
    """Per-member quantitative result for ensemble-based Okamura analysis.

    Notes
    -----
    `thymic_tissue_fraction` is stored as a fraction in the range 0-1.
    """

    member_id: str
    valid: bool = True
    trq_hu_mode: float | None = None
    trq_volume_ml: float | None = None
    etv_ml: float | None = None
    thymic_tissue_fraction: float | None = None
    etv_fraction_adjusted: float | None = None
    atrq_below_aadipose: bool | None = None
    second_peak_ratio: float | None = None
    flags: Sequence[str] = field(default_factory=tuple)
    failure_reason: str | None = None
    trq_mask: Any | None = None
    airway_mask: Any | None = None


@dataclass(slots=True)
class AnalysisResultOkamura(AnalysisResultBase):
    """Study-level output for the Okamura method.

    Fields with `_members` suffix contain per-member values in the same order
    as `members`.

    Notes
    -----
    `thymic_tissue_fraction` and `thymic_tissue_fraction_members` are fractions
    in the range 0-1.
    """

    trq_hu_mode: float | None = None
    trq_volume_ml: float | None = None
    etv_ml: float | None = None
    thymic_tissue_fraction: float | None = None
    atrq_below_aadipose_any: bool | None = None
    trq_hu_mode_members: Sequence[float] = field(default_factory=tuple)
    trq_volume_ml_members: Sequence[float] = field(default_factory=tuple)
    etv_ml_members: Sequence[float] = field(default_factory=tuple)
    thymic_tissue_fraction_members: Sequence[float] = field(default_factory=tuple)
    atrq_below_aadipose_members: Sequence[bool] = field(default_factory=tuple)
    qc: OkamuraQC | None = None
    members: Sequence[OkamuraMemberResult] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return nested representation with optional member-level payload."""
        return _json_safe({
            "study_id": self.study_id,
            "method": self.method,
            "status": self.status,
            "flags": list(self.flags),
            "warnings": list(self.warnings),
            "failure_reason": self.failure_reason,
            "meta": None if self.meta is None else asdict(self.meta),
            "summary": {
                "trq_hu_mode": self.trq_hu_mode,
                "trq_volume_ml": self.trq_volume_ml,
                "etv_ml": self.etv_ml,
                "thymic_tissue_fraction": self.thymic_tissue_fraction,
                "atrq_below_aadipose_any": self.atrq_below_aadipose_any,
            },
            "member_values": {
                "trq_hu_mode_members": list(self.trq_hu_mode_members),
                "trq_volume_ml_members": list(self.trq_volume_ml_members),
                "etv_ml_members": list(self.etv_ml_members),
                "thymic_tissue_fraction_members": list(self.thymic_tissue_fraction_members),
                "atrq_below_aadipose_members": list(self.atrq_below_aadipose_members),
            },
            "qc": None if self.qc is None else asdict(self.qc),
            "members": [
                {
                    "member_id": m.member_id,
                    "valid": m.valid,
                    "trq_hu_mode": m.trq_hu_mode,
                    "trq_volume_ml": m.trq_volume_ml,
                    "etv_ml": m.etv_ml,
                    "thymic_tissue_fraction": m.thymic_tissue_fraction,
                    "etv_fraction_adjusted": m.etv_fraction_adjusted,
                    "atrq_below_aadipose": m.atrq_below_aadipose,
                    "second_peak_ratio": m.second_peak_ratio,
                    "flags": list(m.flags),
                    "failure_reason": m.failure_reason,
                }
                for m in self.members
            ],
        })

    def to_record(self) -> dict[str, Any]:
        """Return flat study-level table record."""
        meta = self.meta
        orientation = None if meta is None or meta.orientation is None else "".join(str(c) for c in meta.orientation)
        return {
            "study_id": self.study_id,
            "method": self.method,
            "status": self.status,
            "flags": tuple(self.flags),
            "failure_reason": self.failure_reason,
            "trq_hu_mode": self.trq_hu_mode,
            "trq_volume_ml": self.trq_volume_ml,
            "etv_ml": self.etv_ml,
            "thymic_tissue_fraction": self.thymic_tissue_fraction,
            "atrq_below_aadipose_any": self.atrq_below_aadipose_any,
            "trq_hu_mode_members": tuple(self.trq_hu_mode_members),
            "trq_volume_ml_members": tuple(self.trq_volume_ml_members),
            "etv_ml_members": tuple(self.etv_ml_members),
            "thymic_tissue_fraction_members": tuple(self.thymic_tissue_fraction_members),
            "atrq_below_aadipose_members": tuple(self.atrq_below_aadipose_members),
            "qc_status": None if self.qc is None else self.qc.status,
            "qc_paper_criteria_met": None if self.qc is None else self.qc.paper_criteria_met,
            "qc_mean_pairwise_jsd": None if self.qc is None else self.qc.mean_pairwise_jsd,
            "qc_mean_pairwise_dsc": None if self.qc is None else self.qc.mean_pairwise_dsc,
            "qc_hu_variance": None if self.qc is None else self.qc.hu_variance,
            "qc_valid_member_count": None if self.qc is None else self.qc.valid_member_count,
            "input_orientation": orientation,
            "orientation_status": None if meta is None else meta.orientation_status,
            "reoriented_for_model": None if meta is None else meta.reoriented_for_model,
            "inplane_shape": None if meta is None or meta.inplane_shape is None else "x".join(str(v) for v in meta.inplane_shape),
            "inplane_is_512": None if meta is None else meta.inplane_is_512,
            "network_pixel_mm": None if meta is None or meta.network_pixel_mm is None else tuple(meta.network_pixel_mm),
        }

    def to_frame(self) -> "pd.DataFrame":
        """Return one-row DataFrame."""
        return _df([self.to_record()])


@dataclass(slots=True)
class BatchAnalysisResultBase(ABC):
    """Abstract base container for batch analysis outputs."""

    method: MethodName
    errors: Sequence[BatchErrorRecord] = field(default_factory=tuple)
    n_requested: int = 0

    @abstractmethod
    def to_frame(self) -> "pd.DataFrame":
        """Return study-level summary table."""
        raise NotImplementedError

    @property
    def n_failed(self) -> int:
        return len(self.errors)

    def errors_to_frame(self) -> "pd.DataFrame":
        """Return one row per recorded batch error."""
        return _df([asdict(e) for e in self.errors])


@dataclass(slots=True)
class BatchAnalysisResultOkamura(BatchAnalysisResultBase):
    """Batch output container for Okamura analyses."""

    results: Sequence[AnalysisResultOkamura] = field(default_factory=tuple)

    @property
    def n_succeeded(self) -> int:
        return len(self.results)

    def to_frame(self) -> "pd.DataFrame":
        """Return one row per study."""
        return _df([r.to_record() for r in self.results])

    def members_to_frame(self) -> "pd.DataFrame":
        """Return one row per study-member pair."""
        rows = []
        for r in self.results:
            for m in r.members:
                rows.append(
                    {
                        "study_id": r.study_id,
                        "member_id": m.member_id,
                        "valid": m.valid,
                        "trq_hu_mode": m.trq_hu_mode,
                        "trq_volume_ml": m.trq_volume_ml,
                        "etv_ml": m.etv_ml,
                        "thymic_tissue_fraction": m.thymic_tissue_fraction,
                        "etv_fraction_adjusted": m.etv_fraction_adjusted,
                        "atrq_below_aadipose": m.atrq_below_aadipose,
                        "second_peak_ratio": m.second_peak_ratio,
                    }
                )
        return _df(rows)
