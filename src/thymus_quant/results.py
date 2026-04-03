from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeAlias

if TYPE_CHECKING:
    import pandas as pd
    from .segmentors import SegmentationResult

MethodName: TypeAlias = Literal["okamura", "chaunzwa"]
DetailLevel: TypeAlias = Literal["summary", "full"]
QCStatus: TypeAlias = Literal["ok", "check", "not_available"]
OnError: TypeAlias = Literal["raise", "record", "skip"]


def _df(records: list[dict[str, Any]]):
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("pandas is required for to_frame()") from e
    return pd.DataFrame.from_records(records)


@dataclass(slots=True)
class ResultMeta:
    study_id: str | None = None
    method: MethodName | None = None
    detail: DetailLevel = "summary"
    input_source: str | None = None
    segmentor_name: str | None = None
    segmentor_repo_id: str | None = None
    segmentor_revision: str | None = None
    segmentor_members: tuple[str, ...] = ()
    library_version: str | None = None


@dataclass(slots=True)
class BatchErrorRecord:
    input_source: str | None = None
    study_id: str | None = None
    error_type: str | None = None
    message: str | None = None


@dataclass(slots=True)
class AnalysisResultBase(ABC):
    study_id: str | None = None
    method: MethodName | None = None
    meta: ResultMeta | None = None
    segmentation: SegmentationResult | None = None

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def to_record(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def to_frame(self) -> "pd.DataFrame":
        raise NotImplementedError


@dataclass(slots=True)
class OkamuraQC:
    status: QCStatus = "not_available"
    paper_criteria_met: bool | None = None
    flags: Sequence[str] = field(default_factory=tuple)
    mean_pairwise_jsd: float | None = None
    mean_pairwise_dsc: float | None = None
    hu_variance: float | None = None
    valid_member_count: int = 0


@dataclass(slots=True)
class OkamuraMemberResult:
    member_id: str
    valid: bool = True
    trq_hu_mode: float | None = None
    trq_volume_ml: float | None = None
    etv_ml: float | None = None
    second_peak_ratio: float | None = None
    trq_mask: Any | None = None
    airway_mask: Any | None = None


@dataclass(slots=True)
class AnalysisResultOkamura(AnalysisResultBase):
    trq_hu_mode: float | None = None
    trq_volume_ml: float | None = None
    etv_ml: float | None = None
    qc: OkamuraQC | None = None
    members: Sequence[OkamuraMemberResult] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "study_id": self.study_id,
            "method": self.method,
            "meta": None if self.meta is None else asdict(self.meta),
            "summary": {
                "trq_hu_mode": self.trq_hu_mode,
                "trq_volume_ml": self.trq_volume_ml,
                "etv_ml": self.etv_ml,
            },
            "qc": None if self.qc is None else asdict(self.qc),
            "members": [
                {
                    "member_id": m.member_id,
                    "valid": m.valid,
                    "trq_hu_mode": m.trq_hu_mode,
                    "trq_volume_ml": m.trq_volume_ml,
                    "etv_ml": m.etv_ml,
                    "second_peak_ratio": m.second_peak_ratio,
                }
                for m in self.members
            ],
        }
        return d

    def to_record(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "method": self.method,
            "trq_hu_mode": self.trq_hu_mode,
            "trq_volume_ml": self.trq_volume_ml,
            "etv_ml": self.etv_ml,
            "qc_status": None if self.qc is None else self.qc.status,
            "qc_paper_criteria_met": None if self.qc is None else self.qc.paper_criteria_met,
            "qc_mean_pairwise_jsd": None if self.qc is None else self.qc.mean_pairwise_jsd,
            "qc_mean_pairwise_dsc": None if self.qc is None else self.qc.mean_pairwise_dsc,
            "qc_hu_variance": None if self.qc is None else self.qc.hu_variance,
            "qc_valid_member_count": None if self.qc is None else self.qc.valid_member_count,
        }

    def to_frame(self) -> "pd.DataFrame":
        return _df([self.to_record()])


@dataclass(slots=True)
class GMMComponent:
    component_id: int
    weight: float | None = None
    mu_hu: float | None = None
    sigma_hu: float | None = None
    posterior_mass_vox: float | None = None
    posterior_volume_ml: float | None = None


@dataclass(slots=True)
class GaussianMixtureFit:
    n_components: int
    converged: bool | None = None
    n_iter: int | None = None
    model_selection: str | None = None
    aic: float | None = None
    bic: float | None = None
    components: Sequence[GMMComponent] = field(default_factory=tuple)


@dataclass(slots=True)
class PosteriorSummary:
    component_posterior_mass_vox: dict[int, float] = field(default_factory=dict)
    component_posterior_volume_ml: dict[int, float] = field(default_factory=dict)
    tissue_posterior_mass_vox: dict[str, float] = field(default_factory=dict)
    tissue_posterior_volume_ml: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PosteriorMaps:
    component_posteriors: dict[int, Any] = field(default_factory=dict)
    tissue_posteriors: dict[str, Any] = field(default_factory=dict)
    hard_component_labels: Any | None = None


@dataclass(slots=True)
class AnalysisResultChaunzwa(AnalysisResultBase):
    atrq_hu: float | None = None
    trq_volume_ml: float | None = None
    etv_ml: float | None = None
    ptt: float | None = None
    gmm: GaussianMixtureFit | None = None
    posterior: PosteriorSummary | None = None
    posterior_maps: PosteriorMaps | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "method": self.method,
            "meta": None if self.meta is None else asdict(self.meta),
            "summary": {
                "atrq_hu": self.atrq_hu,
                "trq_volume_ml": self.trq_volume_ml,
                "etv_ml": self.etv_ml,
                "ptt": self.ptt,
            },
            "gmm": None if self.gmm is None else asdict(self.gmm),
            "posterior": None if self.posterior is None else asdict(self.posterior),
            "posterior_maps": None if self.posterior_maps is None else {
                "component_ids": sorted(self.posterior_maps.component_posteriors.keys()),
                "tissue_keys": sorted(self.posterior_maps.tissue_posteriors.keys()),
                "has_hard_labels": self.posterior_maps.hard_component_labels is not None,
            },
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "method": self.method,
            "atrq_hu": self.atrq_hu,
            "trq_volume_ml": self.trq_volume_ml,
            "etv_ml": self.etv_ml,
            "ptt": self.ptt,
            "gmm_n_components": None if self.gmm is None else self.gmm.n_components,
            "gmm_converged": None if self.gmm is None else self.gmm.converged,
            "gmm_bic": None if self.gmm is None else self.gmm.bic,
            "gmm_aic": None if self.gmm is None else self.gmm.aic,
        }

    def to_frame(self) -> "pd.DataFrame":
        return _df([self.to_record()])


@dataclass(slots=True)
class BatchAnalysisResultBase(ABC):
    method: MethodName
    errors: Sequence[BatchErrorRecord] = field(default_factory=tuple)

    @abstractmethod
    def to_frame(self) -> "pd.DataFrame":
        raise NotImplementedError


@dataclass(slots=True)
class BatchAnalysisResultOkamura(BatchAnalysisResultBase):
    results: Sequence[AnalysisResultOkamura] = field(default_factory=tuple)

    def to_frame(self) -> "pd.DataFrame":
        return _df([r.to_record() for r in self.results])

    def members_to_frame(self) -> "pd.DataFrame":
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
                        "second_peak_ratio": m.second_peak_ratio,
                    }
                )
        return _df(rows)


@dataclass(slots=True)
class BatchAnalysisResultChaunzwa(BatchAnalysisResultBase):
    results: Sequence[AnalysisResultChaunzwa] = field(default_factory=tuple)

    def to_frame(self) -> "pd.DataFrame":
        return _df([r.to_record() for r in self.results])

    def components_to_frame(self) -> "pd.DataFrame":
        rows = []
        for r in self.results:
            if r.gmm is None:
                continue
            for c in r.gmm.components:
                rows.append(
                    {
                        "study_id": r.study_id,
                        "component_id": c.component_id,
                        "weight": c.weight,
                        "mu_hu": c.mu_hu,
                        "sigma_hu": c.sigma_hu,
                        "posterior_mass_vox": c.posterior_mass_vox,
                        "posterior_volume_ml": c.posterior_volume_ml,
                    }
                )
        return _df(rows)
