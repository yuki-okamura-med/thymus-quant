from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantSummary:
    trq_hu_mode: float
    trq_volume_ml: float
    etv_ml: float
    ptt_fraction: float
    ptt_percent: float
    n_members: int
    n_valid_members: int


@dataclass
class QCReport:
    status: str
    passed: bool
    mean_pairwise_jsd: float | None
    mean_pairwise_dsc: float | None
    hu_variance: float | None
    fail_reasons: list[str] = field(default_factory=list)


@dataclass
class ResultMeta:
    study_id: str | None
    segmenter: str
    protocol: str
    library_version: str


@dataclass
class MemberResult:
    member_id: str
    checkpoint_id: str | None
    metrics: dict[str, float]
    valid: bool


@dataclass
class AnalysisResult:
    summary: QuantSummary
    qc: QCReport
    members: list[MemberResult]
    meta: ResultMeta

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": vars(self.summary),
            "qc": {**vars(self.qc), "fail_reasons": list(self.qc.fail_reasons)},
            "members": [
                {
                    "member_id": m.member_id,
                    "checkpoint_id": m.checkpoint_id,
                    "metrics": dict(m.metrics),
                    "valid": m.valid,
                }
                for m in self.members
            ],
            "meta": vars(self.meta),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "study_id": self.meta.study_id,
            "segmenter": self.meta.segmenter,
            "protocol": self.meta.protocol,
            "qc_status": self.qc.status,
            "qc_passed": self.qc.passed,
            "trq_hu_mode": self.summary.trq_hu_mode,
            "trq_volume_ml": self.summary.trq_volume_ml,
            "etv_ml": self.summary.etv_ml,
            "ptt_fraction": self.summary.ptt_fraction,
            "ptt_percent": self.summary.ptt_percent,
            "qc_mean_pairwise_jsd": self.qc.mean_pairwise_jsd,
            "qc_mean_pairwise_dsc": self.qc.mean_pairwise_dsc,
            "qc_hu_variance": self.qc.hu_variance,
            "n_members": self.summary.n_members,
            "n_valid_members": self.summary.n_valid_members,
        }

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_frame()") from e
        return pd.DataFrame([self.to_record()])

    to_dataframe = to_frame
