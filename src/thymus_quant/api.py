from __future__ import annotations

from .core import analyze, quantify


class Analyzer:
    def __init__(self, segmenter: str = "okamura2025_ensemble_v1", device: str = "auto"):
        self.segmenter = segmenter
        self.device = device

    @classmethod
    def from_pretrained(cls, segmenter: str = "okamura2025_ensemble_v1", *, device: str = "auto") -> "Analyzer":
        return cls(segmenter=segmenter, device=device)

    def analyze(self, image, *, trq_mask, study_id: str | None = None, protocol: str = "okamura2025", detail: str = "summary"):
        _ = detail
        return analyze(image, trq_mask=trq_mask, study_id=study_id, segmenter=self.segmenter, protocol=protocol)

    def analyze_many(self, items, *, protocol: str = "okamura2025", detail: str = "summary", on_error: str = "record"):
        _ = detail
        rows = []
        for item in items:
            try:
                if isinstance(item, dict):
                    res = self.analyze(item["ct"], trq_mask=item["trq_mask"], study_id=item.get("study_id"), protocol=protocol)
                else:
                    raise ValueError("analyze_many expects dict items with ct/trq_mask")
                rows.append(res)
            except Exception:
                if on_error == "raise":
                    raise
                if on_error == "record":
                    rows.append(None)
                # skip for on_error == "skip"
        return BatchAnalysisResult(rows)


class BatchAnalysisResult:
    def __init__(self, results):
        self.results = results

    def to_frame(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_frame()") from e

        records = []
        for r in self.results:
            if r is None:
                continue
            records.append(r.to_record())
        return pd.DataFrame.from_records(records)
