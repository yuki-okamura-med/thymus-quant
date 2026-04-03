"""Public API for :mod:`thymus_quant`.

Recommended user import::

    import thymus_quant as thyq

This package exposes a small public API surface:
`load_segmentor`, `segment_trq`, `quantify`, `analyze`, and `analyze_many`.
"""

from .api import (
    analyze,
    analyze_many,
    list_methods,
    list_segmentors,
    load_segmentor,
    quantify,
    segment_trq,
)
from .methods import ChaunzwaOptions, OkamuraOptions
from .results import (
    AnalysisResultBase,
    AnalysisResultChaunzwa,
    AnalysisResultOkamura,
    BatchAnalysisResultBase,
    BatchAnalysisResultChaunzwa,
    BatchAnalysisResultOkamura,
    BatchErrorRecord,
    GaussianMixtureFit,
    GMMComponent,
    OkamuraMemberResult,
    OkamuraQC,
    PosteriorMaps,
    PosteriorSummary,
    ResultMeta,
)
from .segmentors import (
    LoadedSegmentor,
    SegmentationMember,
    SegmentationResult,
    SegmentorInfo,
    SegmentorMember,
)

__all__ = [
    "AnalysisResultBase",
    "AnalysisResultChaunzwa",
    "AnalysisResultOkamura",
    "BatchAnalysisResultBase",
    "BatchAnalysisResultChaunzwa",
    "BatchAnalysisResultOkamura",
    "BatchErrorRecord",
    "ChaunzwaOptions",
    "GaussianMixtureFit",
    "GMMComponent",
    "LoadedSegmentor",
    "OkamuraMemberResult",
    "OkamuraOptions",
    "OkamuraQC",
    "PosteriorMaps",
    "PosteriorSummary",
    "ResultMeta",
    "SegmentationMember",
    "SegmentationResult",
    "SegmentorInfo",
    "SegmentorMember",
    "analyze",
    "analyze_many",
    "list_methods",
    "list_segmentors",
    "load_segmentor",
    "quantify",
    "segment_trq",
]

__version__ = "0.1.0a0"
