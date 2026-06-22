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
from ._version import __version__
from .exceptions import (
    ExperimentalWarning,
    InputValidationError,
    MissingGeometryError,
    NumericalError,
    QuantificationError,
    SegmentorConfigurationError,
    ThymusQuantError,
)
from .inputs import ImageContext
from .quantification.okamura import OkamuraOptions
from .results import (
    AnalysisResultBase,
    AnalysisResultOkamura,
    BatchAnalysisResultBase,
    BatchAnalysisResultOkamura,
    BatchErrorRecord,
    OkamuraMemberResult,
    OkamuraQC,
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
    "AnalysisResultOkamura",
    "BatchAnalysisResultBase",
    "BatchAnalysisResultOkamura",
    "BatchErrorRecord",
    "ExperimentalWarning",
    "ImageContext",
    "InputValidationError",
    "LoadedSegmentor",
    "MissingGeometryError",
    "NumericalError",
    "OkamuraMemberResult",
    "OkamuraOptions",
    "OkamuraQC",
    "QuantificationError",
    "ResultMeta",
    "SegmentationMember",
    "SegmentationResult",
    "SegmentorInfo",
    "SegmentorMember",
    "SegmentorConfigurationError",
    "ThymusQuantError",
    "analyze",
    "analyze_many",
    "list_methods",
    "list_segmentors",
    "load_segmentor",
    "quantify",
    "segment_trq",
    "__version__",
]
