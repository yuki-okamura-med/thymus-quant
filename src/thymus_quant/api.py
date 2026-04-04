from __future__ import annotations

"""Top-level public API for :mod:`thymus_quant`.

Design intent:
- `api.py` keeps public entry points and orchestration.
- method-specific quantification logic lives under `thymus_quant.quantification`.
"""

import os
import threading
from pathlib import Path
from typing import Literal, Sequence, overload

import nibabel as nib
import numpy as np

from .quantification.chaunzwa import ChaunzwaOptions, quantify_chaunzwa
from .quantification.okamura import OkamuraOptions, quantify_okamura
from .results import (
    AnalysisResultBase,
    AnalysisResultChaunzwa,
    AnalysisResultOkamura,
    BatchAnalysisResultBase,
    BatchAnalysisResultChaunzwa,
    BatchAnalysisResultOkamura,
    BatchErrorRecord,
    DetailLevel,
    MethodName,
    OnError,
    ResultMeta,
)
from .segmentors import ImageInput, LoadedSegmentor, SegmentationResult, SegmentorInfo, SegmentorMember


_SEGMENTOR_CACHE: dict[tuple, LoadedSegmentor] = {}
_SEGMENTOR_CACHE_LOCK = threading.Lock()


def list_methods() -> tuple[MethodName, ...]:
    """Return supported quantification method names."""
    return ("okamura", "chaunzwa")


def list_segmentors() -> dict[str, SegmentorInfo]:
    """Return built-in segmentor registry."""
    trqseg_v1_fold_members = tuple(f"fold-{i}" for i in range(5))
    return {
        "trqseg_v1": SegmentorInfo(
            name="trqseg_v1",
            repo_id="yuki-okamura-hf/TRQseg-v1",
            architecture="deeplabv3_resnet50",
            is_ensemble=True,
            available_members=trqseg_v1_fold_members,
            selected_members=trqseg_v1_fold_members,
        ),
        "heuristic_trq": SegmentorInfo(
            name="heuristic_trq",
            repo_id=None,
            architecture="heuristic",
            is_ensemble=False,
            available_members=("single",),
            selected_members=("single",),
        ),
    }


def _resolve_members(members: Sequence[int | str] | None, default: Sequence[str]) -> tuple[str, ...]:
    """Normalize member selector into fold-name tuple."""
    if members is None:
        return tuple(default)
    out = []
    for m in members:
        if isinstance(m, int):
            out.append(f"fold-{m}")
        else:
            s = str(m)
            out.append(s if s.startswith("fold-") else f"fold-{s}")
    return tuple(out)


def _find_local_trqseg_v1_repo() -> str | None:
    """Find local mirror path for TRQseg-v1 if available."""
    env_path = os.environ.get("THYQ_TRQSEG_V1_LOCAL_REPO")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        Path("/mnt/w/repos/TRQseg-v1"),
        Path("/mnt/w/repos/trqseg-v1"),
    ])
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)
    return None


def _segmentor_cache_key(
    info: SegmentorInfo,
    requested_members: tuple[str, ...],
    local_files_only: bool,
    device: str,
) -> tuple:
    """Build deterministic key for loaded-segmentor cache."""
    return (
        info.name,
        info.repo_id,
        info.resolved_revision,
        requested_members,
        bool(local_files_only),
        info.cache_dir,
        device,
    )


def load_segmentor(
    segmentor: str | None = None,
    *,
    revision: str | None = None,
    ensemble: bool | Literal["auto"] = "auto",
    members: Sequence[int | str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
    device: str = "auto",
) -> LoadedSegmentor:
    """Resolve and return a (cached) loaded segmentor handle.

    `segmentor` must be explicitly specified.
    """
    if segmentor is None:
        raise ValueError(
            "segmentor must be specified explicitly (e.g. 'trqseg_v1'). "
            f"Available aliases: {', '.join(sorted(list_segmentors().keys()))}"
        )

    reg = list_segmentors()
    info = reg.get(segmentor)
    if info is None:
        info = SegmentorInfo(name=segmentor, repo_id=segmentor, architecture="unknown", is_ensemble=True)

    default_members = info.available_members or tuple(f"fold-{i}" for i in range(5))
    requested = _resolve_members(members, default_members)
    if ensemble is False and requested:
        requested = (requested[0],)

    info = SegmentorInfo(
        name=info.name,
        repo_id=info.repo_id,
        requested_revision=revision,
        resolved_revision=revision,
        architecture=info.architecture,
        is_ensemble=(len(requested) > 1),
        available_members=info.available_members,
        selected_members=requested,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )

    cache_key = _segmentor_cache_key(info, requested, local_files_only, device)
    with _SEGMENTOR_CACHE_LOCK:
        cached = _SEGMENTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    local_repo = _find_local_trqseg_v1_repo() if info.repo_id == "yuki-okamura-hf/TRQseg-v1" else None

    resolved_members = []
    for m in requested:
        rel = f"weights/{m}/model.safetensors"
        local_path = None
        if local_repo is not None:
            candidate = Path(local_repo) / rel
            if candidate.exists():
                local_path = str(candidate)
        resolved_members.append(
            SegmentorMember(
                member_id=m,
                relative_path=rel,
                local_path=local_path,
                revision=revision,
            )
        )

    loaded = LoadedSegmentor(
        info=info,
        members=tuple(resolved_members),
        device=device,
        local_files_only=local_files_only,
    )

    with _SEGMENTOR_CACHE_LOCK:
        _SEGMENTOR_CACHE[cache_key] = loaded
    return loaded


def segment_trq(
    image: ImageInput,
    *,
    study_id: str | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    device: str = "auto",
) -> SegmentationResult:
    """Segment TRQ region and return segmentation container."""
    if segmentor is None:
        raise ValueError(
            "segmentor must be specified explicitly (string alias/repo ID or LoadedSegmentor)."
        )
    seg = load_segmentor(segmentor, revision=revision, device=device) if isinstance(segmentor, str) else segmentor
    return seg.segment_trq(image, study_id=study_id)


def _meta(segmentation: SegmentationResult, method: MethodName, detail: DetailLevel) -> ResultMeta:
    """Construct common result metadata from segmentation context."""
    sinfo = segmentation.segmentor
    return ResultMeta(
        study_id=segmentation.study_id,
        method=method,
        detail=detail,
        input_source=segmentation.study_id,
        segmentor_name=None if sinfo is None else sinfo.name,
        segmentor_repo_id=None if sinfo is None else sinfo.repo_id,
        segmentor_revision=None if sinfo is None else sinfo.resolved_revision,
        segmentor_members=() if sinfo is None else tuple(sinfo.selected_members),
        library_version="0.1.0a0",
    )


@overload
def quantify(
    segmentation: SegmentationResult,
    *,
    method: Literal["okamura"],
    options: OkamuraOptions | None = None,
    detail: DetailLevel = "summary",
) -> AnalysisResultOkamura: ...


@overload
def quantify(
    segmentation: SegmentationResult,
    *,
    method: Literal["chaunzwa"],
    options: ChaunzwaOptions | None = None,
    detail: DetailLevel = "summary",
) -> AnalysisResultChaunzwa: ...


def quantify(
    segmentation: SegmentationResult,
    *,
    method: MethodName,
    options: OkamuraOptions | ChaunzwaOptions | None = None,
    detail: DetailLevel = "summary",
) -> AnalysisResultBase:
    """Quantify TRQ segmentation with selected method implementation."""
    if method == "okamura":
        opt = options if isinstance(options, OkamuraOptions) else OkamuraOptions()
        return quantify_okamura(segmentation, options=opt, detail=detail, meta=_meta(segmentation, "okamura", detail))

    if method == "chaunzwa":
        opt = options if isinstance(options, ChaunzwaOptions) else ChaunzwaOptions()
        return quantify_chaunzwa(segmentation, options=opt, detail=detail, meta=_meta(segmentation, "chaunzwa", detail))

    raise ValueError(f"Unsupported method: {method}")


@overload
def analyze(
    image: ImageInput,
    *,
    method: Literal["okamura"],
    study_id: str | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    options: OkamuraOptions | None = None,
    detail: DetailLevel = "summary",
    device: str = "auto",
) -> AnalysisResultOkamura: ...


@overload
def analyze(
    image: ImageInput,
    *,
    method: Literal["chaunzwa"],
    study_id: str | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    options: ChaunzwaOptions | None = None,
    detail: DetailLevel = "summary",
    device: str = "auto",
) -> AnalysisResultChaunzwa: ...


def analyze(
    image: ImageInput,
    *,
    method: MethodName,
    study_id: str | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    options: OkamuraOptions | ChaunzwaOptions | None = None,
    detail: DetailLevel = "summary",
    device: str = "auto",
) -> AnalysisResultBase:
    """Run one-shot segmentation + quantification."""
    if segmentor is None:
        raise ValueError(
            "segmentor must be explicitly specified (string alias/repo ID or LoadedSegmentor)."
        )

    seg = segment_trq(image, study_id=study_id, segmentor=segmentor, revision=revision, device=device)

    # Attach CT context for downstream method implementations.
    img = image if isinstance(image, nib.spatialimages.SpatialImage) else nib.load(str(image))
    ct = np.asarray(img.get_fdata(), dtype=float)
    for m in seg.members:
        base = m.raw_output if isinstance(m.raw_output, dict) else {}
        base["ct_array"] = ct
        base["ct_image"] = img
        m.raw_output = base

    return quantify(seg, method=method, options=options, detail=detail)


@overload
def analyze_many(
    images: Sequence[ImageInput],
    *,
    method: Literal["okamura"],
    study_ids: Sequence[str | None] | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    options: OkamuraOptions | None = None,
    detail: DetailLevel = "summary",
    on_error: OnError = "record",
    device: str = "auto",
) -> BatchAnalysisResultOkamura: ...


@overload
def analyze_many(
    images: Sequence[ImageInput],
    *,
    method: Literal["chaunzwa"],
    study_ids: Sequence[str | None] | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    options: ChaunzwaOptions | None = None,
    detail: DetailLevel = "summary",
    on_error: OnError = "record",
    device: str = "auto",
) -> BatchAnalysisResultChaunzwa: ...


def analyze_many(
    images: Sequence[ImageInput],
    *,
    method: MethodName,
    study_ids: Sequence[str | None] | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    options: OkamuraOptions | ChaunzwaOptions | None = None,
    detail: DetailLevel = "summary",
    on_error: OnError = "record",
    device: str = "auto",
) -> BatchAnalysisResultBase:
    """Run one-shot analysis for multiple studies."""
    if study_ids is not None and len(study_ids) != len(images):
        raise ValueError("study_ids must match images length")
    if segmentor is None:
        raise ValueError(
            "segmentor must be explicitly specified (string alias/repo ID or LoadedSegmentor)."
        )

    seg_loaded = load_segmentor(segmentor, revision=revision, device=device) if isinstance(segmentor, str) else segmentor

    results = []
    errors: list[BatchErrorRecord] = []
    for i, image in enumerate(images):
        sid = None if study_ids is None else study_ids[i]
        try:
            r = analyze(
                image,
                method=method,
                study_id=sid,
                segmentor=seg_loaded,
                options=options,
                detail=detail,
                device=device,
            )
            results.append(r)
        except Exception as e:
            if on_error == "raise":
                raise
            if on_error == "record":
                errors.append(
                    BatchErrorRecord(
                        input_source=str(image),
                        study_id=sid,
                        error_type=type(e).__name__,
                        message=str(e),
                    )
                )
            # on_error == "skip" -> ignore

    if method == "okamura":
        return BatchAnalysisResultOkamura(method="okamura", results=tuple(results), errors=tuple(errors))
    if method == "chaunzwa":
        return BatchAnalysisResultChaunzwa(method="chaunzwa", results=tuple(results), errors=tuple(errors))
    raise ValueError(f"Unsupported method: {method}")
