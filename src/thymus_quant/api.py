from __future__ import annotations

"""Top-level public API for :mod:`thymus_quant`.

Design intent:
- `api.py` keeps public entry points and orchestration.
- method-specific quantification logic lives under `thymus_quant.quantification`.
"""

import logging
import os
import subprocess
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Sequence, overload

from .exceptions import SegmentorConfigurationError
from ._version import __version__
from .quantification.okamura import OkamuraOptions, quantify_okamura
from .results import (
    AnalysisResultBase,
    AnalysisResultOkamura,
    BatchAnalysisResultBase,
    BatchAnalysisResultOkamura,
    BatchErrorRecord,
    DetailLevel,
    MethodName,
    OnError,
    ResultMeta,
)
from .segmentors import ImageInput, LoadedSegmentor, SegmentationResult, SegmentorInfo, SegmentorMember


logger = logging.getLogger(__name__)

_SEGMENTOR_CACHE: dict[tuple, LoadedSegmentor] = {}
_SEGMENTOR_CACHE_LOCK = threading.Lock()


def list_methods() -> tuple[MethodName, ...]:
    """Return supported quantification method names."""
    return ("okamura",)


def _validate_method(method: str, *, function: str, study_id: str | None = None) -> MethodName:
    allowed = list_methods()
    if method not in allowed:
        raise ValueError(f"{function}(study_id={study_id!r}) received unknown method {method!r}; allowed values are {allowed}")
    return method  # type: ignore[return-value]


def _validate_detail(detail: str, *, function: str, study_id: str | None = None) -> DetailLevel:
    allowed: tuple[DetailLevel, ...] = ("summary", "full")
    if detail not in allowed:
        raise ValueError(f"{function}(study_id={study_id!r}) received unknown detail {detail!r}; allowed values are {allowed}")
    return detail  # type: ignore[return-value]


def _validate_on_error(on_error: str, *, function: str) -> OnError:
    allowed: tuple[OnError, ...] = ("raise", "record", "skip")
    if on_error not in allowed:
        raise ValueError(f"{function} received unknown on_error {on_error!r}; allowed values are {allowed}")
    return on_error  # type: ignore[return-value]


def _validate_options(method: MethodName, options: OkamuraOptions | None, *, function: str, study_id: str | None = None):
    if method == "okamura":
        if options is not None and not isinstance(options, OkamuraOptions):
            raise TypeError(
                f"{function}(study_id={study_id!r}, method='okamura') requires OkamuraOptions or None; got {type(options).__name__}"
            )
        return options if options is not None else OkamuraOptions()
    raise ValueError(f"Unsupported method: {method}")


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
            preprocessing_version="trqseg_v1_preprocess_v1",
        ),
        "heuristic_trq": SegmentorInfo(
            name="heuristic_trq",
            repo_id=None,
            architecture="heuristic",
            is_ensemble=False,
            available_members=("single",),
            selected_members=("single",),
            weight_source="none",
            preprocessing_version="heuristic_v1",
        ),
    }


def _resolve_members(members: Sequence[int | str] | None, default: Sequence[str]) -> tuple[str, ...]:
    """Normalize member selector into fold-name tuple."""
    if members is None:
        return tuple(default)
    if len(members) == 0:
        raise SegmentorConfigurationError("load_segmentor(members=...) must not be empty")
    out = []
    for m in members:
        if isinstance(m, int):
            if m < 0:
                raise SegmentorConfigurationError(f"load_segmentor received invalid negative member index {m}")
            out.append(f"fold-{m}")
        else:
            s = str(m)
            out.append(s if (s in default or s.startswith("fold-")) else f"fold-{s}")
    resolved = tuple(out)
    if len(set(resolved)) != len(resolved):
        raise SegmentorConfigurationError(f"load_segmentor received duplicate members: {resolved}")
    allowed = set(default)
    bad = [m for m in resolved if m not in allowed]
    if bad:
        raise SegmentorConfigurationError(f"load_segmentor received unknown members {bad}; allowed members are {tuple(default)}")
    return resolved


def _find_local_trqseg_v1_repo() -> str | None:
    """Find local mirror path for TRQseg-v1 if available."""
    env_path = os.environ.get("THYQ_TRQSEG_V1_LOCAL_REPO")
    if not env_path:
        return None
    for c in [Path(env_path)]:
        if c.exists() and c.is_dir():
            return str(c)
    return None


def _resolve_local_git_revision(repo_path: str | None) -> str | None:
    """Return local git HEAD for a model mirror, when available."""
    if repo_path is None:
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    revision = proc.stdout.strip()
    return revision or None


def _same_commit(revision: str, local_revision: str | None) -> bool:
    """True when `revision` is the full commit SHA of the local mirror HEAD.

    Branch or tag names and short SHAs are not treated as a match, because the
    local mirror cannot tell which commit they name on the Hub.
    """
    return local_revision is not None and revision.strip().lower() == local_revision.strip().lower()


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
        info.requested_revision,
        info.resolved_revision,
        requested_members,
        bool(local_files_only),
        info.cache_dir,
        info.local_source,
        info.weight_source,
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
        raise SegmentorConfigurationError(
            "segmentor must be specified explicitly (e.g. 'trqseg_v1'). "
            f"Available aliases: {', '.join(sorted(list_segmentors().keys()))}"
        )

    reg = list_segmentors()
    info = reg.get(segmentor)
    if info is None:
        raise SegmentorConfigurationError(
            f"load_segmentor received unknown segmentor alias {segmentor!r}; "
            f"allowed aliases are {tuple(sorted(reg.keys()))}. Custom Hugging Face repo IDs are not accepted as aliases."
        )

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
        preprocessing_version=info.preprocessing_version,
    )

    local_repo = _find_local_trqseg_v1_repo() if info.repo_id == "yuki-okamura-hf/TRQseg-v1" else None
    local_revision = _resolve_local_git_revision(local_repo)
    if local_repo is not None and revision is not None and not _same_commit(revision, local_revision):
        # A pinned revision must not be served by a mirror at another (or an unknown) commit.
        logger.warning(
            "Local TRQseg-v1 mirror %s is at %s, not at the requested revision %s; it is not used, and the "
            "requested revision is loaded from Hugging Face instead.",
            local_repo,
            local_revision or "an unknown commit",
            revision,
        )
        local_repo = None
        local_revision = None

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
                resolved_revision=revision,
                weight_source="local" if local_path else "unknown",
            )
        )
    sources = {m.weight_source for m in resolved_members}
    weight_source = "mixed" if len(sources) > 1 else (next(iter(sources)) if sources else "none")
    all_weights_local = bool(resolved_members) and all(m.local_path is not None for m in resolved_members)
    # The mirror's HEAD is recorded as local_revision, not as a resolved Hugging Face revision:
    # the mirror can have its own history (a commit that does not exist on the Hub).
    resolved_revision = revision
    for m in resolved_members:
        m.resolved_revision = None if m.local_path else revision
    any_local = any(m.local_path for m in resolved_members)
    info = SegmentorInfo(
        name=info.name,
        repo_id=info.repo_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        architecture=info.architecture,
        is_ensemble=(len(requested) > 1),
        available_members=info.available_members,
        selected_members=requested,
        cache_dir=None if cache_dir is None else str(cache_dir),
        weight_source=weight_source,  # updated when the weights are loaded
        local_source=local_repo if any_local else None,
        local_revision=local_revision if any_local else None,
        preprocessing_version=info.preprocessing_version,
    )

    cache_key = _segmentor_cache_key(info, requested, local_files_only, device)
    with _SEGMENTOR_CACHE_LOCK:
        cached = _SEGMENTOR_CACHE.get(cache_key)
        if cached is not None:
            return cached

    loaded = LoadedSegmentor(
        info=info,
        members=tuple(resolved_members),
        device=device,
        local_files_only=local_files_only,
    )

    with _SEGMENTOR_CACHE_LOCK:
        existing = _SEGMENTOR_CACHE.get(cache_key)
        if existing is not None:
            return existing
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
    prep = segmentation.preprocessing or {}
    return ResultMeta(
        study_id=segmentation.study_id,
        method=method,
        detail=detail,
        input_source=None if segmentation.image is None else segmentation.image.source,
        segmentor_name=None if sinfo is None else sinfo.name,
        segmentor_repo_id=None if sinfo is None else sinfo.repo_id,
        segmentor_requested_revision=None if sinfo is None else sinfo.requested_revision,
        segmentor_revision=None if sinfo is None else sinfo.resolved_revision,
        segmentor_members=() if sinfo is None else tuple(sinfo.selected_members),
        segmentor_weight_source=None if sinfo is None else sinfo.weight_source,
        segmentor_local_source=None if sinfo is None else sinfo.local_source,
        segmentor_local_revision=None if sinfo is None else sinfo.local_revision,
        segmentor_weight_sha256=() if sinfo is None else tuple(sinfo.weight_sha256),
        preprocessing_version=None if sinfo is None else sinfo.preprocessing_version,
        image_shape=None if segmentation.image is None else tuple(segmentation.image.shape),
        spacing_mm=None if segmentation.image is None else tuple(segmentation.image.spacing_mm),
        orientation=None if segmentation.image is None else segmentation.image.orientation,
        model_orientation=prep.get("model_orientation"),
        orientation_status=prep.get("orientation_status"),
        reoriented_for_model=prep.get("reoriented_for_model"),
        geometry=None if segmentation.image is None else segmentation.image.geometry,
        intensity=None if segmentation.image is None else segmentation.image.intensity,
        inplane_shape=None if prep.get("inplane_shape") is None else tuple(prep["inplane_shape"]),
        inplane_is_512=prep.get("inplane_is_512"),
        network_pixel_mm=None if prep.get("network_pixel_mm") is None else tuple(prep["network_pixel_mm"]),
        library_version=__version__,
    )


@overload
def quantify(
    segmentation: SegmentationResult,
    *,
    method: Literal["okamura"],
    options: OkamuraOptions | None = None,
    detail: DetailLevel = "summary",
) -> AnalysisResultOkamura: ...


def quantify(
    segmentation: SegmentationResult,
    *,
    method: MethodName,
    options: OkamuraOptions | None = None,
    detail: DetailLevel = "summary",
) -> AnalysisResultBase:
    """Quantify TRQ segmentation with selected method implementation."""
    method = _validate_method(method, function="quantify", study_id=segmentation.study_id)
    detail = _validate_detail(detail, function="quantify", study_id=segmentation.study_id)
    opt = _validate_options(method, options, function="quantify", study_id=segmentation.study_id)
    meta = _meta(segmentation, method, detail)
    meta.method_version = f"{method}_v2"
    meta.options = asdict(opt)
    if method == "okamura":
        return quantify_okamura(segmentation, options=opt, detail=detail, meta=meta)

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


def analyze(
    image: ImageInput,
    *,
    method: MethodName,
    study_id: str | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    options: OkamuraOptions | None = None,
    detail: DetailLevel = "summary",
    device: str = "auto",
) -> AnalysisResultBase:
    """Run one-shot segmentation + quantification."""
    method = _validate_method(method, function="analyze", study_id=study_id)
    detail = _validate_detail(detail, function="analyze", study_id=study_id)
    options = _validate_options(method, options, function="analyze", study_id=study_id)
    if segmentor is None:
        raise SegmentorConfigurationError(
            "segmentor must be explicitly specified (string alias/repo ID or LoadedSegmentor)."
        )

    seg = segment_trq(image, study_id=study_id, segmentor=segmentor, revision=revision, device=device)

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


def analyze_many(
    images: Sequence[ImageInput],
    *,
    method: MethodName,
    study_ids: Sequence[str | None] | None = None,
    segmentor: str | LoadedSegmentor | None = None,
    revision: str | None = None,
    options: OkamuraOptions | None = None,
    detail: DetailLevel = "summary",
    on_error: OnError = "record",
    device: str = "auto",
) -> BatchAnalysisResultBase:
    """Run one-shot analysis for multiple studies."""
    method = _validate_method(method, function="analyze_many")
    detail = _validate_detail(detail, function="analyze_many")
    on_error = _validate_on_error(on_error, function="analyze_many")
    options = _validate_options(method, options, function="analyze_many")
    if study_ids is not None and len(study_ids) != len(images):
        raise ValueError("study_ids must match images length")
    if segmentor is None:
        raise SegmentorConfigurationError(
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
            # Failed inputs have no row in to_frame(), so record the position in `images`.
            if r.meta is not None:
                r.meta.input_index = i
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
                        input_index=i,
                    )
                )
            # on_error == "skip" -> ignore

    if method == "okamura":
        return BatchAnalysisResultOkamura(method="okamura", results=tuple(results), errors=tuple(errors), n_requested=len(images))
    raise ValueError(f"Unsupported method: {method}")
