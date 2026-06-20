from __future__ import annotations

"""Segmentation backends and result containers for :mod:`thymus_quant`.

This module provides:
- `LoadedSegmentor`: runtime handle for a segmentation backend
- lightweight data classes that describe segmentor metadata and outputs
- TRQ segmentation using `TRQseg-v1` (DeepLabV3-ResNet50) when available
- explicit heuristic segmentation for debug environments
"""

import os
import warnings
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeAlias, Union

import nibabel as nib
import numpy as np

from .exceptions import SegmentorConfigurationError
from .inputs import ImageContext, load_image_context, make_image_context, validate_binary_mask, validate_mask_has_finite_ct

if TYPE_CHECKING:
    ImageInput: TypeAlias = Union[str, os.PathLike[str], nib.spatialimages.SpatialImage]
else:
    ImageInput: TypeAlias = Any


def _load_image(image: ImageInput) -> nib.spatialimages.SpatialImage:
    """Load NIfTI image from path-like input or pass through nibabel image."""
    if isinstance(image, nib.spatialimages.SpatialImage):
        return image
    return nib.load(str(image))


def _simple_trq_segmentation(ct: np.ndarray) -> np.ndarray:
    """Compute simple debug-only heuristic TRQ mask."""
    finite = np.isfinite(ct)
    mask = finite & (ct > -250) & (ct < 200)
    if mask.sum() == 0:
        return mask.astype(np.uint8)

    z_mid = ct.shape[2] // 2
    z0 = max(0, int(z_mid - ct.shape[2] * 0.25))
    z1 = min(ct.shape[2], int(z_mid + ct.shape[2] * 0.25))
    y1 = max(1, int(ct.shape[1] * 0.55))
    x0 = int(ct.shape[0] * 0.2)
    x1 = int(ct.shape[0] * 0.8)

    roi = np.zeros_like(mask, dtype=bool)
    roi[x0:x1, :y1, z0:z1] = True
    return (mask & roi).astype(np.uint8)


def _upsample_label_to_xyz(y_zxy: np.ndarray, x_size: int, y_size: int) -> np.ndarray:
    """Upsample downsampled label map back to original XY size and XYZ order."""
    y_up = np.repeat(np.repeat(y_zxy, 2, axis=1), 2, axis=2)
    y_up = y_up[:, :x_size, :y_size]
    return y_up.transpose(1, 2, 0)


@dataclass(slots=True)
class SegmentorMember:
    """Descriptor for one model member (e.g., one fold) in a segmentor."""

    member_id: str
    relative_path: str | None = None
    local_path: str | None = None
    revision: str | None = None
    resolved_revision: str | None = None
    weight_source: Literal["local", "downloaded", "unknown"] = "unknown"
    checksum: str | None = None


@dataclass(slots=True)
class SegmentorInfo:
    """Resolved metadata for a segmentation backend."""

    name: str
    repo_id: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    architecture: str | None = None
    is_ensemble: bool = False
    available_members: tuple[str, ...] = ()
    selected_members: tuple[str, ...] = ()
    cache_dir: str | None = None
    weight_source: Literal["local", "downloaded", "mixed", "none", "unknown"] = "unknown"
    local_source: str | None = None
    preprocessing_version: str | None = None


@dataclass(slots=True)
class LoadedSegmentor:
    """In-memory segmentor handle with lazy-loaded model objects.

    Notes
    -----
    - Models are loaded only once per instance (`_ensure_models_loaded`).
    - A lock protects concurrent first-load calls.
    - NN load/inference failures are reported; they do not silently fall back to
      the heuristic backend.
    """

    info: SegmentorInfo
    members: tuple[SegmentorMember, ...] = ()
    device: str = "auto"
    local_files_only: bool = False
    _models: dict[str, Any] = field(default_factory=dict, repr=False)
    _runtime_device: str | None = field(default=None, repr=False)
    _load_lock: Any = field(default_factory=threading.Lock, repr=False)

    def _resolve_device(self) -> str:
        """Resolve concrete execution device from user hint."""
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _resolve_weight_path(self, member: SegmentorMember) -> str:
        """Resolve local or Hub path for one member's weight file."""
        if member.local_path and os.path.exists(member.local_path):
            member.weight_source = "local"
            return member.local_path

        if not self.info.repo_id:
            raise FileNotFoundError(
                f"No local weight found for member '{member.member_id}', and repo_id is not set."
            )

        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:  # pragma: no cover - dependency/runtime environment dependent
            raise RuntimeError(
                "huggingface_hub is required to download TRQseg-v1 weights when local files are not found."
            ) from e

        filename = member.relative_path or f"weights/{member.member_id}/model.safetensors"
        path = hf_hub_download(
            repo_id=self.info.repo_id,
            filename=filename,
            revision=self.info.resolved_revision,
            cache_dir=self.info.cache_dir,
            local_files_only=self.local_files_only,
        )
        member.local_path = path
        member.weight_source = "downloaded"
        return path

    def _ensure_models_loaded(self) -> None:
        """Lazy-load NN models for selected members if not loaded yet."""
        if self._models:
            return

        with self._load_lock:
            if self._models:
                return

            # heuristic-only segmentor
            if self.info.repo_id is None or self.info.name == "heuristic_trq":
                self._runtime_device = "cpu"
                return

            try:
                import torchvision
                from safetensors.torch import load_file
            except Exception as e:  # pragma: no cover - dependency/runtime environment dependent
                raise RuntimeError(
                    "NN segmentation requires torch, torchvision, and safetensors."
                ) from e

            runtime_device = self._resolve_device()
            models: dict[str, Any] = {}

            target_members = self.members or (SegmentorMember(member_id="fold-0"),)
            for m in target_members:
                weight_path = self._resolve_weight_path(m)

                model = torchvision.models.segmentation.deeplabv3_resnet50(
                    weights=None,
                    weights_backbone=None,
                    num_classes=3,
                    aux_loss=None,
                )
                state = load_file(str(weight_path))
                model.load_state_dict(state)
                model.to(runtime_device)
                model.eval()
                models[m.member_id] = model

            self._models = models
            self._runtime_device = runtime_device

    def _predict_logits(self, model: Any, x_np: np.ndarray, split_size: int = 64) -> np.ndarray:
        """Run batched forward pass and return raw logits on CPU numpy arrays."""
        import torch

        device = self._runtime_device or self._resolve_device()
        batch = torch.from_numpy(x_np.astype(np.float32)).to(device)
        chunks = torch.split(batch, split_size)
        outs = []
        with torch.no_grad():
            for c in chunks:
                outs.append(model(c)["out"].detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def segment_trq(
        self,
        image: ImageInput,
        *,
        study_id: str | None = None,
    ) -> "SegmentationResult":
        """Segment TRQ region for one input image.

        Returns
        -------
        SegmentationResult
            Includes representative/fused mask and member-level masks.
        """
        context = load_image_context(image)
        ct = np.asarray(context.ct_hu, dtype=np.float32)

        self._ensure_models_loaded()

        member_outputs: list[SegmentationMember] = []

        # NN path
        if self._models:
            x = ct[::2, ::2, :].transpose(2, 0, 1)
            x = np.clip(x, -1500, 1500) / 1500.0
            x = np.stack([x, x, x], axis=1).astype(np.float32)

            for m in self.members:
                model = self._models.get(m.member_id)
                if model is None:
                    continue

                logits = self._predict_logits(model, x)
                pred_ds = np.argmax(logits, axis=1).astype(np.uint8)
                pred_xyz = _upsample_label_to_xyz(pred_ds, ct.shape[0], ct.shape[1])

                trq = (pred_xyz == 2).astype(np.uint8)
                airway = (pred_xyz == 1).astype(np.uint8)

                member_outputs.append(
                    SegmentationMember(
                        member_id=m.member_id,
                        trq_mask=trq,
                        airway_mask=airway,
                        raw_output={
                            "backend": "hf_trqseg_v1",
                            "preprocessing_version": self.info.preprocessing_version,
                            "input_shape": tuple(ct.shape),
                            "output_grid": "input",
                        },
                    )
                )

        # explicit heuristic path
        if not member_outputs:
            if self.info.name != "heuristic_trq":
                raise RuntimeError(
                    f"Segmentor '{self.info.name}' produced no member outputs; heuristic fallback is disabled. "
                    "Choose segmentor='heuristic_trq' explicitly for debug-only heuristic segmentation."
                )
            warnings.warn(
                "heuristic_trq is a debug/experimental segmentor and is not the validated TRQseg-v1 research model.",
                UserWarning,
                stacklevel=2,
            )
            for m in self.members or (SegmentorMember(member_id="single"),):
                trq = _simple_trq_segmentation(ct)
                member_outputs.append(
                    SegmentationMember(
                        member_id=m.member_id,
                        trq_mask=trq,
                        airway_mask=None,
                        raw_output={"backend": "heuristic", "input_shape": tuple(ct.shape), "output_grid": "input"},
                    )
                )

        fused = (
            member_outputs[0].trq_mask
            if len(member_outputs) == 1
            else (np.mean(np.stack([x.trq_mask for x in member_outputs], axis=0), axis=0) >= 0.5).astype(np.uint8)
        )

        return SegmentationResult(
            study_id=study_id,
            segmentor=self.info,
            image=context,
            trq_mask=fused,
            airway_mask=None,
            members=tuple(member_outputs),
            merge_strategy="none" if len(member_outputs) == 1 else "vote",
        )


@dataclass(slots=True)
class SegmentationMember:
    """One member-level segmentation payload."""

    member_id: str
    trq_mask: Any
    airway_mask: Any | None = None
    raw_output: Any | None = None


@dataclass(slots=True)
class SegmentationResult:
    """Common segmentation container passed into quantification methods."""

    study_id: str | None = None
    segmentor: SegmentorInfo | None = None
    image: ImageContext | None = None
    trq_mask: Any | None = None
    airway_mask: Any | None = None
    members: Sequence[SegmentationMember] = field(default_factory=tuple)
    merge_strategy: Literal["none", "vote", "union", "intersection", "custom"] = "none"

    @classmethod
    def from_mask(
        cls,
        *,
        trq_mask: Any,
        ct_hu: Any,
        spacing_mm: tuple[float, float, float],
        study_id: str | None = None,
        affine: Any | None = None,
        source: str | None = None,
        orientation: tuple[str, str, str] | None = None,
    ) -> "SegmentationResult":
        """Build a segmentation result from one external TRQ mask."""
        context = make_image_context(
            ct_hu=ct_hu,
            spacing_mm=spacing_mm,
            affine=affine,
            source=source,
            orientation=orientation,
        )
        mask = validate_binary_mask(trq_mask, ct_shape=context.shape, name="trq_mask")
        validate_mask_has_finite_ct(mask, np.asarray(context.ct_hu), name="trq_mask")
        member = SegmentationMember(member_id="single", trq_mask=mask, raw_output={"backend": "external_mask"})
        return cls(
            study_id=study_id,
            segmentor=None,
            image=context,
            trq_mask=mask,
            members=(member,),
            merge_strategy="none",
        )

    @classmethod
    def from_member_masks(
        cls,
        *,
        member_masks: Sequence[Any],
        ct_hu: Any,
        spacing_mm: tuple[float, float, float],
        study_id: str | None = None,
        member_ids: Sequence[str] | None = None,
        affine: Any | None = None,
        source: str | None = None,
        orientation: tuple[str, str, str] | None = None,
    ) -> "SegmentationResult":
        """Build a segmentation result from multiple external member masks."""
        if not member_masks:
            raise SegmentorConfigurationError("from_member_masks requires at least one member mask")
        context = make_image_context(
            ct_hu=ct_hu,
            spacing_mm=spacing_mm,
            affine=affine,
            source=source,
            orientation=orientation,
        )
        ids = tuple(member_ids) if member_ids is not None else tuple(f"member-{i}" for i in range(len(member_masks)))
        if len(ids) != len(member_masks):
            raise SegmentorConfigurationError("member_ids length must match member_masks length")
        if len(set(ids)) != len(ids):
            raise SegmentorConfigurationError("member_ids must be unique")
        masks = []
        members = []
        for i, (member_id, raw_mask) in enumerate(zip(ids, member_masks)):
            mask = validate_binary_mask(raw_mask, ct_shape=context.shape, name=f"member_masks[{i}]")
            validate_mask_has_finite_ct(mask, np.asarray(context.ct_hu), name=f"member_masks[{i}]")
            masks.append(mask)
            members.append(
                SegmentationMember(member_id=str(member_id), trq_mask=mask, raw_output={"backend": "external_mask"})
            )
        fused = (np.mean(np.stack(masks, axis=0), axis=0) >= 0.5)
        return cls(
            study_id=study_id,
            segmentor=None,
            image=context,
            trq_mask=fused,
            members=tuple(members),
            merge_strategy="vote" if len(members) > 1 else "none",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return lightweight JSON-friendly segmentation summary."""
        return {
            "study_id": self.study_id,
            "segmentor": None
            if self.segmentor is None
            else {
                "name": self.segmentor.name,
                "repo_id": self.segmentor.repo_id,
                "requested_revision": self.segmentor.requested_revision,
                "resolved_revision": self.segmentor.resolved_revision,
                "selected_members": list(self.segmentor.selected_members),
                "weight_source": self.segmentor.weight_source,
                "local_source": self.segmentor.local_source,
                "preprocessing_version": self.segmentor.preprocessing_version,
            },
            "image": None
            if self.image is None
            else {
                "shape": list(self.image.shape),
                "spacing_mm": list(self.image.spacing_mm),
                "orientation": None if self.image.orientation is None else list(self.image.orientation),
                "source": self.image.source,
            },
            "merge_strategy": self.merge_strategy,
            "n_members": len(self.members),
            "member_ids": [m.member_id for m in self.members],
            "has_trq_mask": self.trq_mask is not None,
            "has_airway_mask": self.airway_mask is not None,
        }
