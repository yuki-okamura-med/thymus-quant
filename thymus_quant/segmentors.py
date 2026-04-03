from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeAlias, Union

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    ImageInput: TypeAlias = Union[str, os.PathLike[str], nib.spatialimages.SpatialImage]
else:
    ImageInput: TypeAlias = Any


def _load_image(image: ImageInput) -> nib.spatialimages.SpatialImage:
    if isinstance(image, nib.spatialimages.SpatialImage):
        return image
    return nib.load(str(image))


def _simple_trq_segmentation(ct: np.ndarray) -> np.ndarray:
    # lightweight fallback segmentation (no DL dependency)
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


@dataclass(slots=True)
class SegmentorMember:
    member_id: str
    relative_path: str | None = None
    local_path: str | None = None
    revision: str | None = None


@dataclass(slots=True)
class SegmentorInfo:
    name: str
    repo_id: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    architecture: str | None = None
    is_ensemble: bool = False
    available_members: tuple[str, ...] = ()
    selected_members: tuple[str, ...] = ()
    cache_dir: str | None = None


@dataclass(slots=True)
class LoadedSegmentor:
    info: SegmentorInfo
    members: tuple[SegmentorMember, ...] = ()
    device: str = "auto"

    def segment_trq(
        self,
        image: ImageInput,
        *,
        study_id: str | None = None,
    ) -> "SegmentationResult":
        img = _load_image(image)
        ct = np.asarray(img.get_fdata(), dtype=float)

        member_outputs: list[SegmentationMember] = []
        for m in self.members or (SegmentorMember(member_id="single"),):
            trq = _simple_trq_segmentation(ct)
            member_outputs.append(SegmentationMember(member_id=m.member_id, trq_mask=trq, airway_mask=None, raw_output=None))

        fused = member_outputs[0].trq_mask if len(member_outputs) == 1 else (np.mean(np.stack([x.trq_mask for x in member_outputs], axis=0), axis=0) >= 0.5).astype(np.uint8)
        return SegmentationResult(
            study_id=study_id,
            segmentor=self.info,
            trq_mask=fused,
            airway_mask=None,
            members=tuple(member_outputs),
            merge_strategy="none" if len(member_outputs) == 1 else "vote",
        )


@dataclass(slots=True)
class SegmentationMember:
    member_id: str
    trq_mask: Any
    airway_mask: Any | None = None
    raw_output: Any | None = None


@dataclass(slots=True)
class SegmentationResult:
    study_id: str | None = None
    segmentor: SegmentorInfo | None = None
    trq_mask: Any | None = None
    airway_mask: Any | None = None
    members: Sequence[SegmentationMember] = field(default_factory=tuple)
    merge_strategy: Literal["none", "vote", "union", "intersection", "custom"] = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "segmentor": None if self.segmentor is None else {
                "name": self.segmentor.name,
                "repo_id": self.segmentor.repo_id,
                "resolved_revision": self.segmentor.resolved_revision,
                "selected_members": list(self.segmentor.selected_members),
            },
            "merge_strategy": self.merge_strategy,
            "n_members": len(self.members),
            "member_ids": [m.member_id for m in self.members],
            "has_trq_mask": self.trq_mask is not None,
            "has_airway_mask": self.airway_mask is not None,
        }
