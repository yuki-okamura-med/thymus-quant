from __future__ import annotations

from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np

ImageLike = Union[str, Path, nib.spatialimages.SpatialImage]


def load_image(image: ImageLike) -> nib.spatialimages.SpatialImage:
    if isinstance(image, nib.spatialimages.SpatialImage):
        return image
    return nib.load(str(image))


def load_array_and_spacing(image: ImageLike) -> tuple[np.ndarray, tuple[float, float, float], nib.spatialimages.SpatialImage]:
    img = load_image(image)
    arr = np.asarray(img.get_fdata())
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return arr, spacing, img
