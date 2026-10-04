# thymus-quant

This package lets you quantify thymic tissue contained within the thymic region
on human CT images. It implements the framework described in Okamura YT et
al., Ann Biomed Eng, 2025. http://dx.doi.org/10.1007/s10439-025-03805-z

From version 0.1.0a4, thymus-quant checks the voxel order (orientation) of the
input image and, when needed, reorders the voxels to the order used for training
(LAS) before segmentation. Earlier versions used the voxel array as stored in the
file. If your NIfTI files were not in LAS order (for example, files written by
ITK or SimpleITK, which are usually LPS, or files reoriented to RAS), the
segmentation with earlier versions may have been poor, and we recommend running
those images again with 0.1.0a4 or later. You can check the voxel order of a file
with `nibabel.aff2axcodes(nibabel.load(path).affine)`; see
[Voxel order (orientation)](#voxel-order-orientation) for details.

Main readouts:

- $A_{TRQ}$: representative HU value of the thymic region of quantification.
- $V_{TRQ}$: volume of the thymic region of quantification.
- **Estimated Thymic Volume (ETV)**: estimated thymic tissue volume in mL.
- Thymic tissue fraction: fraction of thymic tissue in the thymic region,
  reported as a 0-1 value.

Main APIs:

- `segment_trq`
- `quantify`
- `analyze`
- `analyze_many`

## Installation

Install from PyPI:
```bash
pip install thymus-quant
```

This installs the dependencies needed for both quantification and TRQseg-v1
segmentation.

Model weights are loaded through the Hugging Face cache unless `local_files_only`
or an explicit local mirror is configured. A local TRQseg-v1 mirror may be set
with `THYQ_TRQSEG_V1_LOCAL_REPO=/path/to/TRQseg-v1`.

## Minimal Example

```python
import thymus_quant as tq

result = tq.analyze(
    "case.nii.gz",
    method="okamura",
    segmentor="trqseg_v1",
    study_id="case-001",
)

print(result.thymic_tissue_fraction)
print(result.etv_ml)
print(result.qc.status)
```

`thymic_tissue_fraction` is the Thymic Tissue Fraction. It is reported as a 0-1
fraction, not as a percentage.
`etv_ml` and `trq_volume_ml` are in mL.

## Reproducible Model Loading

```python
segmentor = tq.load_segmentor(
    "trqseg_v1",
    revision="<immutable-tested-revision>",
    device="auto",
)
```

`device` controls where TRQseg-v1 model inference runs. Supported values are
`"auto"`, `"cpu"`, and `"cuda"`. `"auto"` selects CUDA when PyTorch detects an
available CUDA device, and otherwise falls back to CPU.

When weights are loaded from Hugging Face, branch/tag/default revisions are
resolved to the underlying commit SHA before download and recorded in result
metadata (`segmentor_revision`).

A local mirror (`THYQ_TRQSEG_V1_LOCAL_REPO`) is used when no `revision` is given,
or when `revision` is the full commit SHA of the mirror's HEAD. If `revision` is
anything else (another commit, a branch or tag name, or a short SHA), or the
mirror's HEAD cannot be read, the mirror is not used: a WARNING is logged by the
`thymus_quant.api` logger and the requested revision is loaded from Hugging Face
(or its cache). The mirror's HEAD is recorded separately as
`segmentor_local_revision`: a mirror can have its own git history, so its HEAD
is not a Hugging Face commit.

Once the weights are loaded, the result metadata records the weight source
(`local`, `downloaded` or `mixed`) and the SHA-256 of each member's weight file
(`segmentor_weight_sha256`, in member order), which identify the weights
whatever their origin.

## Two-Stage Workflow

```python
seg = tq.segment_trq(
    "case.nii.gz",
    segmentor=segmentor,
)

result = tq.quantify(
    seg,
    method="okamura",
)
```

`segmentor` is always explicit. Omitting it never selects the heuristic backend.

## Existing Mask

Single external mask:

```python
seg = tq.SegmentationResult.from_mask(
    trq_mask=mask,
    ct_hu=ct_hu,
    spacing_mm=(0.7, 0.7, 1.0),
    study_id="case-001",
)

result = tq.quantify(seg, method="okamura")
```

Okamura member ensemble:

```python
seg = tq.SegmentationResult.from_member_masks(
    member_masks=[mask0, mask1, mask2, mask3, mask4],
    ct_hu=ct_hu,
    spacing_mm=(0.7, 0.7, 1.0),
    study_id="case-001",
    member_ids=["fold-0", "fold-1", "fold-2", "fold-3", "fold-4"],
)

result = tq.quantify(seg, method="okamura")
```

## Inputs

Accepted image inputs for segmentation are NIfTI path-like values and nibabel
spatial images. Existing masks can be supplied with NumPy-compatible arrays via
`SegmentationResult.from_mask` or `from_member_masks`.

Data contract:

- CT values must be HU. A chest CT in HU contains air and lung, so when the 1st
  percentile of the CT values is above -500 HU, the values are probably not HU
  (for example, the rescale intercept of -1024 was not applied, or the image
  was windowed or normalized). Such inputs are not rejected: a WARNING is
  logged by the `thymus_quant.inputs` logger and a note is added to
  `result.warnings`. The values checked are recorded in `ImageContext.intensity`
  and the result metadata (`hu_p01`, `hu_p50`, `hu_p99`, `looks_like_hu`).
- CT arrays must be 3D.
- Masks must be 3D bool or binary 0/1 arrays.
- CT and mask shapes must match exactly.
- `spacing_mm` is 3 positive finite values in millimeters.
- `spacing_mm` (for NIfTI files, the header voxel sizes) is normalized to Python
  floats and used for voxel-volume calculation. NIfTI files whose spatial unit
  (`xyzt_units`) is meter or micron are converted to mm, with a note; "unknown"
  is taken as mm. When an affine is available, each voxel size is compared with
  the matching axis length of the affine after removing shear (QR
  decomposition); when any differs by more than 0.1%, a WARNING is logged by the
  `thymus_quant.inputs` logger and a note is added to `result.warnings`. Volumes
  and ETV still use `spacing_mm`. Gantry tilt (shear of the slice axis) and
  rotation do not cause this warning when the header gives the perpendicular
  slice spacing. See the geometry report below.
- Other header notes: the NIfTI orientation is unspecified
  (`qform_code = sform_code = 0`; nibabel's fallback LAS is not used as known
  orientation), the qform and sform disagree on left and right, the affine is
  not usable (non-finite or singular), or the header looks like library
  defaults (1 mm voxels, axis-aligned unit affine, origin 0, as written by
  `nibabel.Nifti1Image(arr, np.eye(4))` or SimpleITK without copying spacing
  and direction; this is only a hint).
- The input notes above do not change values, `flags`, `qc.status` or
  `qc.paper_criteria_met`. `result.warnings` is also in the `warnings` column of
  `to_record()` / `to_frame()`.
- External CT/mask inputs are not resampled, cropped, padded, flipped, or
  permuted. Shape mismatches raise `InputValidationError` instead of being
  repaired automatically.
- Masked CT voxels must contain at least one finite HU value. In quantification, a
  member whose TRQ contains any non-finite CT value (NaN/inf) is not computed
  (`computation_failed`, `nonfinite_hu`): measuring only the finite part would still
  count the missing voxels in the volume. Non-finite CT values anywhere in the image
  are also reported in `result.warnings` (`ImageContext.intensity["n_nonfinite"]`).
- DICOM directories are not directly supported by the public API.
- Current TRQseg-v1 preprocessing records input/output shapes and returns masks
  on the input grid. Inputs outside the reference preprocessing domain should be
  treated cautiously until a pinned regression fixture exists.

### Voxel order (orientation)

TRQseg-v1 segments each axial slice as a 2D image and was trained on NIfTI files
converted from DICOM with dcm2niix, which stores axial CT in LAS voxel order
(first array axis toward the patient's left, second toward anterior, third toward
superior; `nibabel.aff2axcodes(img.affine) == ("L", "A", "S")`). The network does
not read the affine, so an array stored in another order (for example LPS, the
usual result of converting DICOM with ITK/SimpleITK, or RAS after
`nibabel.as_closest_canonical`) is seen flipped or permuted.

- For TRQseg-v1 segmentation, when the input orientation is known and is not LAS,
  the voxel array is permuted/flipped to LAS for the network only, and the
  predicted masks are permuted/flipped back. No interpolation is done. Returned
  masks are on the input grid, in the input voxel order.
- Orientation is taken from the affine of NIfTI/nibabel inputs, or from
  `ImageContext.orientation` / `make_image_context(..., orientation=...)`.
  Arrays without affine or orientation are used as given (assumed LAS).
- LAS inputs take the same path as before, with identical results.
- Oblique or sheared affines are not resampled. Their geometry is recorded.
- When the orientation is unspecified (`qform_code = sform_code = 0`) or the
  affine is not usable, the array is used as given (`assumed_model_orientation`)
  and a note is added. An `ImageContext` built directly with an `affine` takes
  its orientation from that affine. `make_image_context(..., orientation=...)`
  raises `InputValidationError` when the given orientation contradicts the
  affine.
- The handling is recorded in `SegmentationResult.preprocessing` and in the
  result metadata: `model_orientation`, `orientation_status`
  (`model_orientation`, `reoriented`, `assumed_model_orientation`, `unresolved`),
  and `reoriented_for_model`. The same information is in the `input_orientation`,
  `orientation_status` and `reoriented_for_model` columns of `to_record()` /
  `to_frame()`. Reorientation is logged at INFO level by the
  `thymus_quant.segmentors` logger.
- `geometry` (in `ImageContext`, `SegmentationResult.to_dict()["image"]` and the
  result metadata) records `voxel_sizes_affine_mm`, `voxel_size_mismatch_max_mm`,
  `voxel_sizes_affine_unsheared_mm`, `voxel_size_mismatch_rel_max`,
  `voxel_volume_affine_mm3`, `voxel_volume_mismatch_rel`, `obliquity_max_deg`,
  `shear_max`, `orientation_source` (`affine`, `fallback`, `explicit` or None),
  `default_like_header`, and, for NIfTI images, `spatial_unit`,
  `unit_scale_to_mm`, `qform_code`, `sform_code`, `qform_sform_max_abs_diff` and
  `qform_sform_handedness_differs`. Only the notes described in the data
  contract are derived from them.
- The heuristic segmentor (`heuristic_trq`, debug only) is not affected.

### In-plane size

The TRQseg-v1 training images were 512 x 512, and the network sees each slice
downsampled by 2 (`ct[::2, ::2, :]`). Images of other in-plane sizes are
segmented as given; they are not padded, cropped or resampled.

- The in-plane size is checked after reorientation to LAS. When it is not
  512 x 512, a WARNING is logged by the `thymus_quant.segmentors` logger and a
  note is added to `result.warnings`. The note does not change `flags`,
  `qc.status` or `qc.paper_criteria_met`.
- `SegmentationResult.preprocessing`, the result metadata and the `to_record()` /
  `to_frame()` columns record `inplane_shape`, `inplane_is_512` and
  `network_pixel_mm` (the in-plane pixel size seen by the network, that is,
  2 x the voxel spacing).
- What matters most for the network is the pixel size (how large the anatomy
  appears), not the matrix size itself. In our checks on two scans, padding a
  512 x 512 image to 640 x 640 or cropping it to 448 x 448 changed the fold-0
  mask only slightly (Dice 0.988 to 0.995 against the 512 x 512 result).

## Segmentor Options

Built-in aliases:

- `trqseg_v1`: DeepLabV3-ResNet50 5-member TRQseg-v1 ensemble.
- `heuristic_trq`: debug/experimental heuristic only, not a research model.

Supported options include revision pinning, `members`/fold selection, `device`,
`local_files_only`, and `cache_dir`. Unknown segmentor strings are rejected and
are not interpreted as arbitrary Hugging Face repository IDs. Empty, duplicate,
negative, or out-of-range members are rejected. Model load failure does not fall
back to `heuristic_trq`.

## Result Glossary

- TRQ: thymic region of quantification.
- `A_TRQ`: representative TRQ attenuation used by published formulas.
- `V_TRQ`: TRQ volume.
- Thymic Tissue Fraction: fraction of thymic tissue in the TRQ, usually 0-1.
- ETV: estimated thymic volume, unit mL.
- `thymic_tissue_fraction`: summary Thymic Tissue Fraction for the study.
- `etv_ml`: summary ETV in mL.
- `trq_volume_ml`: summary TRQ volume in mL.

### How the Okamura values are computed (method version `okamura_v2`)

These follow the analysis code of the paper.

- For each member, a Gaussian KDE (`scipy.stats.gaussian_kde`, Scott's rule) is
  fitted to the HU values of every TRQ voxel and evaluated on fixed grids from
  -300 to 300 HU. Peaks are the strict local maxima inside this window, so a
  peak outside it (for example, air) is never the mode.
- `A_TRQ` (`trq_hu_mode`) is the highest peak on a 0.1 HU grid. The paper used
  a 1 HU grid; the 0.1 HU grid only removes up to 0.5 HU of rounding, which
  matters for fatty TRQs because ETV is proportional to `A_TRQ + 110`.
- The second-peak ratio (multimodality, invalid when above 0.5) and the mode
  used for the HU value variance are taken on the 1 HU grid, as in the paper.
- A member needs at least 2 TRQ voxels and a peak inside the window; otherwise
  it is not computed (`computation_failed`, with `too_few_voxels` or
  `kde_failed`).
- The ensemble QC values are computed over all supplied members, as in the
  paper: mean pairwise DSC (pairs of two empty masks are skipped; a member that
  was not computed counts with its mask), mean pairwise Jensen-Shannon value
  (pairs where both members have a KDE), and the unbiased variance of the 1 HU
  modes (None, with `hu_variance_unavailable`, when any member has no mode).
- The Jensen-Shannon value is the JS **distance** of
  `scipy.spatial.distance.jensenshannon` on the 1 HU KDE curves: the square root
  of the JS divergence with the natural logarithm. The paper calls it the JS
  divergence, and its threshold 0.1 refers to this value. Grid points where
  both curves are below 1e-300 are left out to avoid an infinite value from
  floating-point underflow.
- The criteria pass when the mean pairwise JS value is at most 0.1, the mean
  pairwise DSC is at least 0.7 and the HU value variance is at most 20.

Before 0.1.0a5 (`okamura_v1`), the mode was taken on a grid spanning the
data's minimum to maximum, peaks outside -300..300 HU could be the mode, and the
JS value was the base-2 JS divergence of HU values clipped to -300..300 HU,
computed only over QC-valid members. That value is much smaller than the JS
distance (on the paper's 31-case test set, median 0.002 and maximum 0.018
against 0.031 and 0.094), so the JS criterion was in practice never triggered.

Okamura member fields are per-member values. The `*_members` tuples have one
entry per member, in the order of `members`; a member whose values could not be
computed (for example, an empty mask) has None. Summary fields are ensemble
aggregates. QC-invalid members can still retain numeric values; check
`result.status`, `result.flags`, and `result.qc.flags` before using them.

## QC and Errors

Statuses are intentionally simple:

- `ok`: no relevant QC flags.
- `check`: numeric values may exist but require review.
- `failed`: no computable result.
- `not_available`: not applicable.

Important flags include `invalid_member`, `all_members_invalid`,
`used_invalid_members`, `kde_failed`, `ensemble_qc_unavailable`,
`ensemble_qc_not_applied`, `high_jsd`, `low_dsc`, `high_hu_variance`,
`jsd_unavailable`, and `hu_variance_unavailable`.

For the Okamura method, a non-unimodal member distribution is marked invalid
with `multimodal_or_invalid`. By default, any invalid member adds
`invalid_member` and makes the study status `check`. If at least one member
passes QC, the summary uses only QC-valid members. If all members fail QC but
still have computable numeric values, the summary uses those invalid members,
sets `all_members_invalid` and `used_invalid_members`, and remains `check`.

KDE computation failure is not converted to a fake median mode. Missing spacing
does not produce a successful result with NaN volume/ETV output. Single-member
analysis can return quantification values, but 5-member paper ensemble criteria
are not passed and `ensemble_qc_unavailable` is set.

`qc.paper_criteria_met` is True or False only for the paper's protocol: five
members (the TRQseg-v1 folds fold-0 to fold-4, or five supplied member masks)
quantified with the default `OkamuraOptions`. For other ensembles (for example
`members=[0, 1]`) or other options, the status still follows the configured
criteria, but `paper_criteria_met` is None and `result.warnings` says why.

`OkamuraOptions` rejects values that cannot give a valid result: every number
must be finite, `athymic_hu` must be greater than `aadipose_hu`, and
`delta_hu_margin` must be at least 0 and smaller than their difference.

When a member's `A_TRQ` is above `athymic_hu` (+80 HU), its thymic tissue
fraction is capped at 1 and a note is added to `result.warnings`. The method is
for non-contrast chest CT; values this high suggest contrast enhancement.

`OkamuraOptions(apply_qc=False)` skips the ensemble criteria (mean pairwise JS
divergence, mean pairwise DSC, HU value variance). The three values are still
reported in `result.qc`, but `ensemble_qc_not_applied` is set, the status is
`check`, and `qc.paper_criteria_met` is None (not evaluated).

`failed` means at least one member/mask was provided, but no computable Okamura
value was produced. `not_available` means the method had no applicable
member/mask to evaluate; this should be uncommon when using the public
constructors.

Batch processing preserves input order and records errors with input index:

```python
batch = tq.analyze_many(paths, method="okamura", segmentor=segmentor, on_error="record")
errors = batch.errors_to_frame()
```

`on_error` values are `raise`, `record`, and `skip`.

Inputs that fail have no row in `batch.to_frame()`, so match rows to inputs by
the `input_index` column (the position in `paths`), not by row position.
`input_source` is the file path for path inputs (None for in-memory images).
`errors_to_frame()` has the same `input_index` for the failed inputs, and
`members_to_frame()` has it for every member row.

## Coming soon: Expanded thymic composition analysis workflows.

Recent work by Chaunzwa et al. introduced an enhanced thymic composition analysis framework that builds upon and extends the methodology currently implemented in this package (Chaunzwa et al., medRxiv, 2025: https://doi.org/10.1101/2025.10.27.25338565, https://doi.org/10.1101/2025.10.20.25338395). Support for this framework is planned for a future release.
