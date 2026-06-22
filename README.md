# thymus-quant

This package lets you quantify thymic tissue contained inside the thymic region based on human CT images.
Main APIs:

- `segment_trq`
- `quantify`
- `analyze`
- `analyze_many`

## Installation

Recommended installation:
```bash
pip install "thymus-quant[segmentation]"
```

When you have your own segmentation model and you are not using our default segmentation model (`trqseg_v1` model) for inference:
```bash
pip install thymus-quant
```

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

`thymic_tissue_fraction` is the Thymic Tissue Fraction. Its unit is a 0-1
fraction, not percent.
`etv_ml` and `trq_volume_ml` are in mL.

## Reproducible Model Loading

```python
segmentor = tq.load_segmentor(
    "trqseg_v1",
    revision="<immutable-tested-revision>",
    device="auto",
)
```

When weights are loaded from Hugging Face, branch/tag/default revisions are
resolved to the underlying commit SHA before download and recorded in result
metadata. When all selected weights are loaded from a local git mirror, the local
mirror's HEAD commit is recorded instead. The result metadata distinguishes
requested and resolved revision and records the weight source (`local`,
`downloaded`, `mixed`, or `none`).

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

- CT values must be HU.
- CT arrays must be 3D.
- Masks must be 3D bool or binary 0/1 arrays.
- CT and mask shapes must match exactly.
- `spacing_mm` is 3 positive finite values in millimeters.
- Masked CT voxels must contain at least one finite HU value.
- DICOM directories are not directly supported by the public API.
- Orientation is recorded from the affine when available. The package does not
  silently flip or permute axes.
- Current TRQseg-v1 preprocessing records input/output shapes and assumes the
  returned mask is on the input grid. Inputs outside the reference preprocessing
  domain should be treated cautiously until a pinned regression fixture exists.

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

Okamura member fields are per-member values. Summary fields are ensemble
aggregates. QC-invalid members can still retain numeric values; check
`result.status`, `result.flags`, and `result.qc.flags` before using them.

## QC and Errors

Statuses are intentionally simple:

- `ok`: no relevant QC flags.
- `check`: numeric values may exist but require review.
- `failed`: no computable result.
- `not_available`: not applicable.

Important flags include `invalid_member`, `all_members_invalid`,
`used_invalid_members`, `kde_failed`, and `ensemble_qc_unavailable`.

KDE computation failure is not converted to a fake median mode. Missing spacing
does not produce successful NaN volume/ETV output. Single-member analysis can
return quantification values, but 5-member paper ensemble criteria are not
passed and `ensemble_qc_unavailable` is set.

Batch processing preserves input order and records errors with input index:

```python
batch = tq.analyze_many(paths, method="okamura", segmentor=segmentor, on_error="record")
errors = batch.errors_to_frame()
```

`on_error` values are `raise`, `record`, and `skip`.

## Experimental: Chaunzwa

Chaunzwa support is experimental and is not a formal release endpoint. Its API
and numerical implementation may change. Do not use the current Chaunzwa output
as a collaborative primary endpoint, and do not treat it as validated equivalently
to Okamura.

When called explicitly, Chaunzwa emits `ExperimentalWarning` and sets
experimental metadata/flags in the result. Known numerical issues are documented
in [docs/chaunzwa_known_issues.md](docs/chaunzwa_known_issues.md).
