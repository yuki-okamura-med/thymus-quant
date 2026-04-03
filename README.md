# thymus-quant

Quantification toolkit for thymic involution metrics from chest CT and TRQ masks.

This implementation focuses on the **quantification layer**:
- TRQ representative HU (mode)
- TRQ volume (mL)
- ETV (Estimator of Thymic Volume)
- Optional ensemble QC (pairwise DSC/JSD/HU variance)

## Install

```bash
pip install -e .
```

## Quick start

```python
import thymus_quant as tq

result = tq.analyze(
    "ct.nii.gz",
    trq_mask="trq_mask.nii.gz",
    study_id="patient001",
    protocol="okamura2025_plus_ptt",
)

print(result.summary.etv_ml)
print(result.qc.status)
```

## CLI

```bash
thymus-quant analyze \
  --ct ct.nii.gz \
  --trq-mask trq_mask.nii.gz \
  --study-id patient001 \
  --protocol okamura2025_plus_ptt
```

Optional ensemble QC:

```bash
thymus-quant analyze \
  --ct ct.nii.gz \
  --trq-mask trq_mask.nii.gz \
  --ensemble fold0_mask.nii.gz fold1_mask.nii.gz fold2_mask.nii.gz fold3_mask.nii.gz fold4_mask.nii.gz
```
