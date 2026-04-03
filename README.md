# thymus-quant

Thymus quantification toolkit.

Implemented APIs:
- `segment_trq`
- `quantify(method="okamura"|"chaunzwa")`
- `analyze`
- `analyze_many`

## Segmentation backend

`okamura_trq_v1` now uses the published model repository:
- Hugging Face: `yuki-okamura-hf/TRQseg-v1`
- architecture: DeepLabV3-ResNet50 (5-fold ensemble)

### No duplicate model loading

The package caches loaded segmentors keyed by:
- segmentor/repo
- revision
- selected fold members
- cache/local-files flags
- device

So repeated calls with the same configuration reuse already loaded models.

### Local mirror priority

If local weights exist (e.g. `/mnt/w/repos/TRQseg-v1/weights/fold-*/model.safetensors`), they are used first.
If not found locally, the code attempts `huggingface_hub` download.
