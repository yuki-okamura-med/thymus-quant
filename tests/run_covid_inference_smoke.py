from __future__ import annotations

"""Integration smoke test for thymus-quant using public COVID-19 CT sample.

What this script does:
1) Download one public NIfTI sample to tests/test_images
2) Run fold-wise TRQ inference (fold-0..fold-4)
3) Run both quantification methods (okamura, chaunzwa)
4) Save outputs in separate directories:
   - non-text segmentation artifacts -> tests/segmentation_outputs
   - text/json summaries -> tests/text_outputs

Usage:
  python tests/run_covid_inference_smoke.py
"""

import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import nibabel as nib
import numpy as np

import thymus_quant as tq


SAMPLE_NIFTI_URL = (
    "https://huggingface.co/datasets/huggingface/CADS-dataset/resolve/main/"
    "0022_tcia_ct_images_covid19/images/"
    "volume-covid19-A-0700_day000_0000.nii.gz?download=true"
)
SAMPLE_FILENAME = "volume-covid19-A-0700_day000_0000.nii.gz"


def _ensure_sample(image_dir: Path) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    sample_path = image_dir / SAMPLE_FILENAME
    if not sample_path.exists():
        print(f"[download] {SAMPLE_NIFTI_URL}")
        urlretrieve(SAMPLE_NIFTI_URL, sample_path)  # nosec B310
    print(f"[sample] {sample_path} ({sample_path.stat().st_size} bytes)")
    return sample_path


def _save_mask(mask_xyz: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask_xyz.astype(np.uint8), affine=affine), str(out_path))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "tests"
    image_dir = tests_dir / "test_images"
    seg_out_dir = tests_dir / "segmentation_outputs"
    text_out_dir = tests_dir / "text_outputs"
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    text_out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if os.environ.get("THYQ_FORCE_CPU") != "1" else "cpu"

    sample_path = _ensure_sample(image_dir)
    img = nib.load(str(sample_path))

    fold_results: list[dict] = []

    for fold_id in range(5):
        print(f"[fold-{fold_id}] start")
        seg_model = tq.load_segmentor("trqseg_v1", members=[fold_id], ensemble=False, device=device)
        seg = tq.segment_trq(
            img,
            study_id=f"covid19-sample-fold-{fold_id}",
            segmentor=seg_model,
            device=device,
        )

        trq_mask = np.asarray(seg.trq_mask)
        _save_mask(trq_mask, img.affine, seg_out_dir / f"sample_fold{fold_id}_trq_mask.nii.gz")

        if seg.members and seg.members[0].airway_mask is not None:
            airway_mask = np.asarray(seg.members[0].airway_mask)
            _save_mask(airway_mask, img.affine, seg_out_dir / f"sample_fold{fold_id}_airway_mask.nii.gz")

        okamura = tq.quantify(seg, method="okamura")
        chaunzwa = tq.quantify(seg, method="chaunzwa")

        fold_result = {
            "fold_id": fold_id,
            "n_trq_voxels": int(trq_mask.sum()),
            "okamura": okamura.to_record(),
            "chaunzwa": chaunzwa.to_record(),
            "chaunzwa_gmm_components": [
                {
                    "component_id": c.component_id,
                    "mu_hu": c.mu_hu,
                    "weight": c.weight,
                    "group": (
                        "exclude"
                        if c.component_id in set(chaunzwa.exclude_component_ids)
                        else (
                            "adipose"
                            if c.component_id in set(chaunzwa.adipose_component_ids)
                            else "nonadipose"
                        )
                    ),
                }
                for c in (chaunzwa.gmm.components if chaunzwa.gmm is not None else [])
            ],
        }
        fold_results.append(fold_result)

        print(
            f"[fold-{fold_id}] trq_voxels={fold_result['n_trq_voxels']} "
            f"okamura_etv={fold_result['okamura'].get('etv_ml')} "
            f"chaunzwa_etv={fold_result['chaunzwa'].get('etv_ml')}"
        )

    summary = {
        "sample_path": str(sample_path),
        "device": device,
        "n_folds": 5,
        "results": fold_results,
    }

    summary_path = text_out_dir / "covid_sample_foldwise_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV summary for quick inspection / spreadsheet use
    import csv

    csv_path = text_out_dir / "covid_sample_foldwise_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "okamura_ptt",
                "okamura_etv_ml",
                "chaunzwa_ptt",
                "chaunzwa_etv_ml",
                "chaunzwa_gmm_mu_weight_group",
            ],
        )
        writer.writeheader()
        for r in fold_results:
            gmm_brief = "; ".join(
                [
                    f"c{c['component_id']}:{c['mu_hu']:.2f}/{c['weight']:.3f}:{c['group']}"
                    for c in r["chaunzwa_gmm_components"]
                ]
            )
            writer.writerow(
                {
                    "fold": r["fold_id"],
                    "okamura_ptt": r["okamura"].get("ptt"),
                    "okamura_etv_ml": r["okamura"].get("etv_ml"),
                    "chaunzwa_ptt": r["chaunzwa"].get("ptt"),
                    "chaunzwa_etv_ml": r["chaunzwa"].get("etv_ml"),
                    "chaunzwa_gmm_mu_weight_group": gmm_brief,
                }
            )

    print(f"[done] summary json -> {summary_path}")
    print(f"[done] summary csv  -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
