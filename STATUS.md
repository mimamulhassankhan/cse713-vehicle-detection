# Implementation and experiment status

## Publication snapshot - 9 September 2026

The code repository excludes datasets, weights and generated experiment artifacts. File paths and results below describe local experiments, not files included in a fresh clone. Local full-run initialization artifacts now exist under `runs/detrac_local` (arguments and training-batch images), but no completed full-training metrics or final test evaluation were available at publication. Initialization files do not establish that a training process is currently active. The older setup notes below are retained as historical context.

## Current workflow: local training

Google Drive capacity is insufficient, so full training will now run locally. Use `LOCAL_RUN.md` and `local.ps1`; the Colab workflow below is historical/optional. Local preparation is resumable and uses `data/local-full`; training uses `configs/local.yaml` and writes `runs/detrac_local`. Full local training has not yet been launched. Existing source images, archives, and older preparation folders are retained. GitHub publication remains a later step.

## Completed locally

- Python preparation, training, resume, evaluation, comparison, and prediction commands.
- Guided Colab notebook, simple-English learning guide, pinned dependencies, and automated tests.
- Complete dataset audit: 100 image sequences, 140,131 JPEG files; 60 official training and 40 official test annotations.
- Excluded 1,879 image files without XML frame entries. No missing annotated images or invalid annotations were found in the full audit.
- Full-data split manifests and annotation counts generated under `data/audit`.
- Two-epoch custom training check: 96 training images, 24 validation images, two spaced frames per sequence.
- Custom smoke weights, training curves, validation evaluation, car/bus comparison, and image/CSV examples.

| Split | Sequences | Annotated images | Retained car boxes | Bus | Van | Others |
|---|---:|---:|---:|---:|---:|---:|
| Training | 48 | 66,043 | 412,145 | 23,617 | 46,926 | 2,751 |
| Validation | 12 | 16,042 | 70,458 | 9,018 | 7,029 | 780 |
| Testing | 40 | 56,167 | 507,049 | 64,844 | 33,445 | 16,186 |

These counts use the README's custom masking/exclusion policy. An additional 22,733 training, 2,824 validation, and 54,250 testing target boxes are excluded by that policy. This is not the official DETRAC benchmark protocol.

## Actual smoke results — not a finished model

On **24 validation images**, the two-epoch checkpoint achieved mAP50 **0.2457** and mAP50–95 **0.1577**. No `others` references occur in this tiny validation sample, so the aggregate covers the represented classes only. Van AP is zero.

At the display confidence threshold 0.25, the smoke checkpoint produced **no car/bus detections** in the comparison. The pretrained model performed better. A new four-class output head and only 96 training images over two epochs are insufficient to claim a useful trained detector. Lowering the display threshold illustrates low-confidence boxes, not improved quality.

Files:

- `runs/detrac_smoke/weights/best.pt`: actual custom-trained smoke checkpoint.
- `runs/detrac_smoke/results.csv` and `results.png`: learning history.
- `runs/smoke_validation/metrics.json`: actual validation metrics.
- `runs/smoke_car_bus_comparison.json`: same-frame conditional comparison.
- `outputs/pretrained_example`: pretrained predictions, not custom-training results.
- `outputs/smoke_example`: custom checkpoint at confidence 0.25, zero detections.
- `outputs/smoke_low_confidence_example`: custom checkpoint at confidence 0.03, explicitly a low-confidence diagnostic.

## Required work remaining

**Full fine-tuning and final test evaluation have not run.** The available local PyTorch environment is CPU-only. The full 66,043-image training split is now supported by the local launcher; runtime will be substantially longer on CPU.

The full local audit is complete. The large local masked-image copy was stopped and placed in `data/full-incomplete`; it has no released `data.yaml` and must not be used for training. The notebook prepares a complete copy in Colab instead. No process continues training in the background.

Optional previous Colab instructions (use LOCAL_RUN.md for the current workflow):

1. Upload `vehicle-detection-project.zip` and `DETRAC_Dataset.zip` to your own Google Drive.
2. Open `notebooks/Train_and_Test_DETRAC.ipynb` in Colab and select a GPU runtime.
3. Authorize Drive access yourself, then run the notebook in order.
4. Complete full training, comparison on validation, and final test evaluation.
5. Download `best.pt` and retain the saved curves and test metrics from Drive.

The test images have been audited/prepared but **no model test-set evaluation has been performed**. They remain reserved for the final experiment. The Colab notebook is syntax-checked but has not been executed in a Colab account. No dataset was uploaded to an external service.
