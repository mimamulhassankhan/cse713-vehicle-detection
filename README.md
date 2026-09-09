# Learn vehicle detection: training and testing

This project takes a traffic image and returns **vehicle boxes, types, confidence scores, and a CSV of pixel coordinates**. It uses YOLOv8n, an existing neural-network implementation. You train its weights further using UA-DETRAC; you do not need to implement the network yourself.

No lanes, movement, tracking, or violation classification are included. Object IDs belong to one image only.

## Start here

This repository contains the implementation, configurations, tests, notebook, and guides. Datasets, annotations, prepared data, model weights, generated predictions, and training runs are excluded from Git. Download UA-DETRAC and the initialization weights separately using the links below. Historical result paths in `STATUS.md` refer to local experiment artifacts and are not bundled with the repository.

When using a different dataset location, run `.\local.ps1 prepare -DatasetRoot 'D:\path\DETRAC Dataset'`. The download paths shown below are examples from the original workspace; replace them with your own paths.

**Local Windows workflow:** follow [LOCAL_RUN.md](LOCAL_RUN.md). Google Drive and Colab are optional. Run `.\local.ps1 check`, then `prepare`, `train`, `resume` if needed, `evaluate`, and `predict` as documented there.

1. Read [the learning guide](docs/LEARNING_GUIDE.md).
2. Use `local.ps1` for local training. The Colab notebook remains an optional alternative.
3. Run the two-epoch check before full training.
4. Download the trained `best.pt` and use the prediction command below.
5. Consult [the implementation status](STATUS.md) for what has actually run locally. A smoke checkpoint is not the completed full model.

## Local setup (Windows)

Open PowerShell in this project folder. Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

For this workspace, dependencies were installed in `D:\Edu\BracU\.venv`; you can use `..\.venv\Scripts\python.exe` immediately instead of creating another environment. All commands below use `python` for readability; substitute the environment's executable. `requirements-lock-windows.txt` records the exact locally tested environment. Colab uses its existing GPU-enabled PyTorch plus the pinned application dependencies in `requirements.txt`; do not install the Windows lock file in Colab.

Download the official initialization weights once:

```powershell
New-Item -ItemType Directory -Path models -Force
Invoke-WebRequest "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" -OutFile models/yolov8n.pt
```

Only load model files from trusted sources: PyTorch checkpoints may contain executable serialized objects.

## Dataset layout

```text
C:\Users\imamu\Downloads\DETRAC Dataset\
  DETRAC-Images\MVI_20011\img00001.jpg
  DETRAC-Train-Annotations-XML\MVI_20011.xml
  DETRAC-Test-Annotations-XML\...
```

MAT annotations are not needed because XML is easier to read and explain. All original files remain unchanged.

## 1. Prepare data

Full preparation makes a separate masked copy and labels. Allow substantial disk space (tens of GB) and time for the full collection. Use a **new empty output directory** for each preparation.

```powershell
python -m trafficvision prepare --root "C:\Users\imamu\Downloads\DETRAC Dataset" --output data/full
```

To inspect the dataset without copying images:

```powershell
python -m trafficvision prepare --root "C:\Users\imamu\Downloads\DETRAC Dataset" --output data/audit-new --audit-only
```

For a quick learning check (two spaced frames per sequence):

```powershell
python -m trafficvision prepare --root "C:\Users\imamu\Downloads\DETRAC Dataset" --output data/smoke-ready --stride 100 --limit 2
```

The default fixed seed is 713. The official 60 training sequences become **48 training and 12 validation**; all **40 official test sequences** remain test. These are sequence counts, not image percentages. The same sequence membership is reused in smoke and full preparation. Known camera-location grouping is not supplied by this converter, so sequence isolation does not guarantee unseen camera views.

Generated evidence:

- `splits.json`: exact sequence membership.
- `manifest.csv`: source image, prepared image, frame number, dimensions, and object counts.
- `report.json`: split sizes, class counts, missing data, and errors.
- `sequences.json`: original XML paths, ignored rectangles, and sequence attributes.
- `exclusions.jsonl`: objects removed under the documented masking policy.
- `data.yaml`, YOLO label files, and train/val/test image lists.

Frames absent from XML are **excluded**, not assumed to have no vehicles. Invalid boxes/classes and missing annotated images block release of `data.yaml` until resolved. True annotated empty frames remain valid negative examples.

## Ignored regions and evaluation limits

The ordinary YOLO training/evaluation API does not implement DETRAC's ignored regions. This project uses a deliberately explicit **custom masked-image protocol**:

1. Black out the XML ignored rectangles in copied images only.
2. If at least 50% of a target's clipped box is masked, remove its label and black out its remaining box.
3. Repeat until no additional box crosses that threshold.
4. Retain partially visible targets below the threshold with their full clipped boxes.

The same transformation is used for training, validation, and testing. This changes the data and creates artificial black regions; it is a learning baseline, **not official DETRAC benchmark evaluation**. Occlusion/truncation attributes are not predicted. No extra truncation filter is imposed. XML coordinates are used as provided and clipped to `[0,width]` / `[0,height]`; fractional values are preserved, with no unverified one-pixel offset.

Prediction on a new raw image does not require XML and does not mask it. Quality on raw images can differ from the masked evaluation results.

## 2. Train

Run the execution check:

```powershell
python -m trafficvision train --config configs/smoke.yaml
```

Then run full training on a GPU (recommended in Colab):

```powershell
python -m trafficvision train --config configs/train.yaml
```

The full run starts from the original pretrained weights, **not** the tiny smoke checkpoint. Defaults: image size 640, up to 50 epochs, AdamW optimizer, initial learning rate 0.001, early-stopping patience 10, automatic GPU batch sizing, and seed 713. CPU uses batch 4. Training augmentation is managed by YOLO; validation/test are not augmented.

Training outputs include `weights/best.pt`, `weights/last.pt`, `results.csv`, `results.png`, validation plots, configuration, and provenance. `best.pt` is selected from validation metrics, never from test results. The validation set is also used to observe training progress; it is not a second training set.

To resume an interrupted run:

```powershell
python -m trafficvision resume --weights runs/detrac_full/weights/last.pt
```

Resume requires the same accessible prepared-data paths. Use a new run name in the config for a new experiment; existing runs are never silently overwritten.

## 3. Evaluate

After settings and checkpoint selection are finished:

```powershell
python -m trafficvision evaluate --weights runs/detrac_full/weights/best.pt --data data/full/data.yaml --split test --output runs/final_test
```

`metrics.json` records precision, recall, mAP50, mAP50-95, per-class results, checkpoint, and data coverage. Precision/recall use the validator's F1 operating point, not necessarily the `.25` threshold used for display. Classes without references may be absent from per-class AP results. Do not tune on final test results.

Compare pretrained and fine-tuned models on the **validation** split at identical fixed thresholds:

```powershell
python -m trafficvision compare --baseline models/yolov8n.pt --trained runs/detrac_full/weights/best.pt --data data/full/data.yaml --split val --output runs/car_bus_comparison.json
```

Only car and bus are compared because COCO's label set does not match DETRAC's van/others categories. This is conditional precision/recall at confidence .25 and matching IoU .5; predictions overlapping unsupported reference categories at IoU >= .5 are excluded. It is not an AP comparison, and exclusions are reported. Four-class custom-model evaluation remains separate.

## 4. Analyze an image

```powershell
python -m trafficvision predict --image "C:\path\traffic.jpg" --weights runs/detrac_full/weights/best.pt --output outputs/my_frame
```

Outputs:

- `annotated.png`: predicted boxes, vehicle names, IDs, confidence, and center dots.
- `vehicles.csv`: one row per vehicle.
- `prediction.json`: image size, checkpoint, actual model classes, and threshold.

CSV columns:

```text
image_id,object_id,vehicle_type,confidence,xmin,ymin,xmax,ymax,center_x,center_y
```

The origin is the top-left of the original image. X increases to the right; Y increases downward. Centers are `(xmin+xmax)/2` and `(ymin+ymax)/2`. The right/bottom box extent can equal image width/height; drawing clips to the last pixel. No detections still yields the annotated image and a CSV header.

Use `--conf 0.4` to display fewer, more confident predictions. Confidence is a model score, not a guarantee of correctness. With baseline COCO weights, retain actual class names such as truck; never pretend COCO predicts DETRAC-specific vans.

## Files to understand

- `trafficvision/data.py`: XML conversion, splitting, masking, and manifests.
- `trafficvision/model.py`: training, evaluation, drawing, and CSV output.
- `trafficvision/comparison.py`: fair limited car/bus comparison.
- `tests/test_workflow.py`: checks for data leakage and output correctness.

## Sources and attribution

- [UA-DETRAC author page and downloads](https://sites.google.com/view/daweidu/projects/ua-detrac)
- [UA-DETRAC paper, Wen et al., CVIU 2020](https://faculty.ucmerced.edu/mhyang/papers/cviu2020_detrac.pdf)
- [YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8/)
- [Ultralytics training](https://docs.ultralytics.com/modes/train/) and [validation](https://docs.ultralytics.com/modes/val/)

The official dataset is a benchmark, not a detector called DETRAC. This project reuses its labels and split definitions and trains an existing YOLO implementation. No official DETRAC toolkit source is copied. Ultralytics licensing and dataset terms remain applicable; consult their sources before redistributing data, weights, or a commercial application.
