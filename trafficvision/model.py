"""Training, held-out evaluation, and image prediction with an existing YOLO implementation."""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from .data import NAMES, write_json

FIELDS = ['image_id', 'object_id', 'vehicle_type', 'confidence', 'xmin', 'ymin',
          'xmax', 'ymax', 'center_x', 'center_y']


def check_weights(weights):
    p = Path(weights)
    if not p.is_file():
        raise ValueError(f'Weights not found: {p}. Download initialization with the setup instructions or train first.')
    return str(p.resolve())


def load_model(weights):
    from ultralytics import YOLO
    return YOLO(check_weights(weights))


def device_name(requested):
    import torch
    return ('0' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested


def data_config(path):
    import yaml
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError(f'Dataset configuration not found: {p}')
    config = yaml.safe_load(p.read_text(encoding='utf-8'))
    if config.get('names') != NAMES:
        raise ValueError('Expected prepared DETRAC class order: car, bus, van, others')
    root = Path(config['path'])
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    if report['errors'] or report['audit_only']:
        raise ValueError('Dataset audit is incomplete or contains errors')
    for part in ('train', 'val', 'test'):
        if not (root/config[part]).read_text(encoding='utf-8').strip():
            raise ValueError(f'Empty {part} split')
    return str(p), config, report


def train(config_path, data=None, weights=None, epochs=None, device=None, batch=None):
    import yaml
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    for key, value in [('data', data), ('weights', weights), ('epochs', epochs), ('device', device), ('batch', batch)]:
        if value is not None:
            config[key] = value
    data, _, report = data_config(config['data'])
    model = load_model(config['weights'])
    target = Path(config.get('project', 'runs'))/config.get('name', 'training')
    if target.exists():
        raise ValueError(f'Run exists: {target}; choose a fresh name or use resume')
    device = device_name(config.get('device', 'auto'))
    batch = config.get('batch', -1)
    if device == 'cpu' and batch == -1:
        batch = 4
    # Each epoch learns from train only. Validation chooses a checkpoint; test is not read.
    model.train(data=data, epochs=config.get('epochs', 50), imgsz=640,
                batch=batch, device=device, workers=0, seed=713, deterministic=True,
                optimizer='AdamW', lr0=.001, patience=config.get('patience', 10),
                project=str(target.parent.resolve()), name=target.name, exist_ok=False,
                plots=True, save=True, save_period=1, cache=False, amp=device != 'cpu',
                close_mosaic=min(10, config.get('epochs', 50)), resume=False)
    directory = Path(model.trainer.save_dir)
    write_json(directory/'training_provenance.json', {'configuration': config, 'dataset_report': report,
               'initial_weights': str(Path(config['weights']).resolve()),
               'best_checkpoint': str(directory/'weights/best.pt'),
               'warning': 'Short smoke training proves execution, not model quality.'})
    return str(directory)


def resume(weights, device='auto'):
    model = load_model(weights)
    model.train(resume=True, device=device_name(device))


def evaluate(weights, data, split='test', output='runs/evaluation', device='auto'):
    data, _, report = data_config(data)
    out = Path(output)
    if out.exists():
        raise ValueError(f'Evaluation output already exists: {out}')
    model = load_model(weights)
    if list(model.names.values()) != NAMES:
        raise ValueError('Four-class DETRAC evaluation requires a custom-trained checkpoint; use compare for COCO baseline')
    results = model.val(data=data, split=split, imgsz=640, device=device_name(device),
                        batch=4, workers=0, plots=True, conf=.001, iou=.7,
                        project=str(out.parent.resolve()), name=out.name, exist_ok=False)
    actual = Path(results.save_dir)
    metrics = {'weights': str(Path(weights).resolve()), 'split': split,
               'protocol': 'Custom masked-image YOLO evaluation; not official DETRAC AP',
               'dataset_report': report, 'overall': {k: float(v) for k, v in results.results_dict.items()},
               'per_class': []}
    # precision/recall are at the validator-selected F1 operating point.
    for i, cls in enumerate(results.box.ap_class_index):
        p, r, ap50, ap = results.box.class_result(i)
        metrics['per_class'].append({'class': NAMES[int(cls)], 'precision': float(p),
            'recall': float(r), 'mAP50': float(ap50), 'mAP50_95': float(ap)})
    write_json(actual/'metrics.json', metrics)
    return metrics


def detection_rows(image_id, detections, width, height):
    """Clip to the original image and calculate centers from those same boxes."""
    rows = []
    for name, confidence, box in detections:
        if not all(math.isfinite(float(v)) for v in [confidence, *box]):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        values = [image_id, f'V{len(rows)+1:03d}', name, float(confidence),
                  x1, y1, x2, y2, (x1+x2)/2, (y1+y2)/2]
        rows.append(dict(zip(FIELDS, values)))
    return rows


def save_prediction(image, rows, output, metadata):
    import cv2
    import numpy as np
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Use a new empty prediction output folder: {out}')
    out.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    h, w = image.shape[:2]
    for row in rows:
        a = (min(w-1, round(row['xmin'])), min(h-1, round(row['ymin'])))
        b = (min(w-1, round(row['xmax'])), min(h-1, round(row['ymax'])))
        color = (50, 220, 50)
        cv2.rectangle(annotated, a, b, color, 2)
        label = f"{row['object_id']} {row['vehicle_type']} {row['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)
        tx = max(0, min(a[0], w-tw-2))
        ty = min(h-baseline-1, max(th+2, a[1]-5))
        cv2.rectangle(annotated, (tx, ty-th-2), (min(w-1, tx+tw+1), ty+baseline), (20, 20, 20), -1)
        cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        cv2.circle(annotated, (min(w-1, round(row['center_x'])), min(h-1, round(row['center_y']))), 3, (0, 200, 255), -1)
    ok, buffer = cv2.imencode('.png', annotated)
    if not ok:
        raise ValueError('Could not encode annotated image')
    buffer.tofile(str(out/'annotated.png'))
    with (out/'vehicles.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out/'prediction.json', {**metadata, 'width': w, 'height': h, 'detections': len(rows),
                                      'coordinates': 'original image pixels; top-left origin'})


def predict(image, weights, output, conf=.25, device='auto'):
    import cv2
    import numpy as np
    image = Path(image)
    if not image.is_file():
        raise ValueError(f'Input image not found: {image}')
    if not 0 < conf <= 1:
        raise ValueError('Confidence must be greater than 0 and at most 1')
    pixels = cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f'Invalid or unsupported image: {image}')
    model = load_model(weights)
    # COCO vans are not relabeled as a DETRAC van: retain the actual model class.
    accepted = {'car', 'bus', 'van', 'others', 'truck', 'motorcycle'}
    classes = [i for i, name in model.names.items() if name in accepted]
    if not classes:
        raise ValueError('Checkpoint does not contain supported vehicle classes')
    result = model.predict(pixels, conf=conf, imgsz=640, classes=classes,
                           device=device_name(device), verbose=False)[0]
    detections = [(model.names[int(b.cls.item())], b.conf.item(), b.xyxy[0].tolist()) for b in result.boxes]
    rows = detection_rows(image.stem, detections, pixels.shape[1], pixels.shape[0])
    save_prediction(pixels, rows, output, {'image': str(image.resolve()),
                    'weights': str(Path(weights).resolve()), 'model_classes': model.names,
                    'confidence_threshold': conf, 'source': 'model predictions, not XML annotations'})
    return rows
