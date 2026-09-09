"""Compare only car and bus at fixed thresholds, without equating COCO van/truck classes.

This is a conditional learning comparison, not COCO AP or official DETRAC AP.
Predictions overlapping an unsupported reference category are excluded. This
avoids punishing a COCO detector for DETRAC-specific van/others definitions.
"""
from collections import Counter
from pathlib import Path
from .data import NAMES, write_json
from .model import load_model, data_config, device_name


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def score(predictions, targets, ignored):
    counts = {name: Counter(tp=0, fp=0, fn=0, excluded=0) for name in ('car', 'bus')}
    used = set()
    for name, confidence, box in sorted(predictions, key=lambda p: p[1], reverse=True):
        candidates = [(iou(box, gtbox), i) for i, (gtname, gtbox) in enumerate(targets)
                      if gtname == name and i not in used]
        overlap, index = max(candidates, default=(0, -1))
        if overlap >= .5:
            counts[name]['tp'] += 1
            used.add(index)
        elif any(iou(box, gtbox) >= .5 for gtbox in ignored):
            counts[name]['excluded'] += 1
        else:
            counts[name]['fp'] += 1
    for i, (name, _) in enumerate(targets):
        if i not in used:
            counts[name]['fn'] += 1
    return counts


def compare(baseline, trained, data, split='val', output='runs/comparison.json', device='auto'):
    _, config, report = data_config(data)
    output = Path(output)
    if output.exists():
        raise ValueError(f'Comparison already exists: {output}')
    paths = (Path(config['path'])/config[split]).read_text(encoding='utf-8').splitlines()
    result = {'split': split, 'images': len(paths), 'confidence': .25, 'matching_IoU': .5,
              'scope': 'Conditional car/bus comparison on custom masked images; predictions matching van/others at IoU >= .5 excluded',
              'models': {}}
    for label, weights in [('pretrained', baseline), ('fine_tuned', trained)]:
        model = load_model(weights)
        common = [i for i, name in model.names.items() if name in ('car', 'bus')]
        if len(common) != 2:
            raise ValueError('Both checkpoints must contain car and bus classes')
        totals = {name: Counter(tp=0, fp=0, fn=0, excluded=0) for name in ('car', 'bus')}
        # One frame at a time avoids loading the entire image collection into RAM.
        for index, path in enumerate(paths):
            prediction = model.predict(path, conf=.25, imgsz=640, classes=common,
                                       device=device_name(device), verbose=False)[0]
            h, w = prediction.orig_shape
            image = Path(path)
            labels = image.parent.parent.parent/'labels'/image.parent.name/(image.stem+'.txt')
            targets, ignored = [], []
            for line in labels.read_text(encoding='utf-8').splitlines():
                cls, x, y, bw, bh = map(float, line.split())
                box = [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]
                name = NAMES[int(cls)]
                if name in totals:
                    targets.append((name, box))
                else:
                    ignored.append(box)
            predictions = [(model.names[int(b.cls.item())], b.conf.item(), b.xyxy[0].tolist()) for b in prediction.boxes]
            for name, value in score(predictions, targets, ignored).items():
                totals[name].update(value)
            if (index+1) % 100 == 0:
                print(f'{label}: {index+1}/{len(paths)}', flush=True)
        result['models'][label] = {'weights': str(Path(weights).resolve()), 'classes': {}}
        for name, count in totals.items():
            p = count['tp']/(count['tp']+count['fp']) if count['tp']+count['fp'] else 0.
            r = count['tp']/(count['tp']+count['fn']) if count['tp']+count['fn'] else 0.
            result['models'][label]['classes'][name] = {**count, 'precision': p, 'recall': r,
                                                       'f1': 2*p*r/(p+r) if p+r else 0.}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
