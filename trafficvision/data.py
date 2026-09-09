"""Convert DETRAC XML labels without modifying the downloaded dataset.

Only this module prepares training data. Prediction never reads XML answers.
"""
import csv
import json
import math
import random
import shutil
import hashlib
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

NAMES = ['car', 'bus', 'van', 'others']


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def clip_box(box, width, height):
    x, y, w, h = [float(box[k]) for k in ('left', 'top', 'width', 'height')]
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f'Invalid box: {box}')
    result = [max(0., x), max(0., y), min(float(width), x+w), min(float(height), y+h)]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f'Box outside image: {box}')
    return result


def yolo_box(box, width, height):
    x1, y1, x2, y2 = box
    return [(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height]


def sequence_split(train, test, seed=713):
    if set(train) & set(test):
        raise ValueError('A sequence occurs in both official splits')
    if len(train) < 2 or not test:
        raise ValueError('Need at least two training sequences and one test sequence')
    shuffled = sorted(train)
    random.Random(seed).shuffle(shuffled)
    n = max(1, min(len(train)-1, round(len(train)*.2)))
    return {'train': sorted(shuffled[n:]), 'val': sorted(shuffled[:n]), 'test': sorted(test)}


def rectangles(boxes, width, height):
    return [(max(0, math.floor(b[0])), max(0, math.floor(b[1])),
             min(width, math.ceil(b[2])), min(height, math.ceil(b[3]))) for b in boxes]


def ignore_mask(width, height, ignored, targets):
    """Custom masked-image protocol, NOT the official DETRAC evaluator.

    Black out ignored areas. Drop a target if >=50% of its box is masked,
    and black out its remaining pixels too. Repeat until no more targets
    qualify, so removed objects cannot become visible unlabeled examples.
    Partially visible retained targets retain their full clipped boxes.
    """
    import numpy as np
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in rectangles(ignored, width, height):
        mask[y1:y2, x1:x2] = True
    rects = rectangles([t['box'] for t in targets], width, height)
    dropped = set()
    while True:
        newly = []
        for i, (x1, y1, x2, y2) in enumerate(rects):
            if i not in dropped and mask[y1:y2, x1:x2].mean() >= .5:
                newly.append(i)
        if not newly:
            break
        for i in newly:
            dropped.add(i)
            x1, y1, x2, y2 = rects[i]
            mask[y1:y2, x1:x2] = True
    return mask, [t for i, t in enumerate(targets) if i not in dropped], sorted(dropped)


def write_yaml(out, splits):
    # JSON is valid YAML and avoids platform-specific path escaping.
    config = {'path': str(out.resolve()), **{s: f'{s}.txt' for s in splits}, 'names': NAMES}
    write_json(out/'data.yaml', config)


def prepare(root, output, seed=713, stride=1, limit=0, audit_only=False, resume=False):
    from PIL import Image
    import numpy as np
    root, out = Path(root).resolve(), Path(output).resolve()
    if out == root or root in out.parents:
        raise ValueError('Output must be outside the original dataset')
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError('Use a new empty output folder; existing preparation is never overwritten')
    if stride < 1 or limit < 0:
        raise ValueError('stride must be positive and limit nonnegative')
    images = root/'DETRAC-Images'
    xmls = {}
    official = {}
    for split, folder in [('train', 'Train'), ('test', 'Test')]:
        files = sorted((root/f'DETRAC-{folder}-Annotations-XML').rglob('*.xml'))
        if not files:
            raise ValueError(f'No {folder} XML annotations under {root}')
        official[split] = [f.stem for f in files]
        for f in files:
            if f.stem in xmls:
                raise ValueError(f'Duplicate annotation sequence: {f.stem}')
            xmls[f.stem] = f
    available = {p.name for p in images.iterdir() if p.is_dir() and any(p.glob('*.jpg'))}
    split = sequence_split([s for s in official['train'] if s in available],
                           [s for s in official['test'] if s in available], seed)
    # A resume must use exactly the same inputs/settings. Image metadata catches
    # replacement of an original frame without hashing all image pixels again.
    digest = hashlib.sha256()
    for sequence in sorted(xmls):
        digest.update(sequence.encode())
        digest.update(xmls[sequence].read_bytes())
    for sequence in sorted(available):
        for image in sorted((images/sequence).glob('*.jpg')):
            stat = image.stat()
            digest.update(f'{sequence}/{image.name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    settings = dict(root=str(root), seed=seed, stride=stride, limit=limit,
                    audit_only=audit_only, protocol_version=1, source_signature=digest.hexdigest())
    marker = out/'preparation-settings.json'
    if resume:
        if not marker.exists() or json.loads(marker.read_text(encoding='utf-8')) != settings:
            raise ValueError('Cannot resume: source/settings differ or this folder predates resumable preparation. Use a new output folder.')
    out.mkdir(parents=True, exist_ok=True)
    write_json(marker, settings)
    # Never expose a potentially incomplete preparation to training.
    (out/'data.yaml').unlink(missing_ok=True)
    write_json(out/'splits.json', split)
    report = {'source': str(root), 'seed': seed, 'stride': stride, 'limit_per_sequence': limit,
              'audit_only': audit_only, 'missing_image_sequences': sorted(set(xmls)-available),
              'unannotated_sequences': sorted(available-set(xmls)), 'errors': [], 'unannotated_images': [],
              'protocol': 'custom masked images, 50% ignored-area exclusion; not official DETRAC',
              'coordinate_policy': 'XML pixels used as supplied, clipped to image extent; no implicit 1-pixel shift',
              'splits': {}}
    sequence_info = {}
    fields = ['split', 'sequence', 'frame', 'source_image', 'prepared_image', 'width', 'height',
              'original_targets', 'retained_targets', 'dropped_targets']
    with (out/'manifest.csv').open('w', newline='', encoding='utf-8') as mf, \
            (out/'exclusions.jsonl').open('w', encoding='utf-8') as ef:
        writer = csv.DictWriter(mf, fieldnames=fields)
        writer.writeheader()
        for part, sequences in split.items():
            counts = Counter()
            frames = dropped_count = missing = 0
            paths = []
            for sequence in sequences:
                tree = ET.parse(xmls[sequence]).getroot()
                if tree.get('name') != sequence:
                    raise ValueError(f'XML sequence name mismatch: {sequence}')
                raw_ignore = [b.attrib for b in tree.findall('./ignored_region/box')]
                sequence_info[sequence] = {'xml': str(xmls[sequence]), 'official_split':
                    'train' if sequence in official['train'] else 'test',
                    'attributes': tree.find('sequence_attribute').attrib if tree.find('sequence_attribute') is not None else {},
                    'ignored_regions': raw_ignore}
                seen = set()
                selected = 0
                for frame in tree.findall('frame'):
                    number = int(frame.get('num'))
                    if number in seen:
                        raise ValueError(f'Duplicate frame {sequence}/{number}')
                    seen.add(number)
                    image = images/sequence/f'img{number:05d}.jpg'
                    if not image.exists():
                        missing += 1
                        report['errors'].append(f'Missing image {image}')
                        continue
                    if (number-1) % stride or (limit and selected >= limit):
                        continue
                    selected += 1
                    with Image.open(image) as src:
                        width, height = src.size
                        ignored = []
                        for b in raw_ignore:
                            try:
                                ignored.append(clip_box(b, width, height))
                            except ValueError as error:
                                report['errors'].append(f'{sequence} ignore region: {error}')
                        targets = []
                        for t in frame.findall('./target_list/target'):
                            attribute = t.find('attribute')
                            try:
                                name = attribute.get('vehicle_type').strip().lower()
                                if name == 'other':
                                    name = 'others'
                                if name not in NAMES:
                                    raise ValueError(f'Unknown class {name}')
                                box = clip_box(t.find('box').attrib, width, height)
                            except (ValueError, AttributeError) as error:
                                report['errors'].append(f'{sequence}/{number}/{t.get("id")}: {error}')
                                continue
                            targets.append({'id': t.get('id'), 'class': NAMES.index(name), 'box': box})
                        mask, kept, dropped = ignore_mask(width, height, ignored, targets)
                        name = f'{sequence}__img{number:05d}'
                        dest = out/'images'/part/(name+'.jpg')
                        if not audit_only:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            label = out/'labels'/part/(name+'.txt')
                            label.parent.mkdir(parents=True, exist_ok=True)
                            label_text = ''.join(f'{t["class"]} '+ ' '.join(f'{v:.8f}' for v in yolo_box(t['box'], width, height))+'\n' for t in kept)
                            reusable = False
                            if resume and dest.exists() and label.exists() and label.read_text(encoding='utf-8') == label_text:
                                try:
                                    with Image.open(dest) as completed:
                                        reusable = completed.size == (width, height)
                                        completed.verify()
                                except (OSError, ValueError):
                                    reusable = False
                            if not reusable:
                                pixels = np.array(src.convert('RGB'))
                                pixels[mask] = 0
                                temporary = dest.with_suffix('.partial.jpg')
                                Image.fromarray(pixels).save(temporary, quality=95)
                                temporary.replace(dest)
                                label.write_text(label_text, encoding='utf-8')
                            paths.append(str(dest.resolve()))
                        counts.update(NAMES[t['class']] for t in kept)
                        frames += 1
                        dropped_count += len(dropped)
                        writer.writerow(dict(zip(fields, [part, sequence, number, str(image),
                            str(dest) if not audit_only else '', width, height, len(targets), len(kept), len(dropped)])))
                        if dropped:
                            ef.write(json.dumps({'sequence': sequence, 'frame': number,
                                                 'excluded_targets': [targets[i] for i in dropped]})+'\n')
                extra = sorted(p.name for p in (images/sequence).glob('*.jpg') if int(p.stem[3:]) not in seen)
                # Unannotated frames are excluded, never silently turned into negative labels.
                report['unannotated_images'].extend(f'{sequence}/{p}' for p in extra)
                print(f'{part}: {sequence}, {selected} frames checked', flush=True)
            report['splits'][part] = {'sequences': len(sequences), 'images': frames,
                'class_counts': dict(counts), 'excluded_targets': dropped_count, 'missing_images': missing}
            if not audit_only:
                (out/f'{part}.txt').write_text('\n'.join(paths)+'\n', encoding='utf-8')
    write_json(out/'sequences.json', sequence_info)
    write_json(out/'report.json', report)
    if report['errors']:
        raise ValueError(f'Preparation has {len(report["errors"])} errors. Inspect {out}/report.json; no data.yaml released')
    if not audit_only:
        write_yaml(out, split)
    return report
