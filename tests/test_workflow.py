import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from trafficvision.data import clip_box, yolo_box, sequence_split, ignore_mask, prepare
from trafficvision.model import detection_rows, save_prediction, predict, FIELDS
from trafficvision.comparison import score


def test_boxes():
    box = clip_box(dict(left=-10, top=20, width=60, height=100), 100, 80)
    assert box == [0, 20, 50, 80]
    assert yolo_box(box, 100, 80) == [.25, .625, .5, .75]
    with pytest.raises(ValueError):
        clip_box(dict(left=0, top=0, width=-2, height=10), 100, 80)


def test_splits():
    result = sequence_split([f't{i}' for i in range(60)], [f'e{i}' for i in range(40)])
    assert [len(result[s]) for s in ('train', 'val', 'test')] == [48, 12, 40]
    assert not set(result['train']) & set(result['val'])
    assert result == sequence_split([f't{i}' for i in range(60)], [f'e{i}' for i in range(40)])
    with pytest.raises(ValueError):
        sequence_split(['a', 'b'], ['a'])


def test_ignore_regions_union_and_closure():
    targets = [{'box': [0, 0, 10, 10]}, {'box': [15, 15, 20, 20]}]
    mask, kept, dropped = ignore_mask(20, 20, [[0, 0, 3, 10], [3, 0, 6, 10]], targets)
    assert dropped == [0]
    assert kept == [targets[1]]
    assert mask[:10, :10].all()


def fixture_dataset(root):
    for name, part in [('MVI_1', 'Train'), ('MVI_2', 'Train'), ('MVI_3', 'Test')]:
        folder = root/'DETRAC-Images'/name
        folder.mkdir(parents=True)
        Image.new('RGB', (100, 80), 'white').save(folder/'img00001.jpg')
        annotations = root/f'DETRAC-{part}-Annotations-XML'
        annotations.mkdir(exist_ok=True)
        (annotations/f'{name}.xml').write_text(f'<sequence name="{name}"><ignored_region/>'
            '<frame num="1"><target_list><target id="1"><box left="10" top="20" width="30" height="40"/>'
            '<attribute vehicle_type="car"/></target></target_list></frame></sequence>')


def test_end_to_end_preparation(tmp_path):
    original = tmp_path/'source'
    fixture_dataset(original)
    image = original/'DETRAC-Images/MVI_1/img00001.jpg'
    before = image.read_bytes()
    out = tmp_path/'prepared'
    report = prepare(original, out)
    assert report['errors'] == []
    assert image.read_bytes() == before
    assert (out/'data.yaml').exists()
    for part in ('train', 'val', 'test'):
        path = Path((out/f'{part}.txt').read_text().strip())
        assert path.exists()
        label = out/'labels'/part/(path.stem+'.txt')
        assert list(map(float, label.read_text().split())) == [0, .25, .5, .3, .5]
    with pytest.raises(ValueError):
        prepare(original, out)


def test_unknown_class_blocks_training_config(tmp_path):
    root = tmp_path/'source'
    fixture_dataset(root)
    xml = root/'DETRAC-Train-Annotations-XML/MVI_1.xml'
    xml.write_text(xml.read_text().replace('vehicle_type="car"', 'vehicle_type="airplane"'))
    out = tmp_path/'prepared'
    with pytest.raises(ValueError, match='errors'):
        prepare(root, out)
    assert not (out/'data.yaml').exists()
    assert json.loads((out/'report.json').read_text())['errors']


def test_unannotated_image_is_excluded_not_negative(tmp_path):
    root = tmp_path/'source'
    fixture_dataset(root)
    Image.new('RGB', (100, 80), 'white').save(root/'DETRAC-Images/MVI_1/img00002.jpg')
    out = tmp_path/'prepared'
    report = prepare(root, out)
    assert report['unannotated_images'] == ['MVI_1/img00002.jpg']
    assert len(list((out/'images').rglob('*.jpg'))) == 3


def test_missing_annotated_image_blocks_training_config(tmp_path):
    root = tmp_path/'source'
    fixture_dataset(root)
    xml = root/'DETRAC-Train-Annotations-XML/MVI_1.xml'
    xml.write_text(xml.read_text().replace('num="1"', 'num="2"'))
    out = tmp_path/'prepared'
    with pytest.raises(ValueError, match='errors'):
        prepare(root, out)
    assert not (out/'data.yaml').exists()


def test_prediction_records_and_image(tmp_path):
    rows = detection_rows('frame', [('car', .9, [-2, 10, 80, 60]), ('bus', .8, [0, 0, 0, 3])], 100, 80)
    assert len(rows) == 1
    assert rows[0]['center_x'] == 40 and rows[0]['center_y'] == 35
    save_prediction(np.zeros((80, 100, 3), dtype=np.uint8), rows, tmp_path/'out', {})
    with (tmp_path/'out/vehicles.csv').open() as f:
        saved = list(csv.DictReader(f))
    assert saved[0]['object_id'] == 'V001'
    assert Image.open(tmp_path/'out/annotated.png').size == (100, 80)
    assert np.array(Image.open(tmp_path/'out/annotated.png')).sum() > 0


def test_empty_prediction(tmp_path):
    save_prediction(np.zeros((10, 20, 3), dtype=np.uint8), [], tmp_path/'empty', {})
    with (tmp_path/'empty/vehicles.csv').open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS
        assert list(reader) == []


def test_invalid_input_before_model_load(tmp_path):
    with pytest.raises(ValueError, match='not found'):
        predict(tmp_path/'missing.jpg', 'missing.pt', tmp_path/'out')
    bad = tmp_path/'bad.jpg'
    bad.write_text('not an image')
    with pytest.raises(ValueError, match='Invalid'):
        predict(bad, 'missing.pt', tmp_path/'out')


def test_comparison_matches_once():
    result = score([('car', .9, [0, 0, 10, 10]), ('car', .8, [0, 0, 10, 10]),
                    ('bus', .7, [20, 20, 30, 30])], [('car', [0, 0, 10, 10])], [[20, 20, 30, 30]])
    assert result['car']['tp'] == 1 and result['car']['fp'] == 1
    assert result['bus']['excluded'] == 1


def test_preparation_resume_reuses_and_repairs(tmp_path):
    root = tmp_path/'source'
    fixture_dataset(root)
    out = tmp_path/'prepared'
    prepare(root, out)
    completed = sorted((out/'images').rglob('*.jpg'))
    unchanged_time = completed[0].stat().st_mtime_ns
    completed[1].write_bytes(b'interrupted image')
    prepare(root, out, resume=True)
    assert completed[0].stat().st_mtime_ns == unchanged_time
    assert Image.open(completed[1]).size == (100, 80)
    with pytest.raises(ValueError, match='Cannot resume'):
        prepare(root, out, seed=42, resume=True)
    source = root/'DETRAC-Images/MVI_1/img00001.jpg'
    Image.new('RGB', (100, 80), 'red').save(source)
    with pytest.raises(ValueError, match='Cannot resume'):
        prepare(root, out, resume=True)
