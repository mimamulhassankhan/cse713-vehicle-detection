"""Regenerate the guided, unexecuted Colab notebook."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.strip().splitlines(True)})


md('''# Learn vehicle detection: UA-DETRAC + YOLOv8n
Train your own fine-tuned detector, validate it, then test it. No lanes or tracking.

**Before starting:** choose Runtime → Change runtime type → GPU. Upload
`vehicle-detection-project.zip` and `DETRAC_Dataset.zip` to your own Google Drive.
The project archive contains code only; create the dataset archive locally using
`python scripts/package_dataset.py --root "C:/Users/imamu/Downloads/DETRAC Dataset" --output "D:/Edu/BracU/DETRAC_Dataset.zip"`.
Uploading those files and authorizing Drive access are actions you perform in your account.

This notebook has not been pre-executed. Run each section in order and inspect its results.
Colab hardware availability is not guaranteed; full training may span several sessions.''')
md('''## 1. Connect Drive and copy the project
Adjust the paths if you uploaded the archives into another folder. Extraction rejects unsafe paths.''')
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import zipfile, shutil, os, sys, subprocess, json

PROJECT_ZIP = Path('/content/drive/MyDrive/vehicle-detection-project.zip')
DATASET_ZIP = Path('/content/drive/MyDrive/DETRAC_Dataset.zip')
ROOT = Path('/content/vehicle-detection')
DATA_ROOT = Path('/content/detrac-source')
SAVE_ROOT = Path('/content/drive/MyDrive/DETRAC_learning_runs')

def extract_checked(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f'Unsafe archive entry: {item.filename}')
        z.extractall(destination)

assert PROJECT_ZIP.is_file(), f'Upload {PROJECT_ZIP.name} first'
extract_checked(PROJECT_ZIP, ROOT)
os.chdir(ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
import trafficvision, torch
assert torch.cuda.is_available(), 'Select a GPU runtime before full training'
print('GPU:', torch.cuda.get_device_name(0))

def run(*args):
    subprocess.run([sys.executable, '-m', 'trafficvision', *map(str, args)], check=True)
''')
md('''## 2. Access images and labels
Work from Colab's local disk for faster training. Check free space before extraction.
The archive should contain `DETRAC-Images`, `DETRAC-Train-Annotations-XML`, and
`DETRAC-Test-Annotations-XML` at its top level. The local packaging script makes that layout.''')
code('''assert DATASET_ZIP.is_file(), f'Upload {DATASET_ZIP.name} first'
print('Local free GB:', round(shutil.disk_usage('/content').free / 1e9, 1))
extract_checked(DATASET_ZIP, DATA_ROOT)
for folder in ['DETRAC-Images', 'DETRAC-Train-Annotations-XML', 'DETRAC-Test-Annotations-XML']:
    assert (DATA_ROOT / folder).is_dir(), f'Missing {folder}'
''')
md('''## 3. Download initialization and run tests
Transfer learning means starting with useful pretrained weights, then changing them through training.
The baseline is an 80-class COCO model; the trained model learns DETRAC's four classes.''')
code('''from urllib.request import urlretrieve
Path('models').mkdir(exist_ok=True)
if not Path('models/yolov8n.pt').exists():
    urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt', 'models/yolov8n.pt')
subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
''')
md('''## 4. Two-epoch execution check
Prepare two spaced frames per sequence. The split is still by whole sequences.
This tiny run verifies the workflow; it is not a reliable accuracy experiment.
Use a fresh preparation folder and fresh run name if repeating an experiment.''')
code('''SMOKE = ROOT / 'data/smoke-ready'
if not (SMOKE / 'data.yaml').exists():
    run('prepare', '--root', DATA_ROOT, '--output', SMOKE, '--stride', 100, '--limit', 2)
print(json.loads((SMOKE / 'report.json').read_text())['splits'])
run('train', '--config', 'configs/smoke.yaml', '--device', '0')
''')
md('''## 5. Prepare the full dataset
The official 60 training sequences become 48 training + 12 validation sequences.
All 40 official test sequences remain separate. Unannotated frames are excluded,
not turned into negative examples. Read `report.json` before training.

Ignored XML regions are masked in copied images using the README's custom policy.
This is not official DETRAC benchmark evaluation. Raw-image performance may differ.''')
code('''FULL = ROOT / 'data/full'
if not (FULL / 'data.yaml').exists():
    run('prepare', '--root', DATA_ROOT, '--output', FULL)
report = json.loads((FULL / 'report.json').read_text())
assert not report['errors'], report['errors'][:10]
print(json.dumps(report['splits'], indent=2))
print('Unannotated frames excluded:', len(report['unannotated_images']))
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
BEST = SAVE_ROOT / 'detrac_full/weights/best.pt'
for name in ['report.json', 'splits.json', 'manifest.csv', 'sequences.json', 'exclusions.jsonl']:
    shutil.copy2(FULL / name, SAVE_ROOT / name)
''')
md('''## 6. Full fine-tuning
An epoch is one pass through the training data. Loss measures prediction error;
validation metrics show how the model performs on separate examples.
The full run starts from the original initialization, not the smoke model.

Checkpoints are written to Drive each epoch. Keep this run's name unique.
Up to 50 epochs run, with early stopping after 10 epochs without improvement.
If GPU memory is insufficient, rerun a new experiment with batch 4.
**Do not inspect the test split to choose training settings.**''')
code('''import yaml
cfg = yaml.safe_load(Path('configs/train.yaml').read_text())
cfg.update(data=str(FULL / 'data.yaml'), weights=str(ROOT / 'models/yolov8n.pt'),
           project=str(SAVE_ROOT), name='detrac_full', device='0', batch=-1)
Path('configs/colab.yaml').write_text(yaml.safe_dump(cfg))
run('train', '--config', 'configs/colab.yaml')
assert BEST.is_file()
''')
md('''### If a session disconnects
Reconnect Drive, reinstall the project, and recreate the dataset at the same `/content` paths.
Then run the following cell **instead of starting full training again**.
Skip this cell after a successfully completed run.''')
code('''# Uncomment only for an interrupted run with unfinished epochs:
# run('resume', '--weights', SAVE_ROOT / 'detrac_full/weights/last.pt', '--device', '0')
''')
md('''## 7. Understand training curves and compare compatible classes
Training loss should generally decrease. Validation quality matters more than
training loss alone. Improving training loss with worsening validation can indicate overfitting.
`best.pt` is chosen by validation fitness.

Compare car and bus only. COCO does not supply the same van/others definitions.
This comparison uses fixed confidence .25 and matching IoU .5, and reports excluded
predictions overlapping unsupported reference classes. It is conditional precision/recall, not AP.''')
code('''from IPython.display import display, Image
display(Image(filename=str(SAVE_ROOT / 'detrac_full/results.png')))
run('compare', '--baseline', 'models/yolov8n.pt', '--trained', BEST,
    '--data', FULL / 'data.yaml', '--split', 'val',
    '--output', SAVE_ROOT / 'car_bus_comparison.json', '--device', '0')
print((SAVE_ROOT / 'car_bus_comparison.json').read_text())
''')
md('''## 8. Final test — only after model selection
Evaluate the selected model on untouched test sequences. Do not adjust the model based on this result.
Precision: fraction of detections that are correct. Recall: fraction of reference vehicles found.
mAP summarizes precision over confidence thresholds; mAP50–95 also requires more precise box overlap.
These are custom masked-image metrics, not official DETRAC AP.''')
code('''run('evaluate', '--weights', BEST, '--data', FULL / 'data.yaml', '--split', 'test',
    '--output', SAVE_ROOT / 'final_test', '--device', '0')
print((SAVE_ROOT / 'final_test/metrics.json').read_text())
''')
md('''## 9. Analyze a raw image and export CSV
Choose an original image below, or replace the path with an uploaded image.
No XML labels are supplied to the predictor. Boxes and types come from the trained model.
Centers are calculated from box corners in the original image's pixel coordinates.''')
code('''EXAMPLE = DATA_ROOT / 'DETRAC-Images/MVI_39031/img00001.jpg'
run('predict', '--image', EXAMPLE, '--weights', BEST,
    '--output', SAVE_ROOT / 'example_prediction', '--device', '0')
display(Image(filename=str(SAVE_ROOT / 'example_prediction/annotated.png')))
print((SAVE_ROOT / 'example_prediction/vehicles.csv').read_text())
''')
md('''## 10. Download the trained model
The full run, curves, metrics, and example outputs are already in your Drive.
Download the checkpoint to use with the local Python prediction command.
Keep the dataset manifests with your report so the results can be reproduced.''')
code('''from google.colab import files
files.download(str(BEST))
''')

notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}, 'colab': {'name': 'Train_and_Test_DETRAC.ipynb'},
    'accelerator': 'GPU'}, 'cells': cells}
for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:03d}'
out = Path(__file__).resolve().parents[1]/'notebooks/Train_and_Test_DETRAC.ipynb'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
print(out)
