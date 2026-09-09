"""Package source/notebook only, excluding data, weights, environments and run outputs."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
out = root.parent/'vehicle-detection-project.zip'
top = ['README.md', 'LOCAL_RUN.md', 'local.ps1', 'STATUS.md', 'requirements.txt', 'requirements-lock-windows.txt', '.gitignore']
folders = ['trafficvision', 'configs', 'docs', 'notebooks', 'scripts', 'tests']
paths = [root/name for name in top]
for folder in folders:
    paths.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, path.relative_to(root).as_posix())
print(out)
