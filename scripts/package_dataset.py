"""Create a ZIP64 archive for the user's own Colab upload. Does not upload files."""
import argparse
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    if out.exists() or out == root or root in out.parents:
        raise ValueError('Choose a new archive path outside the original dataset')
    folders = ['DETRAC-Images', 'DETRAC-Train-Annotations-XML', 'DETRAC-Test-Annotations-XML']
    for folder in folders:
        if not (root/folder).is_dir():
            raise ValueError(f'Missing {folder}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for folder in folders:
            for path in sorted((root/folder).rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
            print(f'Archived {folder}', flush=True)
    print(f'Upload this archive manually to your own Drive: {out}')


if __name__ == '__main__':
    main()
