"""A small, teachable vehicle-detection workflow."""
import os
from pathlib import Path

# Keep library settings beside the project, not in the user's roaming profile.
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path('.runtime/ultralytics').resolve()))
os.environ.setdefault('MPLCONFIGDIR', str(Path('.runtime/matplotlib').resolve()))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
