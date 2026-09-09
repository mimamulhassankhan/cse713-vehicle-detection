from pathlib import Path
import json
import shutil
import sys

import cv2
import torch
import ultralytics


def get_device_info():
    if torch.cuda.is_available():
        return {
            "backend": "cuda",
            "available": True,
            "gpu": torch.cuda.get_device_name(0),
            "selected_device": "cuda:0",
        }

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return {
            "backend": "xpu",
            "available": True,
            "gpu": torch.xpu.get_device_name(0),
            "selected_device": "xpu:0",
        }

    return {
        "backend": "cpu",
        "available": True,
        "gpu": None,
        "selected_device": "cpu",
    }


root = Path(__file__).resolve().parents[1]
device_info = get_device_info()

print(json.dumps({
    "python": sys.executable,
    "torch": torch.__version__,
    "opencv": cv2.__version__,
    "ultralytics": ultralytics.__version__,

    "cuda_available": torch.cuda.is_available(),
    "xpu_available": hasattr(torch, "xpu") and torch.xpu.is_available(),

    "backend": device_info["backend"],
    "gpu": device_info["gpu"],
    "selected_device": device_info["selected_device"],

    "initial_weights_present": (root / "models/yolov8n.pt").is_file(),
    "full_dataset_prepared": (root / "data/local-full/data.yaml").is_file(),
    "project_disk_free_GB": round(shutil.disk_usage(root).free / 1e9, 1),
}, indent=2))