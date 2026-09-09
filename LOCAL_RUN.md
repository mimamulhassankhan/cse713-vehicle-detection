# Run locally on Windows

Google Drive and Colab are no longer required. No ZIP extraction or upload is needed: the program reads the existing images and XML files in Downloads. The old Colab notebook remains optional.

Open PowerShell in `D:\Edu\BracU\vehicle-detection` and use these steps.

## 1. Check the environment

```powershell
.\local.ps1 check
```

The launcher finds the already installed environment at `D:\Edu\BracU\.venv`. It also supports a project-local `.venv` for future installations. If PowerShell blocks the script, run each command with `powershell -ExecutionPolicy Bypass -File .\local.ps1` followed by the action; this affects that process only.

For a fresh computer, install Python 3.12, then create the environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download `models/yolov8n.pt` as described in README.md if it is absent.
Device `auto` uses an available CUDA backend, otherwise CPU. CPU-only PyTorch does not establish whether your computer physically has a GPU. Full training on CPU can take a long time; the already completed small run is only an execution check.

## 2. Prepare the complete local dataset

```powershell
.\local.ps1 prepare
```

This creates one masked training copy at `data/local-full`, keeping the original files unchanged. It still needs local disk space for that copy and labels, but **does not create another ZIP or use cloud storage**. Run the same command again after an interruption: it checks source/settings, reuses verified completed images, and rebuilds manifests. Do not run two preparation processes on the same output directory.

The older `data/full-incomplete` folder is not reused because it predates the resume metadata. Existing data and ZIP archives have not been deleted. Changing source images, annotations, or split settings requires a new preparation folder.

## 3. Train

```powershell
.\local.ps1 train
```

Defaults in `configs/local.yaml`: up to 50 epochs, early-stopping patience 10, original pretrained initialization, and automatic device selection. CPU uses batch 4. Keep the computer awake while training; progress appears in the terminal. Results and curves are saved under `runs/detrac_local`.

To resume unfinished training after at least one completed epoch:

```powershell
.\local.ps1 resume
```

Resume uses the same run's last checkpoint, including optimizer state. It does not restart a completed 50-epoch run. For a different experiment, change the run name in the configuration.

## 4. Test the selected model

After training and all model selection are finished:

```powershell
.\local.ps1 evaluate
```

This evaluates the untouched official test sequences and saves metrics in `runs/local_final_test`. Do not use test results to tune the model. The existing custom ignored-region policy is unchanged; results are not official DETRAC AP.

## 5. Detect vehicles in a new image

```powershell
.\local.ps1 predict -Image 'C:\path\traffic.jpg' -Output outputs/my_frame
```

The output folder contains `annotated.png`, `vehicles.csv`, and `prediction.json`. Use a fresh output folder for each run.

## Repository contents

The repository includes source, configs, tests, notebooks, and guides. Ignore rules exclude datasets, annotations, environments, model weights, archives, logs, and run outputs. Obtain data and initialization weights separately using README.md. Use `prepare -DatasetRoot 'D:\path\DETRAC Dataset'` for your own dataset location. The training commands do not publish files to GitHub.
