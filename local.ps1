param(
    [ValidateSet('check','prepare','train','resume','evaluate','predict')]
    [string]$Action = 'check',
    [string]$DatasetRoot = 'C:\Users\imamu\Downloads\DETRAC Dataset',
    [string]$Image,
    [string]$Output = 'outputs/local_prediction',
    [string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    # Prefer this workspace's verified Python 3.12 environment over a newer
    # project environment that may point to an unavailable Store installation.
    $pythonCandidates = @("$PSScriptRoot/../.venv/Scripts/python.exe", "$PSScriptRoot/.venv/Scripts/python.exe")
    $pythonExe = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $pythonExe) { throw 'Create the Python environment using LOCAL_RUN.md first.' }
    $arguments = @('-m','trafficvision')
    switch ($Action) {
        'check' {
            & $pythonExe scripts/check_local.py
        }
        'prepare' {
            $arguments += @('prepare','--root',$DatasetRoot,'--output','data/local-full')
            if (Test-Path -LiteralPath 'data/local-full/preparation-settings.json') { $arguments += '--resume' }
            & $pythonExe @arguments
        }
        'train' { & $pythonExe @arguments train --config configs/local.yaml --device $Device }
        'resume' { & $pythonExe @arguments resume --weights runs/detrac_local/weights/last.pt --device $Device }
        'evaluate' { & $pythonExe @arguments evaluate --weights runs/detrac_local/weights/best.pt --data data/local-full/data.yaml --split test --output runs/local_final_test --device $Device }
        'predict' {
            if (-not $Image) { throw 'Supply -Image with the path to your image.' }
            & $pythonExe @arguments predict --image $Image --weights runs/detrac_local/weights/best.pt --output $Output --device $Device
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE. Read the error above." }
} finally { Pop-Location }
