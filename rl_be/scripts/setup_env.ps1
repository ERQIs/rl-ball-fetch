param(
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPath = Join-Path $projectRoot ".venv310"

if (-not (Test-Path $venvPath)) {
    py -$PythonVersion -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\\python.exe"

& $pythonExe -m pip install --upgrade pip wheel "setuptools<81"
& $pythonExe -m pip install -r (Join-Path $projectRoot "requirements.txt")

& $pythonExe -m pip check

Write-Host "Environment ready: $venvPath"
Write-Host "Pinned core versions: mlagents=1.1.0, torch=2.1.2, onnx=1.15.0, protobuf=3.20.3"
