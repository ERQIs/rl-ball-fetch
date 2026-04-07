param(
    [string]$RunId = "temp_realTime_0001",
    [int]$TimeoutWait = 300,
    [switch]$Resume,
    [switch]$Inference,
    [Nullable[double]]$TimeScale = $null,
    [string]$WarmupCkpt = ""
)

$ErrorActionPreference = "Stop"

$ExpRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectRoot = Resolve-Path (Join-Path $ExpRoot "..\..\..")
$PythonExe = Join-Path $ProjectRoot "rl_be\.venv\Scripts\python.exe"
$TrainPy = Join-Path $ExpRoot "train_embodied.py"
$TrainPyc = Join-Path $ExpRoot "__pycache__\train_embodied.cpython-310.pyc"
$ConfigPath = Join-Path $ExpRoot "config\ppo_embodied_recurrent.yaml"
$ResultsDir = Join-Path $ProjectRoot "rl_be\results\exp_ec_0001"

if (-not (Test-Path $PythonExe)) {
    throw "Python venv not found: $PythonExe"
}
if (-not (Test-Path $ConfigPath)) {
    throw "Config not found: $ConfigPath"
}

$TrainEntry = ""
if (Test-Path $TrainPy) {
    $TrainEntry = $TrainPy
}
elseif (Test-Path $TrainPyc) {
    $TrainEntry = $TrainPyc
}
else {
    throw "Neither embodied trainer source nor bytecode was found."
}

if (-not $WarmupCkpt) {
    $WarmupCkpt = Join-Path $ProjectRoot "temperal\experiments\embodied_control\output\offline_warmup_manual_capture_20260401_101517\best.pt"
}
if (-not (Test-Path $WarmupCkpt)) {
    throw "Warmup checkpoint not found: $WarmupCkpt"
}

$env:EC_CKPT = (Resolve-Path $WarmupCkpt)
$env:EC_WINDOW_LENGTH = "8"
$env:EC_FRAME_CHANNELS = "1"
$env:EC_SELF_STATE_DIM = "2"
$env:EC_FREEZE_LOCAL_ENCODER = "1"
$env:EC_FREEZE_LOCAL_DYNAMICS = "0"
$env:EC_EXPORT_ONNX = "0"

$args = @(
    $TrainEntry,
    $ConfigPath,
    "--run-id", $RunId,
    "--results-dir", $ResultsDir,
    "--timeout-wait", $TimeoutWait
)

if ($Inference) {
    $args += "--inference"
}
if ($TimeScale -ne $null) {
    $args += @("--time-scale", [string]$TimeScale)
}
if ($Resume) {
    $args += "--resume"
}
else {
    $args += "--force"
}

Push-Location $ProjectRoot
try {
    & $PythonExe @args
}
finally {
    Pop-Location
}
