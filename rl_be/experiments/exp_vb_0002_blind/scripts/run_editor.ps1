param(
    [string]$RunId = "exp_vb_0002_blind_editor",
    [int]$TimeoutWait = 300,
    [switch]$Resume,
    [switch]$KeepVectorObs
)

$ErrorActionPreference = "Stop"

$ExpRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectRoot = Resolve-Path (Join-Path $ExpRoot "..\..\..")
$PythonExe = Join-Path $ProjectRoot "rl_be\.venv\Scripts\python.exe"
$TrainPy = Join-Path $ExpRoot "train_blind.py"
$ConfigPath = Join-Path $ExpRoot "config\ppo_blind_baseline.yaml"
$ResultsDir = Join-Path $ProjectRoot "rl_be\results\exp_vb_0002_blind"

if (-not (Test-Path $PythonExe)) {
    throw "Python venv not found: $PythonExe"
}

$env:BLIND_ENABLE = "1"
$env:BLIND_ZERO_VECTOR = if ($KeepVectorObs) { "0" } else { "1" }

$args = @(
    $TrainPy,
    $ConfigPath,
    "--run-id", $RunId,
    "--results-dir", $ResultsDir,
    "--timeout-wait", $TimeoutWait
)

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
