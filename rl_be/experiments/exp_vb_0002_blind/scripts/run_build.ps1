param(
    [Parameter(Mandatory = $true)]
    [string]$EnvPath,
    [string]$RunId = "exp_vb_0002_blind_build",
    [int]$NumEnvs = 1,
    [int]$TimeoutWait = 300,
    [switch]$Resume,
    [switch]$NoGraphics,
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
if (-not (Test-Path $EnvPath)) {
    throw "Unity build not found: $EnvPath"
}

$env:BLIND_ENABLE = "1"
$env:BLIND_ZERO_VECTOR = if ($KeepVectorObs) { "0" } else { "1" }

$args = @(
    $TrainPy,
    $ConfigPath,
    "--run-id", $RunId,
    "--results-dir", $ResultsDir,
    "--env", $EnvPath,
    "--num-envs", $NumEnvs,
    "--timeout-wait", $TimeoutWait
)

if ($NoGraphics) {
    $args += "--no-graphics"
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
