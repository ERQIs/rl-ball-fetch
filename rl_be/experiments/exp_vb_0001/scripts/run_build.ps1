param(
    [Parameter(Mandatory = $true)]
    [string]$BackboneCkpt,
    [Parameter(Mandatory = $true)]
    [string]$EnvPath,
    [string]$RunId = "exp_vb_0001_build",
    [int]$NumEnvs = 1,
    [int]$TimeoutWait = 300,
    [switch]$Resume,
    [switch]$UnfreezeBackbone,
    [switch]$NoGraphics,
    [switch]$Inference,
    [Nullable[double]]$TimeScale = $null,
    [Nullable[int]]$Width = $null,
    [Nullable[int]]$Height = $null
)

$ErrorActionPreference = "Stop"

$ExpRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectRoot = Resolve-Path (Join-Path $ExpRoot "..\..\..")
$PythonExe = Join-Path $ProjectRoot "rl_be\.venv\Scripts\python.exe"
$TrainPy = Join-Path $ExpRoot "train_with_backbone.py"
$ConfigPath = Join-Path $ExpRoot "config\ppo_vision_backbone.yaml"
$ResultsDir = Join-Path $ProjectRoot "rl_be\results\exp_vb_0001"

if (-not (Test-Path $PythonExe)) {
    throw "Python venv not found: $PythonExe"
}
if (-not (Test-Path $BackboneCkpt)) {
    throw "Backbone checkpoint not found: $BackboneCkpt"
}
if (-not (Test-Path $EnvPath)) {
    throw "Unity build not found: $EnvPath"
}

$env:VBB_ENABLE = "1"
$env:VBB_CKPT = (Resolve-Path $BackboneCkpt)
$env:VBB_FREEZE_BACKBONE = if ($UnfreezeBackbone) { "0" } else { "1" }

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

if ($Inference) {
    $args += "--inference"
}

if ($TimeScale -ne $null) {
    $args += @("--time-scale", [string]$TimeScale)
}
if ($Width -ne $null) {
    $args += @("--width", [string]$Width)
}
if ($Height -ne $null) {
    $args += @("--height", [string]$Height)
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
