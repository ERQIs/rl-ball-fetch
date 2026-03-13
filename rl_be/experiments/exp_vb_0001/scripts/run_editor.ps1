param(
    [Parameter(Mandatory = $true)]
    [string]$BackboneCkpt,
    [string]$RunId = "exp_vb_0001_editor",
    [int]$TimeoutWait = 300,
    [switch]$Resume,
    [switch]$UnfreezeBackbone,
    [switch]$Inference,
    [Nullable[double]]$TimeScale = $null
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

$env:VBB_ENABLE = "1"
$env:VBB_CKPT = (Resolve-Path $BackboneCkpt)
$env:VBB_FREEZE_BACKBONE = if ($UnfreezeBackbone) { "0" } else { "1" }

$args = @(
    $TrainPy,
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
