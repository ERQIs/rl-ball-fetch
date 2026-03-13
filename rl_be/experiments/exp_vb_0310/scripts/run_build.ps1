param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("E1", "E2", "E3", "E4")]
    [string]$Experiment,
    [Parameter(Mandatory = $true)]
    [string]$EnvPath,
    [string]$RunId = "",
    [string]$BackboneCkpt = "",
    [int]$NumEnvs = 1,
    [int]$TimeoutWait = 300,
    [switch]$Resume,
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
$TrainPy = Join-Path $ExpRoot "train_variants.py"
$ResultsDir = Join-Path $ProjectRoot "rl_be\results\exp_vb_0310\$Experiment"

if (-not (Test-Path $PythonExe)) {
    throw "Python venv not found: $PythonExe"
}
if (-not (Test-Path $EnvPath)) {
    throw "Unity build not found: $EnvPath"
}

$ConfigPath = switch ($Experiment) {
    "E1" { Join-Path $ExpRoot "config\E1_global_scratch.yaml" }
    "E2" { Join-Path $ExpRoot "config\E2_spatial_scratch.yaml" }
    "E3" { Join-Path $ExpRoot "config\E3_spatial_pretrain_frozen.yaml" }
    "E4" { Join-Path $ExpRoot "config\E4_spatial_pretrain_finetune.yaml" }
}

$env:VB_MODE = switch ($Experiment) {
    "E1" { "global_scratch" }
    "E2" { "spatial_scratch" }
    "E3" { "spatial_pretrain_frozen" }
    "E4" { "spatial_pretrain_finetune" }
}

if ($Experiment -in @("E3", "E4")) {
    if (-not $BackboneCkpt) {
        throw "BackboneCkpt is required for $Experiment."
    }
    if (-not (Test-Path $BackboneCkpt)) {
        throw "Backbone checkpoint not found: $BackboneCkpt"
    }
    $env:VB_CKPT = (Resolve-Path $BackboneCkpt)
}
else {
    $env:VB_CKPT = ""
}

$env:VB_FREEZE_BACKBONE = if ($Experiment -eq "E3") { "1" } else { "0" }
$env:VB_SPATIAL_CHANNELS = "8"

if (-not $RunId) {
    $RunId = "exp_vb_0310_${Experiment}_build"
}

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
