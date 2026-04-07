param(
    [Parameter(Mandatory = $true)]
    [string]$EnvPath,
    [string]$SourceRunId = "temp_realTime_0001",
    [string]$EvalRunId = "",
    [ValidateSet("baseline", "short", "self", "long", "short_readout", "self_readout", "long_readout")]
    [string]$Condition = "baseline",
    [int]$NumEnvs = 4,
    [int]$TimeoutWait = 300,
    [int]$MaxSteps = 20000,
    [int]$SummaryFreq = 1000,
    [switch]$NoGraphics,
    [switch]$InvertVisual,
    [Nullable[double]]$TimeScale = $null,
    [Nullable[int]]$Width = $null,
    [Nullable[int]]$Height = $null,
    [string]$WarmupCkpt = ""
)

$ErrorActionPreference = "Stop"

$ExpRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectRoot = Resolve-Path (Join-Path $ExpRoot "..\..\..")
$PythonExe = Join-Path $ProjectRoot "rl_be\.venv\Scripts\python.exe"
$TrainPy = Join-Path $ExpRoot "train_embodied.py"
$TrainPyc = Join-Path $ExpRoot "__pycache__\train_embodied.cpython-310.pyc"
$BaseConfigPath = Join-Path $ExpRoot "config\ppo_embodied_recurrent.yaml"
$ResultsDir = Join-Path $ProjectRoot "rl_be\results\exp_ec_0001"
$TempConfigPath = Join-Path $ExpRoot "config\ppo_embodied_recurrent_eval_temp.yaml"

if (-not (Test-Path $PythonExe)) {
    throw "Python venv not found: $PythonExe"
}
if (-not (Test-Path $BaseConfigPath)) {
    throw "Config not found: $BaseConfigPath"
}
if (-not (Test-Path $EnvPath)) {
    throw "Unity build not found: $EnvPath"
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

$SourceRunDir = Join-Path $ResultsDir $SourceRunId
if (-not (Test-Path $SourceRunDir)) {
    throw "Source run not found: $SourceRunDir"
}

if (-not $EvalRunId) {
    $EvalRunId = "${SourceRunId}_ablate_${Condition}"
}

$EvalRunDir = Join-Path $ResultsDir $EvalRunId
if (Test-Path $EvalRunDir) {
    throw "Evaluation run already exists: $EvalRunDir"
}

$configText = Get-Content -Path $BaseConfigPath -Raw
$frozenBufferSize = [Math]::Max($MaxSteps + 1024, 65536)
$configText = [regex]::Replace($configText, 'max_steps:\s*\d+', "max_steps: $MaxSteps")
$configText = [regex]::Replace($configText, 'summary_freq:\s*\d+', "summary_freq: $SummaryFreq")
$configText = [regex]::Replace($configText, 'checkpoint_interval:\s*\d+', "checkpoint_interval: $($MaxSteps + 1)")
$configText = [regex]::Replace($configText, 'buffer_size:\s*\d+', "buffer_size: $frozenBufferSize")
$configText = [regex]::Replace($configText, 'learning_rate:\s*[-+eE0-9.]+', "learning_rate: 0.0")
$configText = [regex]::Replace($configText, 'beta:\s*[-+eE0-9.]+', "beta: 0.0")
$configText = [regex]::Replace($configText, 'num_epoch:\s*\d+', "num_epoch: 1")
[System.IO.File]::WriteAllText(
    $TempConfigPath,
    $configText,
    [System.Text.UTF8Encoding]::new($false)
)

$env:EC_CKPT = (Resolve-Path $WarmupCkpt)
$env:EC_WINDOW_LENGTH = "8"
$env:EC_FRAME_CHANNELS = "1"
$env:EC_SELF_STATE_DIM = "2"
$env:EC_FREEZE_LOCAL_ENCODER = "1"
$env:EC_FREEZE_LOCAL_DYNAMICS = "0"
$env:EC_EXPORT_ONNX = "0"
$env:EC_INVERT_VISUAL = "0"
$env:EC_ABLATE_SHORT = "0"
$env:EC_ABLATE_SELF = "0"
$env:EC_ABLATE_LONG = "0"
$env:EC_ABLATE_SHORT_READOUT = "0"
$env:EC_ABLATE_SELF_READOUT = "0"
$env:EC_ABLATE_LONG_READOUT = "0"

switch ($Condition) {
    "short" { $env:EC_ABLATE_SHORT = "1" }
    "self" { $env:EC_ABLATE_SELF = "1" }
    "long" { $env:EC_ABLATE_LONG = "1" }
    "short_readout" { $env:EC_ABLATE_SHORT_READOUT = "1" }
    "self_readout" { $env:EC_ABLATE_SELF_READOUT = "1" }
    "long_readout" { $env:EC_ABLATE_LONG_READOUT = "1" }
}
if ($InvertVisual) {
    $env:EC_INVERT_VISUAL = "1"
}

Write-Host "[exp_ec_0001] Running ablation condition=$Condition source_run=$SourceRunId eval_run=$EvalRunId max_steps=$MaxSteps invert_visual=$InvertVisual"
if ($NoGraphics) {
    Write-Warning "NoGraphics is not recommended for this visual task; camera rendering may fail."
}

$args = @(
    $TrainEntry,
    $TempConfigPath,
    "--run-id", $EvalRunId,
    "--results-dir", $ResultsDir,
    "--initialize-from", $SourceRunId,
    "--env", $EnvPath,
    "--num-envs", $NumEnvs,
    "--timeout-wait", $TimeoutWait
)

if ($NoGraphics) {
    $args += "--no-graphics"
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

Push-Location $ProjectRoot
try {
    & $PythonExe @args
}
finally {
    Pop-Location
}
