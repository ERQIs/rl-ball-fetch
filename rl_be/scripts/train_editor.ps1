param(
    [string]$RunId = "carcatch_v1",
    [int]$TimeoutWait = 300,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonExe = Join-Path $projectRoot ".venv310\\Scripts\\python.exe"
$configPath = Join-Path $projectRoot "config\\catch_ppo.yaml"

if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found. Run .\\scripts\\setup_env.ps1 first."
}

$args = @(
    "-m", "mlagents.trainers.learn",
    $configPath,
    "--run-id", $RunId,
    "--timeout-wait", $TimeoutWait
)

if (-not $Resume) {
    $args += "--force"
}
else {
    $args += "--resume"
}

Push-Location $projectRoot
try {
    & $pythonExe @args
}
finally {
    Pop-Location
}
