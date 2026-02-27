$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonExe = Join-Path $projectRoot ".venv310\\Scripts\\python.exe"
$resultDir = Join-Path $projectRoot "results"

if (-not (Test-Path $pythonExe)) {
    throw "Python venv not found. Run .\\scripts\\setup_env.ps1 first."
}

if (-not (Test-Path $resultDir)) {
    New-Item -ItemType Directory -Path $resultDir | Out-Null
}

Push-Location $projectRoot
try {
    & $pythonExe -m tensorboard.main --logdir $resultDir --port 6006
}
finally {
    Pop-Location
}
