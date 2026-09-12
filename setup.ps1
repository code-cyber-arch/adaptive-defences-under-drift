param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -c "import sys; assert sys.version_info[:2] == (3, 14), 'Python 3.14 is required'"
    if ($LASTEXITCODE -ne 0) { throw "Python version check failed" }
    if (-not (Test-Path -LiteralPath ".venv/Scripts/python.exe")) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Environment creation failed" }
    }
    & ".venv/Scripts/python.exe" -m pip install -r requirements.lock.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
    & ".venv/Scripts/python.exe" -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed" }
} finally { Pop-Location }
