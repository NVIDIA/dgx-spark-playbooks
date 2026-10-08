[CmdletBinding()]
param([string]$Python = 'python', [string]$Node = 'node', [switch]$WindowsBinaries)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
foreach ($directory in @('workflows', 'packaging', 'runtime\health\tests', 'runtime\ui\tests')) {
    & $Python -m unittest discover -s (Join-Path $root $directory) -p 'test_*.py' -v
    if ($LASTEXITCODE -ne 0) { throw "Tests failed: $directory" }
}
$plugin = Join-Path $root 'runtime\openclaw-adapter'
Push-Location $plugin
try {
    & $Node 'scripts\build.mjs'
    if ($LASTEXITCODE -ne 0) { throw 'Adapter build failed' }
    & $Node '--test' 'test\adapter.test.mjs'
    if ($LASTEXITCODE -ne 0) { throw 'Adapter tests failed' }
} finally { Pop-Location }
& $Python (Join-Path $PSScriptRoot 'audit_repository.py')
if ($LASTEXITCODE -ne 0) { throw 'Source content audit failed' }
if ($WindowsBinaries) {
    & (Join-Path $root 'installer\build.ps1') -Tests
    & (Join-Path $root 'runtime\ui\build.ps1')
}
Write-Output 'Source checks passed. These checks do not validate models, cron execution, or physical devices.'
