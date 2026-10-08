[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [string]$NodeExecutable,
    [string]$OpenClawEntry,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw 'Provide the own installation Python executable.'
}
$arguments = @((Join-Path $PSScriptRoot 'configure_home_report.py'), '--config', $ConfigPath)
if ($Apply) {
    if (-not $NodeExecutable -or -not $OpenClawEntry) { throw '-Apply requires -NodeExecutable and -OpenClawEntry.' }
    $arguments += @('--node', $NodeExecutable, '--openclaw', $OpenClawEntry, '--apply')
}
& $PythonExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw 'Home report configuration did not complete.' }
