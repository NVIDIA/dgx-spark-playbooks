[CmdletBinding()]
param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$features = @(Get-CimInstance Win32_OptionalFeature | Where-Object Name -Like 'Microsoft-Hyper-V*' | Select-Object Name,InstallState)
if (-not $Apply) {
    [pscustomobject]@{
        PreviewOnly = $true
        Action = 'Enable Microsoft-Hyper-V-All and its dependencies'
        AutomaticRestart = $false
        ExistingVMsChanged = $false
        Features = $features
        NextStep = 'After owner authorization, run this script as Administrator with -Apply. Review RestartNeeded before scheduling a reboot.'
    } | ConvertTo-Json -Depth 4
    exit 0
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator PowerShell is required to enable the Windows Hyper-V feature. No feature was changed.'
}
$reportPath = Join-Path $PSScriptRoot 'evidence\hyperv-enable-result.json'
try {
    $result = Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -All -NoRestart
    $report = [pscustomobject]@{Status='completed';EnabledFeature='Microsoft-Hyper-V-All';RestartNeeded=[bool]$result.RestartNeeded;AutomaticRestart=$false;CompletedAt=(Get-Date).ToUniversalTime().ToString('o')}
    $report | ConvertTo-Json | Set-Content -LiteralPath $reportPath -Encoding UTF8
    $report | ConvertTo-Json
} catch {
    [pscustomobject]@{Status='failed';Error=$_.Exception.Message;AutomaticRestart=$false;CompletedAt=(Get-Date).ToUniversalTime().ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $reportPath -Encoding UTF8
    throw
}
