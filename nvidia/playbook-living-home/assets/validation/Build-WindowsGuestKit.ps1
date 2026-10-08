[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$kitName = 'windows-guest-kit-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$root = Join-Path $PSScriptRoot ('evidence\' + $kitName)
$sources = @(
    (Join-Path $project 'dist\LivingHomeSetup.exe'),
    (Join-Path $project 'installer\tests\InstallerTests.exe'),
    (Join-Path $PSScriptRoot 'Run-WindowsGuestChecks.ps1')
)
foreach ($path in $sources) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path. Build the installer and tests first." }
}
New-Item -ItemType Directory -Path $root -ErrorAction Stop | Out-Null
$files = @()
foreach ($source in $sources) {
    $name = Split-Path -Leaf $source
    $target = Join-Path $root $name
    Copy-Item -LiteralPath $source -Destination $target
    $files += [ordered]@{name=$name; sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()}
}
[ordered]@{kind='Windows guest bootstrapper test kit; not a Living Home release'; files=$files} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'inventory.json') -Encoding UTF8
$zip = $root + '.zip'
Compress-Archive -LiteralPath $root -DestinationPath $zip
[pscustomobject]@{Path=$zip; Sha256=(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant(); WindowsGuestExecution='PENDING'} | ConvertTo-Json
