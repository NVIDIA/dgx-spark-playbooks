[CmdletBinding()]
param([switch]$Tests)
$ErrorActionPreference = 'Stop'
$installerRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $installerRoot
$distRoot = Join-Path $projectRoot 'dist'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
}
if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { throw 'The .NET Framework C# compiler is not installed. Install .NET Framework 4.8.1 first.' }
New-Item -ItemType Directory -Path $distRoot -Force | Out-Null
$common = @('/nologo', '/optimize+', '/platform:anycpu', '/r:System.dll', '/r:System.Core.dll', '/r:System.Management.dll', '/r:System.Web.Extensions.dll', '/r:System.IO.Compression.dll', '/r:System.IO.Compression.FileSystem.dll')
$binary = Join-Path $distRoot 'LivingHomeSetup.exe'
& $compiler @common '/target:winexe' '/r:System.Drawing.dll' '/r:System.Windows.Forms.dll' ("/out:{0}" -f $binary) (Join-Path $installerRoot 'InstallerCore.cs') (Join-Path $installerRoot 'SetupApp.cs')
if ($LASTEXITCODE -ne 0) { throw "Bootstrapper compilation failed: $LASTEXITCODE" }
$hash = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath (Join-Path $distRoot 'LivingHomeSetup.exe.sha256') -Value ("{0}  LivingHomeSetup.exe" -f $hash) -Encoding ASCII
Write-Output ("Built {0} ({1} bytes), SHA256 {2}" -f $binary, (Get-Item -LiteralPath $binary).Length, $hash)
if ($Tests) {
    $testBinary = Join-Path $installerRoot 'tests\InstallerTests.exe'
    & $compiler @common '/target:exe' '/define:INSTALLER_TESTS' '/main:LivingHome.Setup.InstallerTests' ("/out:{0}" -f $testBinary) (Join-Path $installerRoot 'InstallerCore.cs') (Join-Path $installerRoot 'tests\InstallerTests.cs')
    if ($LASTEXITCODE -ne 0) { throw "Safety-test compilation failed: $LASTEXITCODE" }
    & $testBinary '--bootstrapper' $binary '--report' (Join-Path $installerRoot 'tests\test-results.json')
    if ($LASTEXITCODE -ne 0) { throw "Safety tests failed: $LASTEXITCODE" }
}
