$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) { $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe' }
$out = Join-Path $project 'dist'
New-Item -ItemType Directory -Path $out -Force | Out-Null
& $compiler '/nologo' '/target:winexe' '/platform:anycpu' '/optimize+' '/r:System.Windows.Forms.dll' '/r:System.Web.Extensions.dll' ("/out:" + (Join-Path $out 'LivingHome.exe')) (Join-Path $PSScriptRoot 'Launcher.cs')
if ($LASTEXITCODE -ne 0) { throw 'Living Home launcher compilation failed' }
Get-FileHash -LiteralPath (Join-Path $out 'LivingHome.exe') -Algorithm SHA256
