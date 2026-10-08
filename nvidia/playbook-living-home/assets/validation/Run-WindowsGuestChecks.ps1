[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
# This kit must run inside the new Hyper-V guest, never on the reference Spark.
$computer = Get-CimInstance Win32_ComputerSystem
if ($computer.Manufacturer -ne 'Microsoft Corporation' -or $computer.Model -ne 'Virtual Machine') {
    throw 'Run this kit inside the clean Hyper-V Windows guest. No tests were launched.'
}
$root = $PSScriptRoot
$inventory = Get-Content -LiteralPath (Join-Path $root 'inventory.json') -Raw | ConvertFrom-Json
foreach ($file in $inventory.files) {
    if ($file.name -notin @('LivingHomeSetup.exe','InstallerTests.exe','Run-WindowsGuestChecks.ps1')) { throw 'Unexpected kit inventory entry.' }
    $path = Join-Path $root $file.name
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $file.sha256) { throw "Kit checksum mismatch: $($file.name)" }
}
$out = Join-Path $root ('results-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
New-Item -ItemType Directory -Path $out -ErrorAction Stop | Out-Null
$binary = Join-Path $root 'LivingHomeSetup.exe'
$tests = Join-Path $root 'InstallerTests.exe'
$report = Join-Path $out 'installer-tests.json'
& $tests '--bootstrapper' $binary '--report' $report 2>&1 | Tee-Object -FilePath (Join-Path $out 'installer-tests.log')
$testExit = $LASTEXITCODE
$suite = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
$readinessSource = Join-Path $root 'check-only-host.json'
Copy-Item -LiteralPath $readinessSource -Destination (Join-Path $out 'installer-readiness.json')
$readiness = Get-Content -LiteralPath $readinessSource -Raw | ConvertFrom-Json
$gpu = @($readiness.checks | Where-Object { $_.name -eq 'NVIDIA GPU' })
$summary = [ordered]@{
    completedAtUtc = [DateTime]::UtcNow.ToString('o')
    environment = 'Windows Hyper-V guest'
    installerSha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    installerTestExitCode = $testExit
    passedTestGroups = $suite.passed
    failedTestGroups = $suite.failed
    productionReadinessChecks = $readiness.checks
    expectedNoGpuRefusalObserved = ($gpu.Count -eq 1 -and $gpu[0].status -eq 'fail')
    fullStackInstalled = $false
    modelInferenceTested = $false
    notes = 'Synthetic bootstrapper fixtures and production check-only gates. No release entry point launched. A GPU refusal is expected in this guest; other failed checks require separate review.'
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $out 'summary.json') -Encoding UTF8
Write-Output "Saved guest results to $out"
if ($testExit -ne 0 -or $suite.failed -ne 0) { exit 1 }
exit 0
