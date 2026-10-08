[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$IsoPath,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Fa-f0-9]{64}$')][string]$IsoSha256,
    [string]$LabRoot = 'C:\LivingHome-installer\validation\machines',
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$vmName = 'LivingHome-Win11-InstallerLab'
$iso = (Resolve-Path -LiteralPath $IsoPath).Path
if ([IO.Path]::GetExtension($iso) -ne '.iso') { throw 'Supply the official Windows 11 ARM64 ISO.' }
if ((Get-FileHash -LiteralPath $iso -Algorithm SHA256).Hash -ne $IsoSha256) { throw 'ISO SHA256 does not match the supplied publisher checksum.' }
$root = [IO.Path]::GetFullPath($LabRoot)
$protected = [IO.Path]::GetFullPath('C:\LivingHome').TrimEnd('\')
if ($root -eq $protected -or $root.StartsWith($protected+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Use a separate lab directory.' }
$folder = Join-Path $root $vmName
if (Test-Path -LiteralPath $folder) { throw 'Lab machine directory already exists. Existing files are preserved.' }
$plan = [ordered]@{Name=$vmName;Generation=2;MemoryGB=8;Processors=4;DiskGB=96;Path=$folder;Iso=$iso;IsoSha256=$IsoSha256;Network='Default Switch';GPUAssignment=$false;StartAutomatically=$false}
if (-not $Apply) { [pscustomobject]$plan | ConvertTo-Json; exit 0 }
Import-Module Hyper-V -ErrorAction Stop
if (Get-VM -Name $vmName -ErrorAction SilentlyContinue) { throw 'A VM already has this name; it will not be changed.' }
$switch = Get-VMSwitch -Name 'Default Switch' -ErrorAction Stop
if ($switch.SwitchType -ne 'Internal') { throw 'Expected the internal NAT Default Switch; no external switch will be created.' }
New-Item -ItemType Directory -Path $folder -ErrorAction Stop | Out-Null
$plan | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $folder 'lab-ownership.json') -Encoding UTF8
$vm = New-VM -Name $vmName -Generation 2 -MemoryStartupBytes 8GB -Path $folder -NewVHDPath (Join-Path $folder 'system.vhdx') -NewVHDSizeBytes 96GB -SwitchName $switch.Name
Set-VMProcessor -VM $vm -Count 4
Set-VM -VM $vm -AutomaticStartAction Nothing -AutomaticStopAction Save
Set-VMKeyProtector -VM $vm -NewLocalKeyProtector
Enable-VMTPM -VM $vm
Set-VMFirmware -VM $vm -EnableSecureBoot On -SecureBootTemplate MicrosoftWindows
$dvd = Add-VMDvdDrive -VM $vm -Path $iso -Passthru
Set-VMFirmware -VM $vm -FirstBootDevice $dvd
Get-VM -Name $vmName | Select-Object Name,State,Generation,MemoryAssigned,ProcessorCount | ConvertTo-Json
