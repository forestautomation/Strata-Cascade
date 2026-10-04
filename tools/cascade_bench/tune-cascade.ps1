# tune-cascade.ps1 - find a good Strata-Cascade tier layout for THIS Windows PC.
# Detects the GPUs, physical RAM and the pack's disk, then calls the portable
# core (tune_cascade.py), which sweeps a few layouts on the 32K/5K bench.
#
# Usage:
#   tools\cascade_bench\tune-cascade.ps1 -Config strata-<model>.json
#   tools\cascade_bench\tune-cascade.ps1 -Config <path> -Port 8080 -Repeats 2
#   tools\cascade_bench\tune-cascade.ps1 -Guard        # A/B the memory guard on the winning layout
#
# --Config is the config setup wrote (strata-<model>.json in the repo root).
# See docs/TUNING.md. The bench prompt overfits - confirm the winner on your real workload.
param(
  [string]$Config = "",
  [int]$Port = 8080,
  [int]$MaxTokens = 5000,
  [int]$Repeats = 1,
  [string]$Tag = "tune",
  [switch]$Guard
)
$ErrorActionPreference = "Continue"
$here = $PSScriptRoot
$root = Split-Path -Parent (Split-Path -Parent $here)   # tools\cascade_bench -> repo root

# default: the config setup wrote, picked by newest
if (-not $Config) {
  $found = Get-ChildItem -Path $root -Filter 'strata-*.json' -File -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($found) { $Config = $found.FullName }
  else { Write-Host "no config found: run START-HERE.bat first, then pass -Config <strata-*.json>"; exit 1 }
}
if (-not [System.IO.Path]::IsPathRooted($Config)) {
  $candidate = Join-Path $root $Config
  if (Test-Path $candidate) { $Config = $candidate }
  else { $Config = Join-Path $here $Config }
}
$py = Join-Path $root ".venv\Scripts\python.exe"
$core = Join-Path $here "tune_cascade.py"

if (-not (Test-Path $py)) { Write-Host "no python venv at $py - run START-HERE.bat once."; exit 1 }

Write-Host "=== detecting this PC ==="
$ramGib = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
$gpus = @()

# NVIDIA via nvidia-smi (memory.free is the driver's real free figure, like the engine's cudaMemGetInfo)
try {
  $smi = & nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>$null
  foreach ($line in $smi) {
    $parts = $line -split ','
    if ($parts.Count -ge 2) {
      $free = if ($parts.Count -ge 3 -and $parts[2].Trim() -match '^\d+$') { [int]$parts[2].Trim() } else { -1 }
      $gpus += [PSCustomObject]@{ Name = $parts[0].Trim(); VramMib = [int]$parts[1].Trim(); FreeMib = $free }
    }
  }
} catch { }

# AMD via the Windows driver (Win32_VideoController), when no nvidia-smi rows or in addition to them.
# The WMI AdapterRAM is 32-bit (at most 4 GB) and there is no per-adapter free figure, so FreeMib = -1
# and the core falls back to the 82% cap.
try {
  $vc = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
  foreach ($v in $vc) {
    if ($v.Name -match 'Radeon|AMD' -and $v.AdapterRAM -gt 0) {
      $mib = [int]([math]::Min($v.AdapterRAM / 1MB, 1024*1024))
      if ($mib -gt 0) { $gpus += [PSCustomObject]@{ Name = $v.Name.Trim() + ' (AMD)'; VramMib = $mib; FreeMib = -1 } }
    }
  }
} catch { }

foreach ($g in $gpus) {
  if ($g.FreeMib -ge 0) { Write-Host ("  GPU: {0}  ({1} MiB, {2} MiB free)" -f $g.Name, $g.VramMib, $g.FreeMib) }
  else { Write-Host ("  GPU: {0}  ({1} MiB)" -f $g.Name, $g.VramMib) }
}
Write-Host ("  RAM: {0} GiB" -f $ramGib)

if ($gpus | Where-Object { $_.Name -match 'AMD' }) {
  Write-Host "  NOTE: AMD/HIP + the cascade is not measured (docs/TUNING.md section 6). Multi-GPU across AMD+NVIDIA is not supported upstream."
}

# physical RAM the model may use: total minus what the OS + desktop already hold (~3 GiB).
# The core then subtracts its own --reserve-gib (the page cache/OS headroom) from this.
$freeGib = [math]::Round($ramGib - 3, 1)

# the OS's REAL free RAM right now (ullAvailPhys equivalent): the honest input for the pinned budget, the same
# figure the engine reads for --host-budget-gib auto.
$availGib = $freeGib
try {
  $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
  if ($os.FreePhysicalMemory -gt 0) { $availGib = [math]::Round($os.FreePhysicalMemory / 1MB, 1) }
} catch { }

# helper cards = every GPU after the first.  Usable VRAM is hybrid, like the engine: the driver's real free figure
# minus a ~0.5 GiB reserve, capped at 82% of the card's total (an over-large count makes the engine refuse to start,
# not clamp).  When no free figure is available (Windows AMD), the 82% cap alone is used.
$helperGibs = @()
if ($gpus.Count -ge 2) {
  for ($i = 1; $i -lt $gpus.Count -and $i -le 3; $i++) {
    $total = $gpus[$i].VramMib / 1024.0
    $cap = $total * 0.82
    if ($gpus[$i].FreeMib -ge 0) {
      $free = ($gpus[$i].FreeMib / 1024.0) - 0.5
      $usable = [math]::Min($cap, [math]::Max(0.0, $free))
    } else {
      $usable = $cap
    }
    $helperGibs += [math]::Round($usable, 2)
  }
  Write-Host ("  {0} helper GPU(s) -> usable VRAM {1} GiB" -f $helperGibs.Count, ($helperGibs -join ', '))
} else {
  Write-Host "  single GPU -> no --expert-cache-device1 sweep"
}

# the main card's total VRAM, for the core's low-memory warning (32 GB RAM / 12 GB card targets)
$primaryGib = 0.0
if ($gpus.Count -ge 1) { $primaryGib = [math]::Round($gpus[0].VramMib / 1024.0, 1) }

$coreArgs = @($core, "--config", $Config, "--python", $py, "--port", $Port,
              "--max-tokens", $MaxTokens, "--repeats", $Repeats,
              "--free-gib", $freeGib, "--available-gib", $availGib, "--tag", $Tag)
if ($helperGibs.Count -gt 0) { $coreArgs += @("--helper-gib", ($helperGibs -join ',')) }
if ($primaryGib -gt 0) { $coreArgs += @("--primary-gib", $primaryGib) }
if ($Guard) { $coreArgs += @("--guard") }

Write-Host ""
& $py @coreArgs
exit $LASTEXITCODE
