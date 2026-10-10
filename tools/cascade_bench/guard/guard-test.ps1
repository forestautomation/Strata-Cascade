# guard-test.ps1 - exercise the Windows memory guard (--windows-memory-guard) beside a real workload.
#
# Starts the tiered engine through serve.server with a run config of your own, optionally adds
# --windows-memory-guard and the STRATA_MEM_GUARD_* knobs, runs the cascade_bench 32K/5K HTTP bench, and
# optionally drives memory pressure with mem-hog.py so the guard releases and then recovers.  It samples
# free RAM, the engine's working set, paging and GPU into a CSV and prints the guard's log lines.
#
# Needs a run config (the same JSON serve.server takes) whose --pack / --native paths point at a model
# you have.  Copy tools/cascade_bench or an existing run-*.json and edit the paths; nothing here is
# hard-coded to one machine.
#
#   baseline, no guard:        .\guard-test.ps1 -Config run.json -NoGuard -NoHog -Tag baseline
#   on-demand guard (default): .\guard-test.ps1 -Config run.json -NoHog -Tag guard-normal
#   sudden 10 GiB app:         .\guard-test.ps1 -Config run.json -HogMB 10240 -HogBurst -Tag guard-burst
#   background app (real):     .\guard-test.ps1 -Config run.json -HogMB 16000 -HogActive -Tag guard-bg
#   pre-ladder hard arm:       .\guard-test.ps1 -Config run.json -GuardTrim hard -GuardNotify 0 -Tag guard-hard
param(
  [Parameter(Mandatory = $true)][string]$Config,
  [int]$Port = 8082,
  [int]$HogMB = 10240,
  [int]$HogHoldS = 120,
  [int]$GuardMinMib = 1024,
  [int]$GuardKeepFreeMib = 1536,
  [int]$GuardEmergencyMib = 512,
  [int]$GuardRecoverMib = 512,
  [int]$GuardReleaseMib = 512,
  [int]$GuardCommitMib = 3072,
  [int]$GuardPollMs = 250,
  [int]$GuardRetrimMs = 1000,
  [int]$GuardCooldownMs = 2000,
  [string]$GuardTrim = "soft",
  [string]$GuardNotify = "1",
  [string]$GuardPriority = "1",
  [string]$GuardPredict = "0",
  [string]$GuardPredictSlope = "128",
  [string]$GuardStats = "1",
  [int]$SampleMs = 2000,
  [int]$MaxTokens = 5000,
  [int]$Repeats = 1,
  [int]$HogAfterS = 25,
  [string]$Prompt = "prompt32k.txt",
  [string]$OutDir = "",
  [string]$Tag = "guard-test",
  [switch]$NoHog,
  [switch]$HogBurst,
  [switch]$HogActive,
  [switch]$NoGuard
)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))   # repo root
$benchDir = Split-Path -Parent $PSScriptRoot                                          # tools/cascade_bench
if (-not [System.IO.Path]::IsPathRooted($Config)) { $Config = Join-Path $root $Config }
if (-not (Test-Path $Config)) { Write-Host "run config not found: $Config"; exit 2 }
if (-not $OutDir) { $OutDir = $PSScriptRoot }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$log = Join-Path $OutDir "$Tag.engine.log"
$console = Join-Path $OutDir "$Tag.console.log"
$benchOut = Join-Path $OutDir "$Tag.bench.log"
$hogOut = Join-Path $OutDir "$Tag.hog.log"
$samples = Join-Path $OutDir "$Tag.samples.csv"

function Stop-Strata {
  Get-Process strata -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*serve.server*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 5
}
Stop-Strata
Remove-Item $log, $console, "$console.err", $benchOut, "$benchOut.err", $hogOut, "$hogOut.err", $samples -ErrorAction SilentlyContinue

$cfg = Get-Content $Config -Raw | ConvertFrom-Json
if ($NoGuard) {
  Write-Host "guard OFF (baseline)"
} else {
  if ($cfg.args -notcontains "--windows-memory-guard") { $cfg.args += "--windows-memory-guard" }
  if (-not ($cfg.PSObject.Properties.Name -contains "env")) { $cfg | Add-Member -NotePropertyName env -NotePropertyValue ([pscustomobject]@{}) }
  $envkv = @{
    STRATA_MEM_GUARD_MIN_MIB         = "$GuardMinMib"
    STRATA_MEM_GUARD_KEEP_FREE_MIB   = "$GuardKeepFreeMib"
    STRATA_MEM_GUARD_EMERGENCY_MIB   = "$GuardEmergencyMib"
    STRATA_MEM_GUARD_RECOVER_MIB     = "$GuardRecoverMib"
    STRATA_MEM_GUARD_RELEASE_MIB     = "$GuardReleaseMib"
    STRATA_MEM_GUARD_COMMIT_MIB      = "$GuardCommitMib"
    STRATA_MEM_GUARD_POLL_MS         = "$GuardPollMs"
    STRATA_MEM_GUARD_RETRIM_MS       = "$GuardRetrimMs"
    STRATA_MEM_GUARD_COOLDOWN_MS     = "$GuardCooldownMs"
    STRATA_MEM_GUARD_TRIM            = "$GuardTrim"
    STRATA_MEM_GUARD_NOTIFY          = "$GuardNotify"
    STRATA_MEM_GUARD_PRIORITY        = "$GuardPriority"
    STRATA_MEM_GUARD_PREDICT         = "$GuardPredict"
    STRATA_MEM_GUARD_PREDICT_SLOPE   = "$GuardPredictSlope"
    STRATA_MEM_GUARD_STATS           = "$GuardStats"
    STRATA_MEM_GUARD_VERBOSE         = "1"
  }
  foreach ($kv in $envkv.GetEnumerator()) { $cfg.env | Add-Member -NotePropertyName $kv.Key -NotePropertyValue $kv.Value -Force }
}
$cfg.log = $log
$runtime = Join-Path $OutDir "$Tag.json"
$cfg | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 $runtime

$server = Start-Process -FilePath $py `
  -ArgumentList @("-m", "serve.server", "--engine", "strata", "--config", $runtime, "--port", $Port) `
  -RedirectStandardOutput $console -RedirectStandardError "$console.err" `
  -WorkingDirectory $root -PassThru -WindowStyle Hidden

$deadline = (Get-Date).AddSeconds(900)
$ready = $false
while ((Get-Date) -lt $deadline) {
  try {
    $m = Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 4
    if ($m.data -and $m.data.Count -gt 0) { $ready = $true; break }
  } catch { }
  if ($server.HasExited) { break }
  Start-Sleep -Seconds 5
}
if (-not $ready) {
  Write-Host "NOT READY - see $console / $log"
  Get-Content $log -Tail 40 -ErrorAction SilentlyContinue
  Stop-Strata
  exit 1
}

$mode = if ($NoGuard) { "no guard" } elseif ($NoHog) { "guard, no hog" } else { "guard + ${HogMB} MiB hog$(if ($HogBurst) { ' (burst)' } elseif ($HogActive) { ' (active/background)' })" }
Write-Host "READY [$mode trim=$GuardTrim keep=$GuardKeepFreeMib notify=$GuardNotify predict=$GuardPredict]. bench ($MaxTokens tokens x$Repeats) ..."
$bench = Start-Process -FilePath $py `
  -ArgumentList @((Join-Path $benchDir "bench_http.py"), "$Port", (Join-Path $benchDir $Prompt), "$MaxTokens", "$Repeats") `
  -RedirectStandardOutput $benchOut -RedirectStandardError "$benchOut.err" `
  -WorkingDirectory $root -PassThru -WindowStyle Hidden

$hog = $null
if (-not $NoHog) {
  Start-Sleep -Seconds $HogAfterS
  $hogArgs = @((Join-Path $PSScriptRoot "mem-hog.py"), "$HogMB", "$HogHoldS")
  if ($HogBurst) { $hogArgs += "burst" }
  elseif ($HogActive) { $hogArgs += "active" }
  $hog = Start-Process -FilePath $py -ArgumentList $hogArgs `
    -RedirectStandardOutput $hogOut -RedirectStandardError "$hogOut.err" `
    -WorkingDirectory $root -PassThru -WindowStyle Hidden
}

"t_s,avail_mib,strata_ws_mib,strata_private_mib,strata_pf_mib,strata_faults,strata_cpu_s,page_in_total,page_out_total,pagefile_mb,pagefile_peak_mb,gpu_util,hog_alive" | Set-Content $samples
$t0 = Get-Date
while (-not $bench.HasExited -and ((Get-Date) - $t0).TotalMinutes -lt 40) {
  $avail = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
  $p = Get-Process strata -ErrorAction SilentlyContinue
  $ws = if ($p) { [math]::Round($p.WorkingSet64 / 1MB, 0) } else { -1 }
  $priv = if ($p) { [math]::Round($p.PrivateMemorySize64 / 1MB, 0) } else { -1 }
  $cpu = if ($p) { [math]::Round($p.CPU, 1) } else { -1 }
  # Per-process page-file volume and cumulative hard-fault count: the guard's job is to keep the engine's
  # PageFileUsage flat (it moves clean pages to standby, not to disk).  Win32_Process reports KB / a count.
  $spf = -1; $faults = -1
  try {
    $wp = Get-CimInstance Win32_Process -Filter "Name='strata.exe'" -ErrorAction Stop | Select-Object -First 1
    if ($wp) { $spf = [math]::Round($wp.PageFileUsage / 1024, 0); $faults = [double]$wp.PageFaults }
  } catch { }
  # Cumulative system page counts (raw perf counters); the delta over the run is the real page-file volume.
  $pin = -1; $pout = -1
  try {
    $pm = Get-CimInstance Win32_PerfRawData_PerfOS_Memory -ErrorAction Stop
    $pin = [double]$pm.PagesInputPersec
    $pout = [double]$pm.PagesOutputPersec
  } catch { }
  $pfmb = -1; $pfpeak = -1
  try {
    $pf = Get-CimInstance Win32_PageFileUsage -ErrorAction Stop
    $pfmb = [math]::Round(($pf | Measure-Object -Property CurrentUsage -Sum).Sum, 0)
    $pfpeak = [math]::Round(($pf | Measure-Object -Property PeakUsage -Sum).Sum, 0)
  } catch { }
  $gpu = ""
  try { $gpu = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null) -join "/" } catch { }
  $hogAlive = if ($hog) { -not $hog.HasExited } else { $false }
  $t = [math]::Round(((Get-Date) - $t0).TotalSeconds, 0)
  "$t,$avail,$ws,$priv,$spf,$faults,$cpu,$pin,$pout,$pfmb,$pfpeak,$gpu,$hogAlive" | Add-Content $samples
  Write-Host "t=$t s avail=$avail MiB strataWS=$ws MiB strataPF=$spf MiB faults=$faults cpu=$cpu s pf=$pfmb MB peak=$pfpeak MB gpu=$gpu hog=$hogAlive"
  Start-Sleep -Milliseconds $SampleMs
}
if ($hog -and -not $hog.HasExited) { Stop-Process -Id $hog.Id -Force -ErrorAction SilentlyContinue }
$bench.WaitForExit()
Start-Sleep -Seconds 5   # let the guard log a recovery once the hog is gone
$p = Get-Process strata -ErrorAction SilentlyContinue
if ($p) {
  Write-Host ("after: strataWS={0} MiB cpu={1}s" -f [math]::Round($p.WorkingSet64 / 1MB, 0), [math]::Round($p.CPU, 1))
  $p | Stop-Process -Force -ErrorAction SilentlyContinue
}

Write-Host "=== guard lines ($log) ==="
Select-String -Path $log -Pattern "memory-guard" -ErrorAction SilentlyContinue | ForEach-Object { $_.Line }
Write-Host "=== bench ==="
Get-Content $benchOut -ErrorAction SilentlyContinue
Write-Host "=== hog ==="
Get-Content $hogOut -ErrorAction SilentlyContinue
Write-Host "samples: $samples"
Stop-Strata
Write-Host "DONE."
