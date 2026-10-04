# guard-summary.ps1 - summarize one guard-test run: bench throughput, guard transitions, memory/paging/GPU.
#   .\guard-summary.ps1 guard
param([Parameter(Mandatory = $true)][string]$Tag)
$samples = Join-Path $PSScriptRoot "$Tag.samples.csv"
$engine = Join-Path $PSScriptRoot "$Tag.engine.log"
$bench = Join-Path $PSScriptRoot "$Tag.bench.log"

Write-Host "===== $Tag ====="
Write-Host "--- bench ---"
if (Test-Path $bench) { Get-Content $bench | Where-Object { $_ -match "prefill|decode|tok|t/s|tokens" } }
else { Write-Host "(no bench log)" }

Write-Host "--- guard transitions ---"
if (Test-Path $engine) {
  $lines = Select-String -Path $engine -Pattern "memory-guard" -ErrorAction SilentlyContinue | ForEach-Object { $_.Line }
  $lines = $lines | Where-Object { $_ -notmatch "pressure on|pressure off" }   # drop per-poll verbose samples
  $lines | ForEach-Object { Write-Host $_ }
  $low = ($lines | Where-Object { $_ -match ": LOW" }).Count
  $rec = ($lines | Where-Object { $_ -match "recovered" }).Count
  $hard = ($lines | Where-Object { $_ -match "trimmed working set|trimmed .* ->" }).Count
  $soft = ($lines | Where-Object { $_ -match "soft ceiling" }).Count
  $pred = ($lines | Where-Object { $_ -match "predictive" }).Count
  Write-Host "counts: LOW=$low recovered=$rec hard=$hard soft=$soft predictive=$pred"
  $lines | Where-Object { $_ -match "monitor thread exiting" } | ForEach-Object { Write-Host $_ }
} else { Write-Host "(no engine log)" }

Write-Host "--- samples ---"
if (Test-Path $samples) {
  $s = Import-Csv $samples
  if ($s.Count -gt 0) {
    $cols = $s[0].PSObject.Properties.Name
    function Stats($vals) { ($vals | Measure-Object -Minimum -Maximum -Average) }
    $av = Stats ($s.avail_mib | ForEach-Object { [double]$_ })
    $ws = Stats ($s.strata_ws_mib | ForEach-Object { [double]$_ } | Where-Object { $_ -ge 0 })
    Write-Host ("samples={0}  avail MiB min/avg/max = {1:N0}/{2:N0}/{3:N0}   strata WS MiB min/avg/max = {4:N0}/{5:N0}/{6:N0}" -f `
      $s.Count, $av.Minimum, $av.Average, $av.Maximum, $ws.Minimum, $ws.Average, $ws.Maximum)
    # Cumulative page counters: the first->last delta is the page-file volume moved over the run.
    # When the hog was active, also report the delta from the first hog-active sample, so an overflow
    # A/B compares the same window (the load phase differs between arms).
    if ($cols -contains "page_out_total") {
      $po = $s.page_out_total | ForEach-Object { [double]$_ } | Where-Object { $_ -ge 0 }
      $pi = $s.page_in_total | ForEach-Object { [double]$_ } | Where-Object { $_ -ge 0 }
      if ($po.Count -ge 2) {
        $wroteP = $po[-1] - $po[0]
        $readP  = $pi[-1] - $pi[0]
        # a page is 4 KiB; report mebibytes
        Write-Host ("page-file volume over run: wrote {0:N0} MiB, read {1:N0} MiB ({2:N0} / {3:N0} pages)" -f `
          ($wroteP * 4 / 1024), ($readP * 4 / 1024), $wroteP, $readP)
      }
      if ($cols -contains "hog_alive") {
        $hogRows = @($s | Where-Object { "$($_.hog_alive)" -match "True" })
        if ($hogRows.Count -ge 2) {
          $hpo = $hogRows.page_out_total | ForEach-Object { [double]$_ }
          $hw = $hpo[-1] - $hpo[0]
          Write-Host ("page-file volume while the hog was active: wrote {0:N0} MiB ({1:N0} pages over {2} samples)" -f `
            ($hw * 4 / 1024), $hw, $hogRows.Count)
        }
      }
    }
    if ($cols -contains "pagefile_peak_mb") {
      $pp = Stats ($s.pagefile_peak_mb | ForEach-Object { [double]$_ } | Where-Object { $_ -ge 0 })
      Write-Host ("page file peak = {0:N0} MiB" -f $pp.Maximum)
    }
    if ($cols -contains "pages_output_s") {
      $po = Stats ($s.pages_output_s | ForEach-Object { [double]$_ } | Where-Object { $_ -ge 0 })
      $pmb = Stats ($s.pagefile_mb | ForEach-Object { [double]$_ } | Where-Object { $_ -ge 0 })
      Write-Host ("pages output/s max = {0:N0}   pagefile MB max = {1:N0}" -f $po.Maximum, $pmb.Maximum)
    }
    if ($cols -contains "pages_input_s") {
      $pi = Stats ($s.pages_input_s | ForEach-Object { [double]$_ } | Where-Object { $_ -ge 0 })
      Write-Host ("pages input/s max = {0:N0} (mostly model-SSD faults on this tiered config)" -f $pi.Maximum)
    }
  }
} else { Write-Host "(no samples)" }
