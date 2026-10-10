# guard - the `--memory-guard` test harness (Windows-only)

The guard itself - what it does on Windows vs Linux, and the measured numbers - is in
[`../../../STRATA-CASCADE.md`](../../../STRATA-CASCADE.md) ("The memory guard"); the knobs and how to test it are in
[`../../../docs/TUNING.md`](../../../docs/TUNING.md) section 1b. This folder is the **Windows-only harness** that
exercises it: it runs the engine under a real workload beside a synthetic memory hog and samples RAM / working-set /
paging / GPU. The guard is cross-platform (on Linux its only lever is the prefetch pause; on the cascade that pause
needs `STRATA_PREFETCH_PAUSE_ON_PRESSURE=1`), but this harness is PowerShell and Windows-only.

| File | What |
| --- | --- |
| `guard-test.ps1` | starts the engine through `serve.server` with your run config, adds the guard and its knobs, runs the 32K/5K bench, optionally drives a memory hog, samples RAM/working-set/paging/GPU |
| `mem-hog.py` | the synthetic memory hog: `chunk` ramp, one-shot `burst`, or `active` (kept resident, the background-app case) |
| `guard-summary.ps1` | summarizes one run: bench throughput, guard releases, page-file volume, free-RAM range |
| `example-config.json` | a template run config - **edit the model paths for your machine** |

`guard-test.ps1` reuses `../bench_http.py` and `../prompt32k.txt`, so it lives beside the rest of the
bench tooling.

## Run it

```powershell
# copy the template and point it at your pack / GGUF
Copy-Item tools\cascade_bench\guard\example-config.json my-run.json
# edit my-run.json: exe, --pack, --native, --ple-gguf, tokenizer, --expert-profile

# baseline with no guard
tools\cascade_bench\guard\guard-test.ps1 -Config my-run.json -NoGuard -NoHog -Tag baseline
# the guard, normal run
tools\cascade_bench\guard\guard-test.ps1 -Config my-run.json -NoHog -Tag guard
# a 10 GiB app that appears in one allocation (a sudden launch)
tools\cascade_bench\guard\guard-test.ps1 -Config my-run.json -HogMB 10240 -HogBurst -Tag burst
# a 16 GiB background app that stays active (the work-in-the-background case)
tools\cascade_bench\guard\guard-test.ps1 -Config my-run.json -HogMB 16384 -HogActive -Tag bg

# summarize a run
tools\cascade_bench\guard\guard-summary.ps1 guard
```

Outputs (`<Tag>.engine.log`, `<Tag>.samples.csv`, `<Tag>.bench.log`, ...) land next to the script, or in
`-OutDir DIR`.

> **A big `-HogMB active` can make a low-RAM box near-unresponsive.** That is the point of the test, but
> on a daily-driver use a size that overflows by a few GiB (for 32 GB, 12-16 GiB beside the engine) and
> give the hog a page-file headroom check. The `active` load also keeps its pages hot on purpose, so it
> cannot just be trimmed away - it is the hardest case.

## What to look for

- The engine log's `strata memory-guard:` lines: `LOW ... soft ceiling -> N MiB` on a release,
  `recovered ...` when it lifts, and `still low, trimmed ...` for a hard release. No `memory-guard:` line
  means the guard did not engage.
- In `<Tag>.samples.csv`: `avail_mib` should stay off the floor; `strata_pf_mib` (the engine's own page-file/commit
  use) should stay flat, and `strata_faults` should not spike - the guard moves clean pages to standby, it does not
  write them to disk. `strata_ws_mib` should fall by roughly the released amount, not collapse to near zero.
- `guard-summary.ps1` prints the page-file **volume** written over the run, the engine's page-file/fault range, and
  (when the hog was active) over just the hog-active window.
- `STRATA_MEM_GUARD_STATS=1` (on by default here) also prints the monitor thread's own CPU share.

## Measured

On this rig (Windows 11, 32 GB, RTX 5060 Ti + RTX 3060, IQ3_XXS), the best two-card config
(`--host-budget-gib 6`, CUDA1 helper 6250), 32K prefill / 5K decode, **one run each**:

| Arm | Prefill t/s | Decode t/s | Cache hit |
| --- | ---: | ---: | ---: |
| best config + guard, no hog | **900.5** | **51.8** | 86.2% |
| 12 GiB active hog, guard | 805.1 | 32.5 | - |

The no-pressure run is the regression check: the guard costs nothing when RAM is not short (it matches the config
without the guard within the +/-5% run-to-run spread). Under the 12 GiB active hog the guard trims the engine's
working set while the OS reports low memory, and keeps the other app's pages out of the page file. These are single runs on a PC that
sits near full - indicative, not precise. The guard itself, and the numbers behind it, are in
[`../../../STRATA-CASCADE.md`](../../../STRATA-CASCADE.md) ("The memory guard").

## Knobs

`guard-test.ps1` exposes the `STRATA_MEM_GUARD_*` settings as parameters; the defaults are the shipped
ones (`soft` trim, `keep_free` 1536 MiB, `recover` 512 MiB, `release_min` 512 MiB, `emergency` 512 MiB).
`-GuardTrim hard` reproduces the pre-ladder behaviour (`EmptyWorkingSet`); `-GuardNotify 0`/`-GuardPredict 0`
disable the OS notification / predictive triggers. The engine reads them from the run config's `env` block, which the
script fills in.

The bundled bench prompt is synthetic and **overfits** - confirm any tuning on your real workload.
