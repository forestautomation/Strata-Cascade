# cascade_bench - the Strata-Cascade layout tuner

Finds a good tier layout (`--host-budget-gib`, `--host-reserve-gib`, `--expert-cache-device1`,
`--vram-reserve-mib`) for **your** RAM, CPU and GPU. See [`../../docs/TUNING.md`](../../docs/TUNING.md).

| File | What |
| --- | --- |
| `tune_cascade.py` | the portable core (Windows and Linux); sweeps a few layouts, writes a CSV and the winning config |
| `tune-cascade.ps1` | Windows wrapper: detects GPUs/RAM, then calls the core |
| `tune-cascade.sh` | Linux wrapper: same |
| `bench_http.py` | the 32K/5K HTTP bench the core runs against the server (at the config's context/KV; measured runs use 131072 / `--kv int8 --kv-resident 20480`) |
| `prompt32k.txt` | the bench prompt |
| `guard/` | the `--memory-guard` test harness (Windows-only; the guard itself is cross-platform): `guard-test.ps1`, `mem-hog.py`, `guard-summary.ps1` (see [`guard/README.md`](guard/README.md)) |

## Run it

```powershell
# Windows - no -Config picks the newest strata-*.json that setup wrote
tools\cascade_bench\tune-cascade.ps1
# or point at a specific config
tools\cascade_bench\tune-cascade.ps1 -Config strata-<model>.json
```

```bash
# Linux
tools/cascade_bench/tune-cascade.sh --config strata-<model>.json
```

The core writes its CSV, winning config and engine logs next to itself (or to `--out DIR`).
The bench prompt is synthetic and **overfits** - confirm the winner on your real workload.

With a second card it sweeps pinned RAM **scaled to your PC** (at most 6 GiB, bounded by free RAM minus the
reserve and ~45% of the primary card, so a 16 GB machine tries ~2-5 GiB instead of a fixed 4/6/8 that would
over-pin), keeps the adaptive tier on with a helper and off without, treats a prefill out-of-memory as a failed
layout (never a slow one), and warns `INSUFFICIENT MEMORY - RESULTS MAY FAIL` below ~32 GB RAM or a 12 GB card
before continuing.
