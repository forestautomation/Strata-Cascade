# Tuning Strata-Cascade for your PC

Strata-Cascade sorts the model's experts across three tiers - **VRAM** (on your graphics card), a **pinned RAM
budget**, and the **SSD**. How well it runs depends on how that budget is split for *your* RAM, CPU and GPU. This
guide explains each setting and gives you a script to measure the best layout for your machine.

Everything here assumes you are running the cascade: `--tiered-experts` in your config. If you are running plain
upstream Strata, this document does not apply.

> **Windows is validated. The Linux path is implemented but not yet measured.** If you run the script on Linux,
> please open an issue with your numbers.

---

## 1. The three budgets

| Setting | What it controls | Turn it up when | Turn it down when |
| --- | --- | --- | --- |
| `--host-budget-gib N` | how much RAM is **pinned** for the hottest experts | you have spare RAM and see many `COLD` reads | the OS/other apps need memory, or you see paging |
| `--host-reserve-gib N` | RAM left for the SSD page cache and the OS when `--host-budget-gib auto` | your machine feels sluggish or swaps | you want more pinned experts and have RAM to spare |
| `--expert-cache auto\|N` | CUDA0's expert cache (VRAM) | you have VRAM and a high miss rate | a helper or the vision encoder needs the VRAM |
| `--expert-cache-device1 N` | experts held on a **second** GPU (the CUDA1 helper) | you have a second card - this is the cascade's whole point | it is a single-GPU machine |
| `--vram-reserve-mib N` | VRAM headroom left for everything else | you see out-of-memory or the vision encoder is on | you want more cache slots |
| `--kv-resident N` | KV **streaming**: cells kept in VRAM, the rest of the KV in pinned RAM (setup sets this itself - see below) | rarely by hand; the tuner never touches it | n/a |

The **rule of thumb** for the pinned budget: *free RAM minus 6-8 GB* for the OS and the page cache. On a 32 GB PC
that is ~6 GB pinned (`--host-budget-gib 6 --host-reserve-gib 4`) - the value measured fastest on the reference rig.
More pinned means fewer SSD reads. The engine caps how much it will **register**, matched to the card layout: **at
most 6 GiB beside a helper** (there the pin and CUDA0's cache share the driver) and **at most 24 GiB on one card**
(upstream's measured driver limit) - any expert past the cap or a refused chunk is demoted to `COLD` rather than
failing the run. Override with `STRATA_TIERED_PIN_CAP_GIB`. Two cautions: with a **second GPU** the pinned budget
competes with CUDA0's cache for the prefill borrow, so past ~6 GiB a 16 GB primary can run out of VRAM in prefill;
and pinning is *not* the same as letting the OS page - the pinned set is locked, so it does not thrash the pagefile
the way upstream's page-cache low-RAM mode can.

**KV streaming is setup's job, not the tuner's.** At 64K context or more, when the whole KV fits in RAM, setup
enables it automatically (`--kv-resident 32768`: a 32K window of each QSA layer in VRAM, the rest in pinned RAM) so
the freed VRAM holds more experts - upstream's own rule. Below 64K, or when it does not fit, the KV stays in VRAM.
Leave it as setup wrote it: the layout tuner never changes `--kv-resident`.

The **adaptive swaps** (`--adapt-every N` rounds between swaps, `--adapt-swaps N` experts per round) move the
most-routed missing experts into VRAM while it runs; `--adapt-every 0` turns them off. They are part of the fastest
config **with a second GPU**, and the cascade's Phase C makes them safe beside its cache (see `STRATA-CASCADE.md`).
On **one card** they hurt - leave them off (`--adapt-every 0`).

`--expert-profile PATH` is **required** with `--tiered-experts`: the PINNED tier is "the profile's order past the
cache". Generate one with:

```powershell
python tools/make_profile.py --help      # then point --expert-profile at the output
```

The profile lists which experts are most-routed; it is a property of the **model**, so the same profile works on
any machine running the same model - only the *layout* (budgets) changes with your hardware.

---

## 1b. When the page file is on a slow disk: `--memory-guard`

What the guard does on Windows and Linux, and its measured numbers, are in
[`STRATA-CASCADE.md`](../STRATA-CASCADE.md) ("The memory guard"). This section is the tuning reference: the flag, the
knobs and how to test it.

`--memory-guard` (alias `--windows-memory-guard`; env `STRATA_MEMORY_GUARD`, alias `STRATA_WINDOWS_MEMORY_GUARD`) is
opt-in. On Windows it lowers the engine's memory priority, pauses cold prefetch and releases its cheap pages while RAM
is short. The release is **proportional to the deficit**: the soft working-set ceiling is set to `entry_ws - deficit`,
where the deficit is what is missing to reach `keep_free + recover`, so the guard asks for exactly what it needs
rather than a flat fraction of the target. It **holds** `VERY_LOW` memory priority for the whole low period, so the OS
keeps choosing the engine's clean, file-backed COLD pages over another app's dirty ones. Three release modes on
Windows, `STRATA_MEM_GUARD_TRIM`: `soft` (default), `hard` (`EmptyWorkingSet` at every trigger) or `off` (priority +
prefetch pause only). The engine's locked footprint (registered experts plus pinned host KV) cannot be trimmed.

> **No per-range drop on Windows.** `OfferVirtualMemory` looks like `madvise(MADV_DONTNEED)` but is not: it is rejected
> for file mappings (`ERROR_INVALID_PARAMETER`) and, where it does work, it makes the range inaccessible until
> `ReclaimVirtualMemory` and may discard the contents - unsafe for expert weights. So Windows releases with the
> working-set mechanism above, not a per-range drop.

The knobs:

- `STRATA_MEM_GUARD_KEEP_FREE_MIB` (1024) - the free-RAM target the guard holds; the release lifts free RAM to
  `keep + recover` and then stops. Raise it to keep more RAM free for other apps.
- `STRATA_MEM_GUARD_RECOVER_MIB` (1024) - how far above `keep_free` the release aims, and where the guard clears.
  It must be at least the predictive band, or the guard would enter and immediately recover.
- `STRATA_MEM_GUARD_RELEASE_MIB` (256) - the smallest single release once the guard acts.
- `STRATA_MEM_GUARD_MIN_MIB` (512) - the fallback free-RAM floor that also counts as pressure.
- `STRATA_MEM_GUARD_EMERGENCY_MIB` (256) - the cliff floor; a genuine cliff releases hard even in `soft` mode.
- `STRATA_MEM_GUARD_COMMIT_MIB` (2048) - the available-commit floor.
- `STRATA_MEM_GUARD_POLL_MS` (500) - the sample interval.
- `STRATA_MEM_GUARD_RETRIM_MS` (3000) - the hard re-trim rate.
- `STRATA_MEM_GUARD_COOLDOWN_MS` (2000) - the settle time after a recovery.
- `STRATA_MEM_GUARD_PREDICT_SLOPE` (128 MiB/s) / `STRATA_MEM_GUARD_PREDICT_BAND_MIB` (512) - the early trigger: a
  free-RAM decline this fast within the band starts the release, and the band is deliberately narrow so the entry sits
  just above `keep_free`, not 2 GiB above it.
- Toggles: `STRATA_MEM_GUARD_PREDICT=0`, `STRATA_MEM_GUARD_PRIORITY=0`, `STRATA_MEM_GUARD_NOTIFY=0` (Windows),
  `STRATA_MEM_GUARD_VERBOSE=1`, `STRATA_MEM_GUARD_STATS=1`.

Test it with `tools/cascade_bench/guard/` (Windows-only harness) - `guard-test.ps1` runs the engine with the guard
(and a synthetic hog) and samples free RAM, working set, paging and GPU; `guard-summary.ps1` summarizes a run. See
[`../tools/cascade_bench/guard/README.md`](../tools/cascade_bench/guard/README.md).

**Linux:** watch the engine log for `memory guard on (linux (MemAvailable; ...))` and the `memory-guard: LOW` /
`recovered` lines. A/B it with `tune_cascade.py --guard` (wrappers: `-Guard` / `--guard`); it re-runs the winning
layout with the guard on and off and keeps the flag only if it does not cost throughput.

The guard does not make paging fast; it makes the engine the cheapest victim, so the OS drops its clean pages instead
of paging your apps to the slow disk. Moving the page file to an SSD is still the real fix.

---

## 2. What the engine tells you

Start the server and watch the engine log (`logs/strata.engine.log`, or the console). The lines that matter:

```
tiers: VRAM 10662 experts (...) | PINNED 3656 (6.00 GiB ...) | COLD 10258 (16.68 GiB, paged from SSD) | budget 6.00 GiB
tiered decode misses: PINNED 68512 (RAM), COLD 113497 (SSD, 195783.0 MB, 109766 prefetched = 97%)
decode expert cache hit rate: 87.6% (...)
```

(These are the example run in [`../STRATA-CASCADE.md`](../STRATA-CASCADE.md); your own counts will differ.)

- **`COLD ... prefetched = 97%`** - good; `begin_layer` is keeping reads in flight.
- **Many `PINNED` misses** - the CPU is re-reading RAM; harmless but a sign of a low bind rate.
- **A small `PINNED ... budget`** - raise `--host-budget-gib` if you have RAM.
- **A low hit rate** - the VRAM cache or the helper is too small; raise `--expert-cache-device1` or lower
  `--vram-reserve-mib`.

---

## 3. The tuner

`tools/cascade_bench/tune_cascade.py` (portable core) plus a wrapper, `tools/cascade_bench/tune-cascade.ps1`
(Windows) or `tools/cascade_bench/tune-cascade.sh` (Linux). The wrappers detect your GPUs and RAM, then call the
Python core, which drives the server over HTTP and sweeps a few layouts, writing a CSV matrix and printing the best.

```powershell
# Windows - no -Config picks the newest strata-*.json setup wrote
tools\cascade_bench\tune-cascade.ps1
tools\cascade_bench\tune-cascade.ps1 -Config strata-<model>.json
```

```bash
# Linux
tools/cascade_bench/tune-cascade.sh --config strata-<model>.json
```

`--config` is the config **setup wrote** (`strata-<model>.json` in the repo root). It must use `--tiered-experts`,
which setup does on a low-RAM PC with a second GPU.

Or let setup run the whole thing: `START-HERE.bat --tune-cascade` (or `./setup.sh --tune-cascade`) calls this script
with the detected RAM and helper GPUs and copies the winning layout over your config. An interactive install also
offers it once the cascade is on.

What it does, in order:

1. detects GPUs (`nvidia-smi`) and physical RAM (`wmic`/`/proc/meminfo`);
2. sweeps `--host-budget-gib`, **scaled to this PC**: with a **helper** it stays at or below 6 GiB and is bounded
   by free RAM minus the reserve and ~45% of the primary card (so a 16 GB PC tries ~2-5 GiB, not a fixed 4/6/8 that
   would over-pin and page), and without one it brackets the engine's `auto` value by +/-2 GiB;
3. if a second GPU exists, also sweeps 2-3 helper sizes;
4. loads the model once per layout, runs the 32K/5K bench at the config's own context and KV quant (the measured
   runs use the 131072-token (128K) context with `--kv int8`), records decode / prefill / hit rate / tier split;
5. with `--guard` (wrappers: `-Guard` / `--guard`), re-runs the **winning layout** once with `--memory-guard` and
   once without, prints the decode / prefill delta, and keeps the flag in the winning config only if it does not
   cost throughput - the guard step for the responsive-system option (see section 1b); the two rows land in the CSV
   with `memory_guard` 1 and 0. Both arms run the shipped knobs (keep 1024 / recover 1024 / release 256 /
   band 512 MiB) through the config's `env`, and those knobs are written into the winning config when the guard is
   kept - the same on Linux and Windows (the Linux guard only pauses cold prefetch, so a near-zero delta is
   expected);
6. writes `tools/cascade_bench/tune-<date>.csv` and prints the winning layout.

**What it changes, and what it leaves alone.** Each layout starts from your existing `strata-<model>.json` and
overwrites **only** `--host-budget-gib`, `--host-reserve-gib`, `--vram-reserve-mib` and the helper sizes
(`--expert-cache-device1..3`), plus `--adapt-every`/`--adapt-swaps` (8/32 with a helper, 0/0 without). The memory guard
is only touched with `--guard`/`-Guard`, which also writes the shipped `STRATA_MEM_GUARD_*` knobs into the config's
`env` (and drops them when the A/B chose no guard). **Everything else is kept as-is** - the model, `--max-context`, the KV quant
and `--kv-resident` (setup's own KV-streaming decision), speculation, vision, the expert profile, host/port and any
other flag. So the tuner assumes an otherwise valid
`--tiered-experts` config and optimizes the tier layout on top of it; it does not fix or validate the rest.

It is deliberately **small**: 3 budgets x up to 3 helper sizes is ~10 model loads (with the one-shot probe), enough
to find a good layout in about ten to twenty minutes depending on the model's load time. A full sweep would overfit
the bench prompt (below).

**It knows the two rules the sweeps found.** With a helper the adaptive tier is on (`--adapt-every 8 --adapt-swaps
32`); on one card it is off (`--adapt-every 0`), where it has nothing to protect and hurts. And a layout that
starts but then **runs out of VRAM in prefill** is recorded as a failure (a 0 t/s row plus an `oom` reason in the
CSV) rather than being mistaken for a slow layout - exactly the 8 GiB + helper case, which the tuner now avoids by
starting its helper sweep at 4 GiB.

**On a small PC it warns but still runs.** If the machine has less than ~32 GB of RAM or a primary card under
12 GB, it prints `WARNING: INSUFFICIENT MEMORY - RESULTS MAY FAIL` with the reason and continues - a smaller model
(such as Q2_0 or IQ2_XS) may still tune fine.

**It scales the helper size to your model.** The expert *bytes per expert* differ by quant (IQ2_XS < IQ3_XXS <
IQ3_S < Coder), so the tuner does a single probe start, reads the pack's real per-expert size from the engine log
(`largest blob ... MB` / the `CUDA1: ... experts, ... GiB` line), and converts each helper card's usable VRAM into an
expert count from that measurement - never a hardcoded constant. The same script therefore works for IQ2_XS,
IQ3_XXS, IQ3_S and the Coder model.

**It handles 1-4 GPUs.** The first card is the primary (CUDA0); every card after it becomes a helper
(`--expert-cache-device1..3`), each sized from that card's own VRAM. A single-GPU machine just skips the helper
sweep.

### 3.1 Files it writes

| File | What |
| --- | --- |
| `tools/cascade_bench/tune-<date>.csv` | one row per layout: budget, helper, decode, prefill, hit, tier split |
| `tools/cascade_bench/tune-<date>-best.json` | the winning config (copy it over your `strata-*.json`) |
| `tools/cascade_bench/tune-<tag>-l<n>.engine.log` | the engine log of each run |

Use `--out DIR` to write them somewhere else.

---

## 4. Do not overfit the bench

The bundled bench prompt (`tools/cascade_bench/prompt32k.txt`) is one code block repeated many times, so its expert
routing is unusually regular. A layout tuned on it can be *wrong* for real work. Use the script to get close, then
**tune the last step on what you actually run** - paste your real prompt into the chat or point the bench at your
own text, and compare two or three nearby budgets by eye.

---

## 5. Reporting results

When you find a good layout, a short report helps everyone:

```
GPU:      <card> (<VRAM>) + <card> (<VRAM>)
RAM:      <GB> <speed>
CPU:      <model>, <cores>
Disk:     <SSD/NVMe model>
Model:    <IQ3_XXS / ...>
Context:  <--max-context N --kv int8/q4_0>   (setup's KV streaming, if on, is in the config)
Layout:   --host-budget-gib N --host-reserve-gib N --expert-cache-device1 N --vram-reserve-mib N
Result:   32K/5K at <context>: <prefill> t/s prefill, <decode> t/s decode, <hit>%
```

Windows numbers so far: see [`../STRATA-CASCADE.md`](../STRATA-CASCADE.md).

---

## 6. AMD and multi-GPU limits (read before reporting)

- **NVIDIA multi-GPU: supported.** The port understands `--expert-cache-device1..3`, and the tier sort (`settle`)
  accounts for every helper, so a blob is never resident in two VRAM tiers. 2-4 cards work; the tuner sizes each
  helper from its own VRAM.
- **AMD (HIP): unmeasured.** The cascade's device calls go through the HIP compatibility layer, and the helper
  (`remote_experts.cpp`) already has HIP guards - so it *compiles* for AMD. But the pinned tier relies on
  `hipHostRegister` over remapped file pages, whose behaviour differs from CUDA's, and **nobody has run the bench
  on an AMD card**. Treat AMD as "should work, please report".
- **Mixed AMD + NVIDIA in one run is not supported** by upstream (`docs/AMD_HIP.md`), so a helper across an
  AMD card and an NVIDIA card is out of scope for the cascade too.
- **Linux:** the port's Linux path is implemented but not measured. Report if you run it.
