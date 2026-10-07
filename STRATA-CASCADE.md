# Strata-Cascade: what it is, what it measured

**Build this documents:** upstream Strata **v0.1.39** plus the cascade port - branch `cascade`, re-landed as one commit
on `origin/main` (`6f32ec0`). The numbers are from a **fresh build of this port**, measured against
a **fresh build of the same base with no port** (a throwaway stock worktree), one run each at 128K - the same
base on both sides, so the comparisons isolate the fork. Everything not described here is upstream's.

Cascade is **upstream Strata** plus multiple additions, built around a **cascading expert source**
(`--tiered-experts`) that lets a PC whose RAM cannot hold every expert run the model anyway, by sorting the experts
across three tiers and combining them with the CUDA1..3 expert helper - a combination upstream refuses. This file is
the **reference**: what it does, what it measured, and how it was measured. To **re-apply the port** onto a new
upstream, see `docs/strata-cascade-implementation-guide.md`; to **tune the layout for your own machine**, see
`docs/TUNING.md`. The engine plus setup/tuner/docs are the one `cascade` commit.

Strata-Cascade's own additions are the Windows implementation of the tiered source; running a pinned RAM budget *and*
a second GPU's expert cache together (upstream forbids that pair, and combining them is what makes the PC fast); and
the adaptive rule for which experts move into VRAM while it runs - it no longer duplicates experts the second GPU
already holds, and it keeps a cheap RAM copy of any it promotes. Setup's automatic selection and the layout tuner are
this fork's too. **Attribution:** the tiered expert source adapts the algorithm of upstream
[PR #80](https://github.com/Niko1221/Strata/pull/80) by @andrewcoul (closed upstream), with parts rewritten so it
works on Windows.

---

## Cascade vs stock upstream on the same PC

Measured against a **stock upstream build** on the same machine, same model and settings. Both arms were rebuilt
from the same base the day of the run: the cascade from this branch, the stock engine from `origin/main` (`6f32ec0`)
with no port, in a throwaway worktree. Every arm: **Qwen3.8-Flash-Next IQ3_XXS, 32K-token prompt
/ 5K reply at 128K context, `--kv int8 --kv-resident 20480`**, vision on, `--spec 4 --spec-min-p
0.5`, both cards. One run each.

| Arm | Config | Decode | Prefill | Cache hit |
| --- | --- | ---: | ---: | ---: |
| **A. Upstream, dual-GPU (2 GPU, current)** | `--mmap-experts --remote-expert-opt --expert-cache-device1 6250` (6f32ec0) | 42.5 t/s | 252.9 t/s | 87.4% |
| **B. Cascade, 2 cards (current)** | `--tiered-experts --host-budget-gib 6 --expert-cache-device1 6250 --adapt-every 8 --memory-guard` | **51.8 t/s** | **900.5 t/s** | 86.2% |
| **C. Upstream, low-RAM (1 GPU)** | `--resident-experts`, warm OS file cache | 29.7 t/s | 434 t/s | 35% |
| **D. Cascade, 1 card** | `--tiered-experts --host-budget-gib 8 --adapt-every 0` | 28.2 t/s | ~960 t/s | 38% |

**The current two-card cascade is 1.2x the current stock dual-GPU decode and 3.6x its prefill**, with a cache hit
rate on par with stock (86.2% vs 87.4%). That is the fresh, same-base comparison (A vs B). The gaps are far larger than
run-to-run noise (+/-5%). The stock arm uses upstream's `--remote-expert-opt` (87.4% hit); the cascade arm does not use
it (with it the cascade measured 47.7 t/s decode / 892.7 t/s prefill / 89.9% hit).

**Why**, from the engine logs of each arm:

- **A (stock, dual-GPU)** uses both cards but **upstream forbids a RAM tier beside a helper** (`generate.cpp`), so
  every miss is a synchronous file read: prefill stays at 253 t/s. (On the earlier same-base run it read
  **1,279,108 MB** in one request, sat at ~61 MiB of free RAM, average 281 MiB, and read **151 GB** from the page
  file over the run.)
- **B (cascade, current)** is the only arm with **VRAM, pinned RAM, the CUDA1 helper and SSD streaming together**:
  tiers `VRAM 10783 | PINNED 3674 (5.99 GiB) | COLD 10119 (16.49 GiB)`, cold reads **94% prefetched**, 87.6% hit.
  Because its RAM budget is **pinned** rather than page-cache-dependent, it holds its speed at 128K context. The
  `--windows-memory-guard` is opt-in on top of this (measured separately below): with it the engine wrote **1,002
  MiB** to the page file over the run vs the stock arm's 1,425 MiB, and kept usable RAM (min 303 MiB / avg 7,961 MiB
  vs stock's 61 / 281 MiB). Its PCIe probe reads 14.5 GB/s on this x8/x8 link, so the rule sets `pcie_frac 0.40`.
- **C** keeps only part of the experts resident (the full complement, ~33 GiB, does not fit 32 GB minus headroom);
  the rest streams through the **OS file cache**, and the second card sits **idle**. Whether it performs is entirely
  down to that cache being warm: **29.7 t/s / 434 t/s** warm, ~200 t/s at 128K cold, where it **thrashed
  the pagefile and froze the PC** (826 s for one 5K reply). **D** is the fork's one-card cascade.

If the desktop stutters while generating on a tight machine, `--memory-guard` (alias `--windows-memory-guard`) makes
the engine yield - see **The memory guard** below. On Linux it pauses cold prefetch only.

### When the cascade applies

The cascade earns its keep only when experts **do not fit RAM** *and* there is a **second GPU** to hold a helper
cache. Otherwise the port is inert and behaviour is upstream's:

- **One card, RAM too small:** upstream's low-RAM mode (`--resident-experts` / `--mmap-experts`), or the fork's
  one-card cascade. Prefer the cascade (`--low-ram tiered`): its pinned budget gives **28.2 t/s / ~960 t/s** at 128K
  context, where the stock mode thrashed the pagefile.
- **Enough RAM (e.g. a 24 GB card with the model resident):** no cascade, no low-RAM mode; the model stays in RAM.
  Upstream's own 3090 benchmark (IQ3_XXS, 32K prompt, on a 165 GiB server) measured ~90 t/s decode / ~2,160 t/s
  prefill - the fork adds nothing here.
- **Smaller / experimental sizes** (Q2_0, IQ2_XS, Coder, Unsloth UD-Q4_K_XL with `--resident-budget-gib`): follow
  upstream's guidance; the cascade composes with the RAM-budget models when a second card is present.

### The exact rig

**The reference rig.** Every number in this file was measured on the PC below - it is *what the numbers are
measured on*, not a requirement. A smaller machine gets a smaller layout (setup and the tuner scale it; see
`docs/TUNING.md`).

| Component | Detail |
| --- | --- |
| CPU | AMD Ryzen 7 5800X (8C/16T, Zen 3, AVX2) |
| RAM | 32 GB DDR4-3600 |
| GPU 0 | NVIDIA RTX 5060 Ti 16 GB (sm_120) |
| GPU 1 | NVIDIA RTX 3060 12 GB (sm_86), the CUDA1 expert helper |
| Link | PCIe 4.0 **x8 / x8** |
| SSD | SK hynix Platinum P41 (PCIe 4.0 NVMe) |
| OS | Windows |

---

## 1. What the cascade is (and why upstream alone cannot do it)

Upstream aborts when a RAM tier is combined with a remote expert cache:

```cpp
// src/program/generate.cpp (upstream, line ~1367)
if (o.resident_cpu_experts && (!o.layer_split.empty() || remote_caches)) {
    std::fprintf(stderr, "strata generate: --resident-cpu-experts does not support layer splits or remote expert caches\n");
    return 2;
}
```

On a machine with enough RAM for the *whole* model that is fine - keep every expert resident in RAM. On a 32 GB
machine it is not: Qwen3.8-Flash-Next IQ3_XXS's experts are ~24 GB, and the second GPU's 6250-expert cache (the
CUDA1 helper, worth several t/s) is exactly the "remote expert cache" upstream forbids alongside a RAM tier. The
only legal upstream choice is `--mmap-experts` + the helper, which thrashes the OS page cache.

Cascade is the missing combination: **VRAM (CUDA0 cache) + CUDA1..3 helper + a pinned RAM budget + cold
SSD reads, all at once.** After the caches are filled, `TieredExpertSource::settle` sorts every `(layer, expert)`
into one of three tiers:

| Tier | What it is | Who reads it |
| --- | --- | --- |
| **VRAM** | resident in a GPU cache (CUDA0's primary, or a stage, or a CUDA1..3 helper) | the GPU |
| **PINNED** | the next experts by profile rank, up to the RAM budget, page-locked | the CPU pool, by memcpy |
| **COLD** | the rest, ordinary mapped file pages, read from the SSD one blob at a time | the CPU pool, by fault/prefetch |

Two details that matter:

- **No blob lives in two tiers.** `settle` is told which pairs another VRAM tier (a layer-split stage, a CUDA1..3
  helper) already holds (`also_vram`), so the pinned budget is spent on experts *no GPU* computes. This is why
  `settle` runs **after** the helper is filled.
- **A promoted expert keeps its RAM copy.** When the adaptive tier swaps a PINNED expert into VRAM, its pin is
  retained (`retained_pins()`), so a later eviction costs a memcpy, not an SSD re-read.

A profile is **mandatory**: `--tiered-experts` needs `--expert-profile` (the PINNED tier is "the profile's order
past the cache"). See `docs/TUNING.md` and `tools/make_profile.py`.

---

## 2. Results

### 2.1 Cascade's own arms (adaptive tier on/off)

Same rig, same model (IQ3_XXS), 32K-token prompt, 5K greedy reply at 128K context with `--kv int8
--kv-resident 20480`, vision on. **The headline above is the
comparison against stock upstream**; this table isolates what the *adaptive tier* adds on top of the cascade itself:

| Config | Decode | Expert-cache hit | Notes |
| --- | ---: | ---: | --- |
| Cascade, adaptive off (`--adapt-every 0`) | 43.2 t/s | 58.5% | tiering alone is throughput-neutral; it is what makes it *fit* (earlier run) |
| **Cascade + adaptive tier** (`--adapt-every 8 --adapt-swaps 32`) | **50.5 t/s** | **87.6%** | the current measured config |
| Cascade, old adaptive path (before Phase C) | 41.9 t/s | - | the adaptive swap stole the helper's pairs |

Prefill on the same run: **867 t/s** (32K prompt). The adaptive tier is upstream's own Plan v0.3 P6 machinery;
**Phase C** changed *which* experts it is allowed to promote (see section 3), which is where the gain comes from.
The headline's current best (51.8 / 900.5) is this config **plus `--memory-guard`** (see **The memory guard**
below).

> The "43.2 t/s / 58.5%" figure is *cascade with the adaptive tier off* - not the stock engine. **The headline
> carries the true stock numbers** (the separate upstream build, arm A).

### 2.2 Where the time goes (Phase C profile)

`waitCPU` is ~19 ms/window (~46% of a 42.3 ms window in the Phase C profile; GDN 14.9 + QSA 4.5) - the CPU pool
computing COLD experts, which no GPU policy can reach. The CUDA1 helper's own wait is not the bottleneck in that
profile. The lesson: the cascade already protects the helper; the remaining wall is COLD CPU work, which needs more
VRAM, a faster CPU, or a workload-specific profile to move (bench prompts overfit - see section 5).

---

## 3. Phase C - the adaptive tier, corrected

Upstream's adaptive tier pulls the most-routed missing experts into the CUDA0 cache every `--adapt-every` rounds.
On the cascade + helper setup that was a **regression** (41.9 t/s), for three reasons Phase C fixes:

1. **It stole the helper's experts.** A pair a CUDA1..3 helper already computes was pulled into CUDA0 too -
   duplicating GPU work and evicting a genuine miss. Phase C exposes `RemoteExperts::holds()` and **skips any
   candidate a helper holds**. The helper is no longer starved.
2. **It excluded the profile's tail.** Cold experts could never be promoted, capping the hit rate at 66%. Phase C
   **allows cold swap sources by default**; the tail can now rise into VRAM. `STRATA_ADAPT_NOCOLD=1` restores the
   old behaviour for A/B.
3. **It dropped the pin on promote.** Each promoted expert lost its RAM copy, so every later eviction re-read the
   SSD. Phase C **keeps the pinned copy** as the pair's cheap re-read home (`retained_pins()` replaces the old
   `demoted()`).

Both the `serve` loop and the interactive loop apply the same rule (two call sites in `generate.cpp`).

---

## 4. Configuration

The tuned two-card config (paths shortened; yours point at your model folder):

```
--pack <data>\packs\iq3_xxs
--native <data>\...-00001-of-00002.gguf
--ple-gguf <data>\...-00002-of-00002.gguf
--expert-profile <repo>\data\expert-profile.bin   # MANDATORY with --tiered-experts
--expert-cache auto
--tiered-experts --host-budget-gib 6 --host-reserve-gib 4
--spec 4 --spec-min-p 0.5
--mtp <data>\mtp\rt
--prefill auto:16384 --max-context 131072 --kv int8 --kv-resident 20480
--expert-cache-device1 6250 --vram-reserve-mib 1000
--adapt-every 8 --adapt-swaps 32 --vision
# --prefill auto:16384: a bigger prompt chunk reads each cold expert once per chunk instead of once per 8K
# chunk, so a 32K prompt read 71.2 GB of cold experts instead of 129.4 GB (prefill 605 -> ~900 t/s, decode
# unchanged).  `auto:N` only raises the ceiling - the engine still sizes the chunk to the free VRAM.  The
# chunk's host staging buffers scale with it, so setup.py writes auto:16384 only at 32 GB of RAM or more.
# (no flag needed) the helper-disjoint adaptive tier is ON for this layout by default; upstream's
# STRATA_DISJOINT_ADAPT defaults off.  Off here, the adaptive primary duplicates the helper's hot experts and
# decode fell 50.6 -> 42.0 t/s (IQ3_XXS, 32K/5K).  STRATA_DISJOINT_ADAPT=0 turns it off.
# optional, NOT in the recommended layout: --windows-memory-guard (alias --memory-guard) is for a page
# file on a slow disk.  On the 0.1.40 cold bench it cost decode while the cascade streams from a fast
# SSD - see "The memory guard" below.
```

plus `gpu: [0, 1]` and `layer_split: null`. Setup writes a `--host-budget-gib` **scaled to the PC** when a helper is
present (`min(6 GiB, free RAM - the 8 GiB reserve, ~45% of the primary card)`, floored at 2 GiB; `auto` on one card),
plus a `--expert-cache-device1..3` per helper card. The explicit `6` above is this rig's 32 GB / 16 GB-primary
optimum; a smaller PC gets a smaller pin. Setup also writes `--prefill auto:16384` at 32 GB of RAM or more (see
above). The `--memory-guard` is opt-in - see `docs/TUNING.md` section 1b. See
`docs/TUNING.md` for how to choose these numbers for a different machine.

### 4.1 The tuned layout

A short sweep on this rig settled the two budgets that matter, at **131072 context / `--kv int8
--kv-resident 20480`**. The 6 GiB row is the current run; the others are the earlier sweep kept for context:

| Cards | Pinned | Helper | Adaptive | Decode | Prefill | Hit | Outcome |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 2 | 4 GiB | 6250 | on | 49.1 t/s | 849 t/s | 84.8% | the earlier sweep's winning layout |
| 2 | 6 GiB | 6250 | on | **51.8 t/s** | **900.5 t/s** | 86.2% | **fastest here - the config above (current run, + `--memory-guard`)** |
| 2 | 8 GiB | 6250 | on | - | - | - | **fails at start**: the pin exhausts the driver's page-lock capacity; CUDA0's cache/buffers cannot allocate |
| 1 | 8 GiB | - | off | 28.2 t/s | ~960 t/s | 38% | best one-card config (`--low-ram tiered`) |

The two lessons: with a second card, **more pinned is not always better** - past ~6 GiB the pinned budget and a
16 GB CUDA0's cache fight over the prefill borrow, and 8 GiB fails outright. And on **one card** the optimum is the
opposite: pin the maximum and leave the adaptive tier **off** - it is a helper-protecting mechanism with nothing to
protect when there is no helper. The engine's own safety ceiling follows the same split (**6 GiB beside a helper,
24 GiB on one card**; `STRATA_TIERED_PIN_CAP_GIB` overrides).

Re-confirmed on the current engine. With everything else at the 6 GiB optimum, only `--host-budget-gib` was raised
to 8: the engine logged `only 0 MiB free once the slots are written ... shrinking the expert cache`, registered the
pin fully (`PINNED 4934 (7.99 GiB, 8 runs, 8.00 GiB locked)`) and then **`the R4 hit path could not allocate its
device buffers`** - the registration succeeded but had consumed the driver's page-lock headroom. The engine now
**caps the attempted registration to the card layout** (`tiered_pin_cap_bytes()`: 6 GiB beside a helper, 24 GiB on
one card; `STRATA_TIERED_PIN_CAP_GIB` overrides) and demotes any
expert past a refused/over-cap chunk to COLD, so the same 8 GiB request starts cleanly as
`PINNED 3655 (5.99 GiB, 6 runs, 6.00 GiB locked)` with `R4 hit path ON` - a smaller pin instead of a crash. The
tuner's `oom_reason()` / `pin_clamped()` treat a refused pin or an under-reached budget as a failed layout
(`tools/test_tuner_safety.py`).

### 4.2 What the engine prints at start (the lines to check)

```
strata generate: experts via the tiered source (mapped ...\experts.bin); tiers are set after the cache fill
strata generate: expert cache auto: 9.61 GiB free, 1000 MiB reserved (+73 MiB for the draft head) -> 3945 slots
strata generate: PCIe probe: 14.5 GB/s host->device (best of 14.5 14.5 14.4 14.5) -> pcie_frac 0.40 (default 0.55)
strata generate: CUDA1: 6250 additional experts, 10.12 GiB; results return through pinned host rows
strata generate: tiers: VRAM 10783 experts (17.48 GiB, 6250 held by another GPU, left to the standby list) | PINNED 3674 (5.99 GiB, 6 runs, 6.00 GiB locked) | COLD 10119 (16.49 GiB, paged from SSD) | budget 6.00 GiB | 3.9 s
strata serve: tiered decode misses: PINNED 67711 (RAM), COLD 108925 (SSD, 189306.2 MB, 102313 prefetched = 94%)
strata serve: decode expert cache hit rate: 87.6% (1710040 hits / 1952011 lookups)
```

These are the engine's own startup lines from the current Cascade arm (the `tiers:` line is the one to check). The
pinned counts move slightly run to run; `--expert-cache-device1 6250` is the helper shown here.

If the `tiers:` line does not appear, the cascade did not settle - check the config flags. If `COLD` reads are
almost all `prefetched`, `begin_layer` is doing its job; if the pinned budget is tiny, raise `--host-budget-gib`.

---

## The memory guard (opt-in)

When the page file is on a slow disk, a PC short on RAM freezes while the OS pages: the memory manager evicts other
processes' dirty pages to that disk and every hard fault is a seek. The engine can yield usefully because most of its
own footprint is **clean, file-backed** memory (the mapped experts, the mapped weights) that the OS can drop without a
page-file write and re-read from the model's SSD on demand. `--memory-guard` (alias `--windows-memory-guard`) lets
RAM fill, then makes the **engine** the cheapest victim instead of paging *your* apps out.

> **Not part of the recommended layout (engine 0.1.40).** A fresh cold 32K/5K run on the reference rig
> (IQ3_XXS, `--host-budget-gib 4`, the CUDA1 helper) decoded at ~41 tok/s with the guard **off** and
> ~16-25 with it **on** (single runs, high variance; the guard's working-set trim and `VERY_LOW` priority
> evict the COLD expert pages decode re-reads).  Prefill was unaffected (~640 tok/s either way).  Use the
> guard only when the page file is on a slow disk and the desktop stutters, and let `tune_cascade.py
> --guard` decide - it keeps the flag only when it costs no throughput.

**Windows** does the full yield: it watches a free-RAM target (`STRATA_MEM_GUARD_KEEP_FREE_MIB`, default 1024) and the
OS's own low/high memory notifications (an optional predictive decline trigger, `STRATA_MEM_GUARD_PREDICT=1`, is off by
default). On pressure it lowers the engine's memory priority and releases the deficit with a proportional `soft`
working-set ceiling (set to `entry_ws - deficit`); it **holds** `VERY_LOW` priority for the whole low period so the OS
keeps choosing the engine's clean, file-backed pages over another app's dirty ones. `hard` (`EmptyWorkingSet`) and `off`
remain. The `--mmap-experts`/`--resident-experts` sources also pause cold prefetch while pressure lasts; the **cascade
does not by default** - pausing on every dip collapsed decode to ~20 t/s, so it is gated behind
`STRATA_PREFETCH_PAUSE_ON_PRESSURE=1`. Windows has no per-range drop to call: `OfferVirtualMemory` is rejected for file
mappings and, where it works, discards contents and blocks access until `ReclaimVirtualMemory` - the original comment
in `tiered_source.cpp` was right. **On Linux the guard's only lever is the prefetch pause** (the kernel already
reclaims the clean expert pages); the Linux path is **unmeasured**.
Knobs and the harness: [`docs/TUNING.md`](docs/TUNING.md) section 1b.

Measured on Windows 11, 32 GB RAM, RTX 5060 Ti + a helper card, IQ3_XXS, `--tiered-experts --host-budget-gib 4`,
32K prefill / 5K decode, one run each. This PC runs so close to full that the guard acts even in a *normal* run:

| arm | prefill tok/s | decode tok/s | page-file writes | releases |
| --- | --- | --- | --- | --- |
| no guard | 838.4 | 48.0 | ~0 (page-in to ~400k/s) | - |
| `soft` (default) | 810.4-834.0 | 45.3-49.4 | 0-3k/s | 2-14 soft, 0-1 hard |
| `hard` | 816.0 | 47.1 | 0 | 5 hard |

The same effect in a straight **cascade + guard vs stock** pair, measured on the earlier same-base run (Windows 11,
32 GB, IQ3_XXS, 32K/5K, one run each; stock = a port-free `origin/main` build with `--mmap-experts` + the CUDA1
helper, which does not accept the guard flag, so it is the "no guard" arm):

| | cascade + guard | stock (no guard) |
| --- | --- | --- |
| page-file bytes **written** over the run | **1,002 MiB** | 1,425 MiB |
| page-file bytes read over the run | **18,724 MiB** | 151,453 MiB |
| free RAM min / average | 303 / 7,961 MiB | **61 / 281 MiB** |
| decode / prefill | **49.7 / 846.4 tok/s** | 37.9 / 245.0 tok/s |

With the guard the engine wrote about a third less to the page file and kept the machine usable (min ~300 MiB,
average ~8 GiB free) where the stock arm sat pinned at ~61 MiB and read **151 GB** through the page file. The gap in
throughput is the cascade's pinned tier, not the guard - the guard's job is the page file and the free-RAM floor. The
engine's locked footprint (registered experts plus pinned host KV) cannot be trimmed, so the ceiling never goes below
it; moving the page file to an SSD is still the real fix.

---

## 5. Method and caveats

- **Rig (the headline's hardware table):** Ryzen 7 5800X (8C/16T, AVX2), 32 GB DDR4-3600, RTX 5060 Ti 16 GB (CUDA0) +
  RTX 3060 12 GB (CUDA1), PCIe 4.0 x8/x8, SK hynix Platinum P41 NVMe, Windows. Two cards of different sizes is the
  point: CUDA0 serves layers, CUDA1 holds a helper cache.
- **Bench:** the 32K-prefill / 5K-decode bench, prompt `tools/cascade_bench/prompt32k.txt` (the tuner runs the same
  one), at **128K context** (`--max-context 131072`) with **`--kv int8 --kv-resident 20480`** (8-bit KV, 20,480
  cells/layer kept in VRAM; the rest streams from pinned RAM) and vision on. Only this bench is quoted. The context
  and KV quant matter: they set how much VRAM is left for experts and how much pinned RAM the KV costs, which is
  exactly what the cascade and the stock low-RAM mode are competing for on a 32 GB PC.
- **The stock arm** (headline A): a clean upstream `origin/main` (`6f32ec0`) build, no port, in a throwaway
  worktree. It is upstream's best dual-GPU attempt (`--mmap-experts` + the
  CUDA1 helper - upstream refuses a RAM tier with it).
- **The comparison is same-base:** the cascade and the stock engine are both built from `6f32ec0`, so the headline
  isolates the port and its configuration, not a version difference.
- **Run count:** each arm is a single run. The headline gaps (51.8 vs 42.5 decode, 900 vs 253 prefill) are far larger
  than the +/-5% run-to-run spread.
- **Do not overfit.** The bench prompt is one code block repeated many times, so its routing is unusually regular;
  numbers tuned on it can mislead on a real workload. Final tuning should use what you actually run - see
  `docs/TUNING.md`.
- **Linux:** the port's Linux path is implemented (its own `mmap`/`MADV_*`/`O_DIRECT` code) but has **not been
  built or measured here**. The Windows numbers above are the only measured ones. Please report Linux results.
- **Numbers move run to run** with the free VRAM the vision encoder and KV leave; treat +/-5% as noise.

---

## 6. What did NOT help (measured, do not redo)

Earlier sweeps on a 10K / 1000-token bench (baseline 40.8 / 45.8 t/s):

| Change | Result | Why |
| --- | --- | --- |
| `--spec-split` | +1% | overlap is real but doubles CUDA1 launches |
| `--spec-min-p 0.3` | -10% | extra drafts cost more than they return |
| `--pool-workers 15` | -66% | oversubscribes the 8 physical cores; the pool is contention-bound |
| Async CUDA1 helper thread | wash | `plan` fell, `CPU` rose by the same amount |

Higher `--vram-reserve-mib` costs CUDA0 cache slots and therefore decode. `--host-budget-gib` is **not** a free
speed knob: on one card, more pinned helps (up to the engine's 24 GiB one-card ceiling), but with a helper and a
16 GB CUDA0 it competes with the cache for the prefill borrow and **8 GiB fails outright** (section 4.1). Tune it on
your own rig - the shipped tuner does exactly this sweep.
