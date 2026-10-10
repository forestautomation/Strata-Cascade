# Implementation guide - the Strata-Cascade expert source

**What this is.** How to re-apply the Cascade changes onto a fresh upstream Strata checkout. Upstream rewrites the
expert-source files often, so the port has to be re-landed regularly. This file is the source of truth for *what
the changes are and where they go*; `STRATA-CASCADE.md` is the *results* document and `docs/TUNING.md` is the
*how to tune it* document.

**Target of the current port.** Upstream `origin/main` (`fb58e0db`, v0.1.41) -> branch `cascade-042` in `<repo>` - the
port re-applied as a single merge commit on the new upstream.

This fork's own additions are the Windows implementation; the `also_vram` mask that lets the pinned RAM tier run
*together with* the CUDA1..3 helper cache (upstream refuses the pair, and it is what makes the cascade faster); and
Phase C (`promote_to_vram` / `retained_pins`), which stops the adaptive tier duplicating experts a helper already
holds and keeps a pinned copy when it promotes one. **Attribution:** the tiered source adapts the algorithm of
upstream [PR #80](https://github.com/Niko1221/Strata/pull/80) by @andrewcoul (closed upstream in favour of folding
the tier into `FileExpertSource`), with parts rewritten so it works on Windows.

**Canonical artifacts kept for re-ports:**
- This repo on branch `cascade-042` - the port applied as one merge commit on top of upstream 0.1.41. Every artifact below
  is recoverable from it alone.
- Archive of the original 0.1.30 port as patches: `<port-patches>\0001..0004-*.patch` (outside the repo).
- The big new file `src/core/tiered_source.cpp` (~1044 lines, verbatim) is NOT reproduced here. Recover it with:

  ```powershell
  git -C <this-repo> show cascade:src/core/tiered_source.cpp > <new-checkout>\src\core\tiered_source.cpp
  ```

---

## 0. Why this port exists (the one fact that decides everything)

Upstream aborts when a RAM tier meets a remote expert cache (`src/program/generate.cpp`, line 1367):

```cpp
if (o.resident_cpu_experts && (!o.layer_split.empty() || remote_caches)) {
    std::fprintf(stderr, "strata generate: --resident-cpu-experts does not support layer splits or remote expert caches\n");
    return 2;
}
```

So a PC whose RAM cannot hold every expert has no legal way to combine a RAM tier with the CUDA1..3 helper. Cascade
is that combination: **VRAM + CUDA1..3 helper + pinned RAM budget + cold SSD reads, at once.** If upstream ever
allows the two together, this port can be dropped.

---

## 1. Files the port touches (the real change surface)

| File | Change | Where |
| --- | --- | --- |
| `src/core/tiered_source.cpp` | **new file**, ~1044 lines | `TieredExpertSource`, both platform paths |
| `include/strata/core/expert_source.hpp` | +`TieredExpertSource` class (+129 lines), +2 base virtuals | `read_into`, `streams_from_ssd` on `ExpertSource` |
| `include/strata/core/remote_experts.hpp` | +6 lines, `holds()` | exposes the helper's live set |
| `src/program/generate.cpp` | +137/-10, the integration | options, validation, source select, settle, adapt, report |
| `src/prefill/prefill.cpp` | +30/-13 | `Stager` SSD ring + `read_into` |
| `src/core/expert_cache.cpp` | +50/-4 | `bounce_copy` for registration-straddling blobs |
| `serve/server.py` | +44/-13 | `layer_split: null`, vision `device`, tiered RAM warning |
| `CMakeLists.txt` | +7/-1 | link `tiered_source.cpp`; add `native_mmap_test` |
| `src/core/native_mmap_test.cpp` | **new file**, 60 lines | standalone mapping test |
| `setup.py` | tiered detection, default and flags | section 8.1 |
| `tools/cascade_bench/` | **new**: the shipped layout tuner | section 8.2 |

---

## 2. File-by-file

### 2.1 `src/core/tiered_source.cpp` (new) + `expert_source.hpp`

`TieredExpertSource : ExpertSource` maps `experts.bin` (writing it from a native pack's shard 1 via `set_gguf` on
first use) and sorts every expert into **VRAM / PINNED / COLD** once the caches are filled (`settle`). Public API:

- `open(pack_dir, n_layers, n_expert, err)`, `set_gguf(shard1)`, `close()`
- `settle(cache, also_vram, profile, budget_bytes, reserve_bytes, threads, err)` - the tier sort. `also_vram` is
  a per-pair mask of experts a *different* VRAM tier holds; those are marked VRAM, never pinned.
- `promote_to_vram(layer, expert)` - **Phase C**: an adaptive promotion KEEPS the pinned copy (`retained_pins_++`).
- `set_residency(host_res)` - lets `begin_layer` prefetch an expert the adaptive swap evicted.
- `blob` / `pinned` / `device_alias` / `begin_layer` / `read_into` / `streams_from_ssd`
- counters: `pinned_reads`, `cold_reads`, `cold_read_bytes`, `cold_prefetched_reads`, `retained_pins`, `note`.

**Platform split (both implemented):**

| | Windows (`#if defined(_WIN32)`) | Linux (`#else`) |
| --- | --- | --- |
| address layout | **two** regions - no in-place anonymous remap, so PINNED lives in `pin_base_` | **one** region - `MAP_FIXED` anonymous remap replaces pages in place |
| cold stream | `CreateFileA` with `FILE_FLAG_NO_BUFFERING \| OVERLAPPED` | `open(O_RDONLY \| O_DIRECT)` |
| prefetch | thread pool | `madvise(MADV_WILLNEED)` / `MADV_DONTNEED` |
| lock | `cudaHostRegister` in 1 GiB chunks, capped at `tiered_pin_cap_bytes()` (6 GiB beside a helper, 24 GiB on one card; `STRATA_TIERED_PIN_CAP_GIB`), a refused/over-cap chunk demotes its experts to COLD | `cudaHostRegister` in place; a refused run is demoted to COLD |
| free RAM | `GlobalMemoryStatusEx` (standby-aware) | `MemAvailable` |

Both read `STRATA_NO_DIRECT_STREAM` (disable the buffered-bypass reader) and `STRATA_SYNC_PREFETCH` (the old
madvise-on-pool-thread arm).

Base-class additions in `expert_source.hpp`:

```cpp
virtual void read_into(const uint8_t* src, uint8_t* dst, size_t n) const { std::memcpy(dst, src, n); }
virtual void read_into_many(const uint8_t* const* src, uint8_t* const* dst, const size_t* n, size_t count) const;
virtual bool streams_from_ssd() const { return false; }
```

`read_into` lets an unpinned tiered blob be read with one unbuffered read instead of ~400 page faults;
`read_into_many` is its batched form (default: `read_into` one by one) and lets `TieredExpertSource` merge a batch's
**contiguous** cold blobs into one request - a layer's experts sit back to back in `experts.bin`, and request size is
what caps an NVMe (~2.3 GB/s at 2 MiB vs ~6.6 GB/s at 8 MiB on the reference rig); `streams_from_ssd()` lets the
prompt path keep more reads in flight.

### 2.2 `include/strata/core/remote_experts.hpp`

```cpp
bool holds(int64_t layer, int64_t expert) const {
    return cache_.valid() && cache_.slot_of(layer, expert) >= 0;
}
```

Used by the adaptive tier so a pair this helper already computes is never pulled into the primary cache too.

### 2.3 `src/program/generate.cpp` (the integration)

- **`struct Options`**: `tiered_experts`; `host_budget_gib` (< 0 = auto); `host_reserve_gib` (default 8).
- **`usage()`**: the `--tiered-experts` / `--host-budget-gib` / `--host-reserve-gib` block. *(Upstream's
  `--expert-profile-save` block sits at the same spot - keep both.)*
- **arg parsing**: the three flags.
- **validation** (after the resident-cpu check):
  - `--tiered-experts` is mutually exclusive with `--mmap-experts` / `--resident-experts`;
  - `--tiered-experts` **requires `--expert-profile`**.
- **source selection**: `if (o.tiered_experts) { tiered_src.set_gguf(o.native_preset); tiered_src.open(...); }`
  before the `else if (o.mmap_experts)` chain.
- **`vram_elsewhere`**: the per-pair mask (a stage or CUDA1..3 helper holds it), filled where the old `claimed`
  was. **`settle` runs after the stages and helpers are filled**, with `also_vram = vram_elsewhere.data()`.
  `STRATA_LEGACY_PIN=1` is the pre-fix A/B arm (pins without the mask).
- `tiered_src.set_residency(host_res.data())` after `host_res` is set up.
- **adaptive commit** (both the serve loop ~4613 and interactive loop ~6277): after `host_res[i] = slot`,
  `if (o.tiered_experts) tiered_src.promote_to_vram(i / n_expert, i % n_expert);`.
- **adaptive candidate** (both loops): skip a candidate a helper holds (`remote[rr]->holds(l,e)`); allow COLD by
  default, `STRATA_ADAPT_NOCOLD=1` to exclude. These are the Phase C hunks.
- **serve report**: the `tiered decode misses: PINNED ... COLD ...` line and the `retained pins` line.

### 2.4 `src/prefill/prefill.cpp`

`Stager` gains `static constexpr int kRingSsd = 48`, a `source` pointer, and uses `source->read_into` when set (a
job with `from` still uses `copy_blob`; else memcpy). `init(blob_bytes, nthreads, ring)` takes the ring; `Prefill::init`
sizes threads and ring from `src->streams_from_ssd()` (SSD: 2-8 threads, ring 48; in-place GGUF: 32, `4*threads`;
else 4, 16).

A worker's batch collects the `source`-backed jobs (`Job::from == nullptr`) and calls `source->read_into_many` once (the
SSD profile batches like the file one: `STRATA_STAGER_BATCH`, default 8). `TieredExpertSource::read_into_many` merges
each run of contiguous file blobs (up to 8 MiB) into one unbuffered read and copies the parts out; pinned/locked blobs,
and a run whose aligned end passes the file, fall back to `read_into`. Measured (IQ3_XXS, 32K/5K, `--host-budget-gib 6`,
CUDA1 helper): cold reads 41,070 (71.6 GB) at ~3.0 GB/s -> 12,998 (77.3 GB) at ~6.6 GB/s, **prefill 760.6 -> 906.9 t/s**.

### 2.5 `src/core/expert_cache.cpp`

`bounce_copy()`: a blob straddling a page-locked registration edge is copied via a pinned bounce buffer. Called
from `fill_slot`, `fill_slot_blocking`, `fill_slot_queued` when `cudaMemcpy*` returns `cudaErrorInvalidValue` (and
`cudaErrorMemoryAllocation` in `fill_slot_queued`).

### 2.6 `serve/server.py`

- `engine_args`: `layer_split` of `null` / `"none"` / `"off"` keeps every GPU visible **without** splitting
  (helpers only).
- `Vision`: optional `device` (runs the encoder on its own card via `CUDA_VISIBLE_DEVICES`).
- `narrate_start(..., facts)`: records `facts["tiered"]` when the log shows `via the tiered source`.
- `warn_tight_ram(arena_mib, tiered)`: **suppressed** when tiered (the arena is the SSD file, not RAM).

### 2.7 `CMakeLists.txt` + `src/core/native_mmap_test.cpp`

Add `src/core/tiered_source.cpp` to `strata_engine`; add the standalone `native_mmap_test` target + `add_test`.
The test covers variable per-layer strides, exact blob bytes, axis bounds, truncation, geometry mismatch, close and
canonical compatibility.

---

## 3. Phase C in one place

Phase C corrects upstream's adaptive tier (`--adapt-every` / `--adapt-swaps`) for the cascade + helper setup:

| Behaviour | Symbol | Behaviour before |
| --- | --- | --- |
| skip a candidate a CUDA1..3 helper holds | `RemoteExperts::holds()`, called in both adapt loops | stole the helper's experts |
| allow COLD swap sources by default | `STRATA_ADAPT_NOCOLD=1` opts out | excluded cold -> hit capped at 66% |
| keep the pinned copy on promote | `TieredExpertSource::promote_to_vram` -> `retained_pins()` | `demoted()` dropped the pin -> SSD re-read |

Measured effect: 41.9 -> 50.5 t/s, hit 58.5% -> 87.6% (current run).

---

## 4. A/B switches

| Env var | Effect |
| --- | --- |
| `STRATA_LEGACY_PIN=1` | pin the pre-Phase-C set (ignore the `also_vram` mask) |
| `STRATA_ADAPT_NOCOLD=1` | exclude cold experts as adaptive swap sources |
| `STRATA_NO_DIRECT_STREAM=1` | disable the unbuffered cold reader |
| `STRATA_SYNC_PREFETCH=1` | prefetch on the pool's thread (old arm) |
| `STRATA_STAGER_THREADS`, `STRATA_STAGER_RING` | prefill staging knobs |

---

## 5. Applying order (the parts that must stay ordered)

1. Fresh checkout at the target upstream; branch (e.g. `cascade`).
2. `git am --3way <port-patches>\0001-*.patch`, or apply the single-commit diff.
3. Resolve conflicts by these rules:
   - `ExpertSource` base: **union** upstream's new virtuals with `read_into` / `streams_from_ssd`.
   - `TieredExpertSource`: additive; place near `ArenaExpertSource`.
   - `generate.cpp` `usage()`: keep upstream's block **and** ours.
   - `prefill.cpp` `Stager`: keep both upstream's `Job::from`/`copy_blob` and our `source`/`read_into`.
4. Relocate `generate.cpp` hunks **by symbol, not line**: `Options`, `usage`, arg parsing, validation, the source
   chain, `vram_elsewhere`, the `settle` call (after helpers), both adaptive loops, the serve report.
5. **`settle` must run after the layer-split stages and the CUDA1..3 helpers are filled** - otherwise the pinned
   budget goes to blobs a GPU already holds.
6. Re-check 0: if upstream lets `--resident-cpu-experts` coexist with the helper, the port can be dropped.
7. Build, run `native_mmap_test`, then the 32K/5K bench (`tools/cascade_bench/`).

---

## 6. Update loop (keeping `cascade` = upstream + one commit)

```powershell
git fetch origin                 # origin fetches upstream
git rebase origin/main           # replay the clean commits on the new release
# resolve conflicts (usually 1-3 files, per 5), rebuild, bench
git push <your-remote> cascade
```

Keep it **rebased**, not merged: conflicts then stay isolated to the port commit, and the branch stays free of
machine-specific paths and the local benchmark harness.

---

## 7. Pitfalls

- **Helper + upstream resident mode is illegal** (0). Cascade must be the source.
- **`--expert-profile` is mandatory** with `--tiered-experts` - the PINNED tier is the profile's order past the
  cache. Generate one with `tools/make_profile.py`.
- **`settle` after the helpers**, not before (the `also_vram` mask).
- **A straddling blob** needs `bounce_copy`; without it `cudaMemcpy` fails with "invalid argument" and the cache
  refuses the fill.
- **Windows pins in chunks** and refuses past a limit; a refusal demotes the lowest-ranked pinned expert.
- **The bench prompt overfits.** Tune with what you actually run (`docs/TUNING.md`).

---

## 8. The two non-engine surfaces (so a fresh clone runs the cascade)

The engine port alone leaves `--tiered-experts` unreachable: upstream's `setup.py` never emits it, and the tuner
used to live in a gitignored local scratch folder. Two surfaces fix that. Both are **additive** and **degrade to
upstream** when the ported engine is absent, so a rebase only ever has to merge them around upstream's changes.

### 8.1 `setup.py`

- **Detection:** `tiered_engine(eng)` probes `<engine>/strata.exe --help` for `--tiered-experts` (cached per path,
  time-bounded, `False` on any error). The version number cannot identify the port - the fork re-lands on the same
  upstream version - so the binary is the only honest signal. No engine or CMake change is needed.
- **The default:** `tiered_should_default(model, ram, found)` = the model does not fit RAM (the low-RAM test) AND
  `together_ok(found)` offers a second shareable card. It looks at the cards **detected**, not chosen: the low-RAM
  recommendation deliberately picks one GPU, and the cascade's win is turning the second into a helper. On **one**
  card with RAM too small, setup **asks** ("Use the cascade on this card?", default **yes** - this fork is the
  cascade fork); a PC whose RAM holds the model has no low-RAM mode, so it is never asked or enabled.
- **The flag:** `--low-ram` gains a `tiered` choice. On the ported engine the auto path sets `tiered = True` and
  extends `chosen`/`multi` to the detected pair; on an upstream engine an explicit `--low-ram tiered` falls back
  with a warning.
- **Emission:** `--tiered-experts --host-budget-gib <scaled>` plus `--expert-cache-device1..3` from
  `tiered_helper_counts` (each helper card's VRAM x 0.82 / the model's bytes-per-expert; **do not raise the
  fraction** - an over-large count makes `RemoteExperts::open` refuse to start, it does not clamp). The budget
  comes from `cascade_host_budget(helpers, ram, primary_gib)`: `auto` on one card, else `min(6 GiB, free RAM - the
  8 GiB reserve, ~45% of the primary card)`, floored at 2 GiB (the same rule as the tuner's `sweep_budgets`).
  With a helper the config's `layer_split` is **null**, not `"auto"`: the extra cards are helper caches, not
  pipeline stages, and the engine refuses a split beside a helper when no GPU is left over. `cascade_split_off()`
  detects it, and `upgrade_config()` rewrites the split out of an older config on the next start/update.
- **Guards:** `split_mmap()` and `offer_together()` return early on a `--tiered-experts` config (the cascade has
  its own layer handling; the resident/mmap rewrites do not apply).
- **Tests:** `tools/test_setup_tiered.py`, plus `tools/test_setup_golden.py` proves upstream output is unchanged
  when the engine is not ported.
- **`--tune-cascade`:** setup runs the shipped tuner (section 8.2) with the RAM and helper GPUs it detected, then
  copies the winning layout over the config; an interactive install also offers it.

### 8.2 `tools/cascade_bench/`

The shipped layout tuner (Python core + Windows/Linux wrappers + the bench prompt). It writes its CSV, winning
config and logs beside itself (or to `--out`), never into a gitignored folder. See `docs/TUNING.md`.
