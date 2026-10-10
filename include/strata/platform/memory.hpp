// include/strata/platform/memory.hpp - plan v0.3 P0.1/P1: keep a large host region resident.
//
// `cudaHostRegister` refuses the 31.6 GiB expert arena on Windows, and an unlocked arena is trimmed under memory
// pressure (the CPU pool then swung 2x between runs). With the n-gram table out of RAM there is headroom to lock
// it instead: raise the process's minimum working set by the region's size, then VirtualLock it (Windows needs
// only SeIncreaseWorkingSetPrivilege, which ordinary accounts hold). Linux: mlock.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// #243, Windows: the GPU's shared (non-local) memory budget and this process's use of it, from DXGI
/// (IDXGIAdapter3::QueryVideoMemoryInfo, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL) for the adapter whose LUID is the
/// 8 bytes at `luid` (cudaDeviceProp::luid).  Page-locked host memory the GPU maps is charged there.  False (and
/// `why` says so) when the query is not possible - always elsewhere than Windows.
bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why);

/// The machine's physical RAM in bytes (0 when unknown).
uint64_t total_physical_memory();

/// #357/#577: whether the OS file cache could keep the `read_bytes` the expert files are read for, beside
/// `arena_bytes` of RAM held by the engine's own copy of the experts and `margin` for everything else, with `avail`
/// bytes of RAM available.  The file tier passes only the expert bytes it really reads from the files (the experts
/// outside its resident RAM copy) and the RAM that copy really holds - not every shard's bytes and the requested
/// budget, which on a 96 GB PC (#577) made the file tier read unbuffered when the cache could keep its reads.
/// #1194: the file tier's reads are a skewed set (the GPU cache and the RAM copy took the hottest experts; the same
/// few of the rest come back token after token), so the cache does not have to hold all `read_bytes` to serve most of
/// them.  With `hot_fraction` < 1 it is enough that the room holds that share of them, but never less than
/// `floor_bytes` (below it the cache holds nothing worth having) and never more than every read.  The defaults ask for
/// every byte, as the Windows rule does.  Measured on Linux (#1194, a Tesla P100 box under 20 to 32 GB cgroups): through
/// the cache beat the unbuffered reads at every room tried, down to 7% of the read bytes (decode +7% to +118%, a third to
/// a sixth of the drive traffic).
inline bool file_cache_keeps(uint64_t avail, uint64_t arena_bytes, uint64_t read_bytes,
                             uint64_t margin = 4ull << 30, double hot_fraction = 1.0, uint64_t floor_bytes = 0) {
    const uint64_t room = avail > arena_bytes + margin ? avail - arena_bytes - margin : 0;
    uint64_t need = (uint64_t) ((double) read_bytes * hot_fraction);
    if (need < floor_bytes) need = floor_bytes;
    if (need > read_bytes) need = read_bytes;
    return room >= need;
}

/// Whether `advise_willneed` asks the OS for anything: not on Windows, nor with STRATA_READ_AHEAD=0.
bool read_ahead_enabled();
/// Asks the OS to start reading [p, p + bytes) of a file mapping, without waiting.  Linux reads at most one
/// readahead window per request, so the range is asked for in 128 KiB steps.
void advise_willneed(const void* p, uint64_t bytes);
/// The same for [offset, offset + bytes) of an open file.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);

// ---- the memory-pressure guard (opt-in; see --memory-guard) -------------------------------------
//
// When the page file is on a slow disk, a PC that is short on RAM freezes while the OS pages: the
// memory manager evicts other processes' dirty pages to that disk and every hard fault is a seek.
// The engine can do something useful about it because most of its own footprint is CLEAN, file-backed
// memory (the mapped experts, the mapped weights) that the OS can drop without a page-file write and
// re-read from the model's SSD on demand.
//
// Windows: the guard lets the machine use RAM up to the point where the OS actually needs it, then
// yields on demand.  It uses the OS's own low/high memory notifications where available, a target of
// `keep_free_mib` free RAM, and a predictive trigger for a fast decline.  On pressure it lowers this
// process's memory priority and pauses prefetch, so the OS reclaims the engine's cheap pages instead
// of paging other processes out, and releases memory at one of three levels: a proportional `soft`
// working-set ceiling (the OS's LRU keeps the hot pages), an immediate `hard` `EmptyWorkingSet`, or
// nothing (`off`, priority + pause only).
//
// Linux: the same state machine samples `/proc/meminfo` `MemAvailable` (plus available commit) and,
// on pressure, pauses prefetch - `memory_pressure_low()` goes true.  The per-process memory priority,
// the working-set ceiling and the OS notifications are Windows-only and are ignored here; the kernel
// already reclaims the engine's clean, file-backed expert pages itself, so there is no trim.
//
// `memory_pressure_low()` is set while the guard reports pressure, so a source can stop prefetching
// that would only add more pages.

/// How the guard releases the engine's working set while it reports pressure.  Windows-only; Linux
/// has no working-set ceiling, so it ignores this and only pauses prefetch.
enum class MemGuardTrim {
    Off,   ///< no release: only lower memory priority and pause prefetch
    Soft,  ///< a proportional working-set ceiling; the OS trims lazily and keeps recent pages
    Hard,  ///< `EmptyWorkingSet` to the standby list at every trigger (the pre-ladder behaviour)
};

struct MemoryGuardConfig {
    int poll_ms = 250;                ///< how often to sample
    uint64_t min_avail_mib = 1024;    ///< fallback floor: pressure while free physical memory is under this
    uint64_t keep_free_mib = 1536;    ///< the free-RAM target the guard holds; pressure under it
    uint64_t min_commit_mib = 3072;   ///< ... or available commit (RAM + page file) is under this
    uint64_t emergency_mib = 512;     ///< below this (or the OS signals low and free RAM is under twice
                                      ///< this), release hard even in soft mode.  The floor is the driver;
                                      ///< the OS notification only widens it, so a cliff still releases with
                                      ///< `notify=0` and off-Windows (there is no notification there).
    uint64_t recover_mib = 512;       ///< release toward (and clear) this far above `keep_free_mib` (hysteresis)
    uint64_t release_min_mib = 512;   ///< smallest release once the guard acts (keeps a release meaningful)
    int retrim_ms = 1000;             ///< while pressure lasts, act again this often (0 = only on entry)
    int cooldown_ms = 2000;           ///< after recovery, ignore soft pressure this long (emergency still acts)
    MemGuardTrim trim = MemGuardTrim::Soft;  ///< Windows-only trim mode; see MemGuardTrim (ignored on Linux)
    /// How a genuine cliff (free RAM under `emergency_mib`) releases, in `soft` trim mode.  `Hard`
    /// (default) is `EmptyWorkingSet`: it looks blunt but is what actually rescues a real cliff - a
    /// proportional soft ceiling sized from the deficit is far too small off a multi-GiB working set, so
    /// RAM stays pinned and the OS thrashes the engine (measured: decode 22.5 vs 50.6 t/s, 30.8M vs 11.9M
    /// faults, 371 GB vs 24.9 GB read back).  `Soft` is kept for experiments only.  Ignored when `trim`
    /// is `Off` or `Hard` (those already decide).
    MemGuardTrim emergency_trim = MemGuardTrim::Hard;
    bool priority = true;             ///< Windows-only: lower this process's memory priority on pressure
    bool notify = true;               ///< Windows-only: also use the OS low/high memory notification as a trigger
    bool predictive = false;          ///< opt-in (`STRATA_MEM_GUARD_PREDICT=1`): start trimming early when free RAM
                                      ///< is falling fast.  Off by default: the reactive floors plus the OS's own
                                      ///< low-memory notification are enough, and a hand-tuned slope/band over-trimmed
                                      ///< on a loaded PC (a wide band pulsed the guard on every transient dip).
    double predict_slope_mib_s = 128.0; ///< a free-RAM decline this fast (MiB/s) counts as "falling fast"
    uint64_t predict_band_mib = 512;    ///< ... within this much of keep_free; used only when `predictive`
    bool verbose = false;             ///< log every sample, not only the transitions
    bool stats = false;               ///< log the monitor thread's own CPU periodically
};

/// What the guard thread has done since it started, for tests and for the overhead question.
struct MemoryGuardStats {
    uint64_t samples = 0;
    uint64_t hard_trims = 0;      ///< `EmptyWorkingSet` calls
    uint64_t soft_trims = 0;      ///< working-set ceilings applied
    uint64_t monitor_cpu_ns = 0;  ///< CPU burned by the monitor thread itself
};

/// One-shot reading, for logs and tests.  False when even the physical size is unknown.
struct MemorySample {
    uint64_t avail_phys = 0;
    uint64_t total_phys = 0;
    uint64_t avail_pagefile = 0;
    uint64_t pagefile_usage = 0;  ///< this process (GetProcessMemoryInfo)
    uint64_t working_set = 0;     ///< this process
};
bool memory_sample(MemorySample& out);

/// Start the guard (idempotent).  False with `why` when unsupported or the thread cannot start.
bool memory_guard_start(const MemoryGuardConfig& cfg, std::string& why);
/// Stop and join the guard thread (best effort; also joins at exit).
void memory_guard_stop();
/// True while the guard reports memory pressure.  Always false when the guard is off or off-Windows.
bool memory_pressure_low();
/// Snapshot of the guard thread's work so far (all zero when the guard is off or off-Windows).
MemoryGuardStats memory_guard_stats();

/// What the guard should do this poll, from the samples and the previous state.  Pure, so the state
/// machine can be unit-tested without a GPU or a memory hog (see `memory_test.cpp`).
enum class GuardAction {
    None,     ///< idle: no pressure, no action
    Enter,    ///< pressure began: start acting and hold
    Stay,     ///< already acting: release the remaining deficit (or an emergency release)
    Recover,  ///< pressure cleared: resume and start the cooldown
};

struct GuardDecision {
    GuardAction action = GuardAction::None;
    uint64_t release_mib = 0;   ///< bytes (MiB) to release on Enter/Stay
    bool emergency = false;     ///< a genuine cliff: release hard even in soft mode
};

/// `avail_mib` is 0 when the sample is unavailable; `commit_mib` is the available commit and is 0 both
/// when the sample is unavailable (`have_commit` false) and when commit is genuinely exhausted
/// (`have_commit` true, a real pressure signal - the old signature could not tell the two apart);
/// `os_low` is the OS's own low-memory notification (always false off-Windows); `in_cooldown`
/// suppresses a new Enter; `slope_mib_s` is the EWMA free-RAM decline (positive = falling).
GuardDecision memory_guard_decide(const MemoryGuardConfig& cfg, bool low, uint64_t avail_mib,
                                  uint64_t commit_mib, bool have_commit, bool os_low, bool in_cooldown,
                                  double slope_mib_s);
/// What the OS says this process read from storage: `read_bytes` (/proc/self/io: bytes fetched from the block layer on
/// its behalf, page faults on a mapping included, page-cache hits not) and `major_faults` (/proc/self/stat).  Linux
/// only; `valid` is false elsewhere.  Take one at the start of a request and one at its end.
struct ProcIo {
    bool valid = false;
    uint64_t read_bytes = 0, major_faults = 0;
};
ProcIo proc_io_sample();

}  // namespace strata::platform
