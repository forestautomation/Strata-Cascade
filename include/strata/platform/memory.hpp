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
    int poll_ms = 500;                ///< how often to sample
    uint64_t min_avail_mib = 512;     ///< fallback floor: pressure while free physical memory is under this
    uint64_t keep_free_mib = 512;     ///< the free-RAM target the soft ceiling holds; pressure under it
    uint64_t min_commit_mib = 2048;   ///< ... or available commit (RAM + page file) is under this
    uint64_t emergency_mib = 256;     ///< below this (or once the OS signals low), release hard even in soft mode
    uint64_t recover_mib = 512;       ///< pressure clears this far above both floors (hysteresis)
    int retrim_ms = 3000;             ///< while pressure lasts, act again this often (0 = only on entry)
    int cooldown_ms = 2000;           ///< after recovery, ignore soft pressure this long (emergency still acts)
    MemGuardTrim trim = MemGuardTrim::Soft;  ///< Windows-only trim mode; see MemGuardTrim (ignored on Linux)
    bool priority = true;             ///< Windows-only: lower this process's memory priority on pressure
    bool notify = true;               ///< Windows-only: also use the OS low/high memory notification as a trigger
    bool predictive = true;           ///< start trimming early when free RAM is falling fast
    double predict_slope_mib_s = 128.0; ///< a free-RAM decline this fast (MiB/s) counts as "falling fast"
    uint64_t predict_band_mib = 2048;   ///< ... within this much of keep_free
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

}  // namespace strata::platform
