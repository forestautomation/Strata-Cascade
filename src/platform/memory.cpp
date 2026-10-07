// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <dxgi1_4.h>
#else
#include <algorithm>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#endif

namespace strata::platform {

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) { why = "dxgi.dll not found"; return false; }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why = "no DXGI adapter has the CUDA device's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage = info.CurrentUsage;
                ok = budget > 0;
                why = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr) a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

uint64_t total_physical_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
}

bool read_ahead_enabled() { return false; }
void advise_willneed(const void*, uint64_t) {}
void advise_willneed(int, uint64_t, uint64_t) {}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_READ_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
}
#endif

// ================================ the memory-pressure guard ================================
//
// Both backends share the config, the stop flag, the thread and the `g_pressure_low` flag the
// sources read lock-free; only the platform calls and the loop differ.  Windows yields by trimming
// this process's working set and lowering its memory priority; Linux samples MemAvailable and pauses
// cold prefetch (the kernel already reclaims the clean expert pages itself).  See memory.hpp.

namespace {
std::atomic<bool> g_pressure_low{false};   ///< read lock-free from the prefetch path
std::atomic<uint64_t> g_stat_samples{0};
std::atomic<uint64_t> g_stat_hard{0};
std::atomic<uint64_t> g_stat_soft{0};
std::atomic<uint64_t> g_stat_cpu_ns{0};

// Shared by both backends: the config, the stop flag, the thread and the handle.
std::mutex g_guard_mu;
struct GuardState {
    MemoryGuardConfig cfg{};
    std::atomic<bool> stop{false};
    std::thread thread;
    ~GuardState() {
        stop.store(true, std::memory_order_relaxed);
        if (thread.joinable()) thread.join();
    }
};
std::unique_ptr<GuardState> g_guard;

#if defined(_WIN32)
using SetProcessInformationFn = BOOL (WINAPI*)(HANDLE, int, LPVOID, DWORD);

SetProcessInformationFn set_process_information_fn() {
    static const SetProcessInformationFn fn =
        (SetProcessInformationFn) (void*) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetProcessInformation");
    return fn;
}

// PROCESS_INFORMATION_CLASS::ProcessMemoryPriority and MEMORY_PRIORITY_* (winnt.h, Windows 8+).  Spelled out
// rather than included so the build does not depend on this TU's _WIN32_WINNT being new enough.
enum { kProcessMemoryPriority = 0 };
struct MemoryPriorityInfo { ULONG MemoryPriority; };
enum : ULONG { kMemoryPriorityVeryLow = 1, kMemoryPriorityNormal = 5 };

bool set_memory_priority(ULONG value) {
    const SetProcessInformationFn fn = set_process_information_fn();
    if (fn == nullptr) return false;
    MemoryPriorityInfo info{value};
    return fn(GetCurrentProcess(), kProcessMemoryPriority, &info, (DWORD) sizeof info) != 0;
}

// The guard's own CPU cost.  `GetThreadTimes` reports zeros on at least one Windows 11 build here, so
// measure in CPU cycles (`QueryThreadCycleTime`) and turn cycles into seconds using the process's own
// cycles-per-100ns ratio - the effective core frequency over the same interval.
uint64_t thread_cycles_now() {
    ULONG64 c = 0;
    return QueryThreadCycleTime(GetCurrentThread(), &c) != 0 ? (uint64_t) c : 0;
}

/// Cycles per second of this core, measured with a ~2 ms spin (only used when stats are enabled).  The
/// process and thread time APIs both report zeros on this Windows build, so calibrate instead.
double calibrate_cycles_per_s() {
    LARGE_INTEGER freq{};
    LARGE_INTEGER t0{}, t1{};
    if (!QueryPerformanceFrequency(&freq) || freq.QuadPart <= 0) return 0.0;
    const uint64_t c0 = thread_cycles_now();
    QueryPerformanceCounter(&t0);
    volatile uint64_t sink = 0;
    for (;;) {
        for (int i = 0; i < 4096; ++i) sink += (uint64_t) i;
        QueryPerformanceCounter(&t1);
        const double elapsed = (double) (t1.QuadPart - t0.QuadPart) / (double) freq.QuadPart;
        if (elapsed >= 0.002) break;
    }
    (void) sink;
    const uint64_t c1 = thread_cycles_now();
    const double elapsed = (double) (t1.QuadPart - t0.QuadPart) / (double) freq.QuadPart;
    return elapsed > 0.0 ? (double) (c1 - c0) / elapsed : 0.0;
}

// A soft working-set ceiling.  `SetProcessWorkingSetSizeEx` with the HARDWS_MAX_DISABLE flag makes the
// maximum a HINT the memory manager trims toward under pressure, not a hard quota that would fault us;
// combined with the lowered memory priority it makes this process the first victim.  Because it is a
// hint the OS keeps the recently used pages and only drops the cold ones - which is the whole point of
// the soft mode.  The original maximum is restored when pressure clears.
struct SoftCeiling {
    bool active = false;
    bool have_base = false;
    SIZE_T base_max = 0;    ///< the working-set maximum to restore (the process's own default)
    DWORD base_flags = 0;
    uint64_t base_ws = 0;   ///< this process's working set when the ceiling was first applied
    uint64_t applied = 0;   ///< the working-set maximum last handed to the OS, for logging
};

/// The most a single soft trim will ask the engine to release, so one app's launch cannot squeeze the
/// engine to nothing in a single step.  Later polls release another slice only while free RAM is still
/// under the target.
const uint64_t kMaxReleasePerTrim = 4ull << 30;

/// This process's working set in bytes (0 when unavailable).
uint64_t working_set_bytes() {
    PROCESS_MEMORY_COUNTERS pmc{};
    return K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc) ? (uint64_t) pmc.WorkingSetSize : 0;
}

/// Read the process's working-set limits.  Call once before any ceiling is applied (the guard thread's
/// start), so the default maximum is remembered for restore.
void capture_ws_base(SoftCeiling& sc) {
    if (sc.have_base) return;
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(GetCurrentProcess(), &min_ws, &max_ws, &flags)) return;
    sc.base_max = max_ws;
    sc.base_flags = flags;
    sc.base_ws = working_set_bytes();   // the working set the ceiling's release is measured against
    sc.have_base = true;
}

bool apply_soft_ceiling(SoftCeiling& sc, uint64_t target_ws) {
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) return false;
    if (!sc.have_base) capture_ws_base(sc);
    // The max is only a hint, so there is no need to bound `want` from above: setting it above the current
    // working set simply means "no ceiling". The locked/pinned pages must never be squeezed, hence the floor.
    const SIZE_T floor = min_ws + (64ull << 20);
    SIZE_T want = (SIZE_T) target_ws;
    if (want < floor) want = floor;
    // A failed call must not be reported (or counted) as a release.  Only mark the ceiling active and
    // remember `applied` once the OS accepted it, so the log and `soft_trims` reflect what really happened.
    if (!SetProcessWorkingSetSizeEx(self, min_ws, want,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE))
        return false;
    sc.active = true;
    sc.applied = (uint64_t) want;
    return true;
}

void clear_soft_ceiling(SoftCeiling& sc) {
    if (!sc.active) return;
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        // The base maximum was captured at guard start, before `lock_resident` raised the minimum for the
        // resident/pinned arena.  Restoring a max below the current min would be rejected (or worse), so
        // fall back to "no maximum" (`(SIZE_T) -1`) whenever the remembered base is below the min.
        SIZE_T restore = sc.have_base ? sc.base_max : max_ws;
        if (restore < min_ws) restore = (SIZE_T) -1;
        SetProcessWorkingSetSizeEx(self, min_ws, restore, sc.base_flags);
    }
    sc.active = false;
    sc.applied = 0;
}

void guard_loop(GuardState* s) {
    const MemoryGuardConfig& cfg = s->cfg;
    const bool retrim = cfg.retrim_ms > 0;

    // The OS's own low/high memory notifications: the low handle is signalled while the memory manager
    // considers RAM low, the high handle while it is plentiful.  Between them is the hysteresis band.
    HANDLE low_evt = nullptr, high_evt = nullptr;
    if (cfg.notify) {
        low_evt = CreateMemoryResourceNotification(LowMemoryResourceNotification);
        high_evt = CreateMemoryResourceNotification(HighMemoryResourceNotification);
    }

    auto now_ms = [] {
        return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    // Sleep in ~50 ms slices so stop stays responsive; when the notification handles are usable and
    // neither is currently signalled, block on them instead - that is the event-driven path.
    auto idle = [&](int ms) {
        if (ms <= 0) return;
        HANDLE h[2];
        DWORD n = 0;
        if (low_evt != nullptr) h[n++] = low_evt;
        if (high_evt != nullptr) h[n++] = high_evt;
        bool signalled = false;
        for (DWORD i = 0; i < n && !signalled; ++i)
            signalled = WaitForSingleObject(h[i], 0) == WAIT_OBJECT_0;
        if (n > 0 && !signalled) {
            WaitForMultipleObjects(n, h, FALSE, (DWORD) ms);
            return;
        }
        for (int slept = 0; slept < ms && !s->stop.load(std::memory_order_relaxed); slept += 50)
            std::this_thread::sleep_for(std::chrono::milliseconds(std::min(50, ms - slept)));
    };

    const double cycles_per_s = cfg.stats ? calibrate_cycles_per_s() : 0.0;
    const uint64_t tc_start = thread_cycles_now();
    const long long t_start = now_ms();
    // Seconds of CPU the monitor thread has used since start, from its cycles.
    auto monitor_cpu_s = [&]() -> double {
        if (cycles_per_s <= 0.0) return 0.0;
        return (double) (thread_cycles_now() - tc_start) / cycles_per_s;
    };

    bool low = false;
    long long last_trim = t_start;   ///< last hard EmptyWorkingSet
    long long cooldown_until = 0;
    long long last_stats = t_start;
    uint64_t prev_avail = 0;
    long long prev_t = t_start;
    bool have_prev = false;
    double slope = 0.0;              ///< EWMA of free-RAM decline, MiB/s (positive = falling)
    SoftCeiling sc;
    capture_ws_base(sc);   // read the working-set maximum before any ceiling is applied
    uint64_t samples = 0, hard_trims = 0, soft_trims = 0;

    auto publish = [&] {
        g_stat_samples.store(samples, std::memory_order_relaxed);
        g_stat_hard.store(hard_trims, std::memory_order_relaxed);
        g_stat_soft.store(soft_trims, std::memory_order_relaxed);
    };
    auto hard_trim = [&](const char* tag) {
        // Drop any soft ceiling first: EmptyWorkingSet already released everything, and leaving a stale
        // ceiling would keep the OS squeezing the already-trimmed working set.
        clear_soft_ceiling(sc);
        const uint64_t before = working_set_bytes();
        const bool trimmed = K32EmptyWorkingSet(GetCurrentProcess()) != 0;
        if (trimmed) ++hard_trims;
        last_trim = now_ms();
        std::fprintf(stderr, "strata memory-guard: %s; trimmed working set %llu -> %llu MiB\n", tag,
                     (unsigned long long) (before >> 20), (unsigned long long) (working_set_bytes() >> 20));
        std::fflush(stderr);
    };
    auto soft_trim = [&](uint64_t want_free) -> uint64_t {
        if (want_free == 0) return sc.applied;   // free RAM is above target again: keep the ceiling, do not clear
        // Size the ceiling from the process's working set when the pressure began, not from the readback
        // after a previous trim: the readback is already small, so using it would ratchet the ceiling down
        // to the floor poll after poll.  The engine's fair share of a deficit it cannot cover on its own is
        // capped so one app's launch cannot squeeze it to nothing.
        const uint64_t base = sc.base_ws > 0 ? sc.base_ws : working_set_bytes();
        const uint64_t share = want_free < kMaxReleasePerTrim ? want_free : kMaxReleasePerTrim;
        const uint64_t target = base > share ? base - share : 0;
        if (apply_soft_ceiling(sc, target)) ++soft_trims;   // count only a ceiling the OS accepted
        return sc.applied;
    };

    while (!s->stop.load(std::memory_order_relaxed)) {
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        PROCESS_MEMORY_COUNTERS pmc{};
        const bool have_ms = GlobalMemoryStatusEx(&ms) != 0;
        const bool have_pmc = K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc) != 0;
        const uint64_t avail = have_ms ? (uint64_t) ms.ullAvailPhys : 0;
        const uint64_t commit = have_ms ? (uint64_t) ms.ullAvailPageFile : 0;
        const uint64_t ws = have_pmc ? (uint64_t) pmc.WorkingSetSize : 0;
        const long long t = now_ms();
        samples++;

        bool notify_low = false;
        if (low_evt != nullptr) notify_low = WaitForSingleObject(low_evt, 0) == WAIT_OBJECT_0;

        // Predictive: a smoothed decline in free RAM (MiB/s) near the target starts releasing early,
        // before the OS is forced to page.  Pure arithmetic on samples we already take.
        if (have_ms && have_prev) {
            const double dt = (double) (t - prev_t) / 1000.0;
            if (dt > 0.02) {
                const double inst = ((double) prev_avail - (double) avail) / 1048576.0 / dt;
                slope = 0.35 * inst + 0.65 * slope;
            }
        }
        if (have_ms) { prev_avail = avail; prev_t = t; have_prev = true; }

        const bool in_cooldown = t < cooldown_until;
        const GuardDecision dec = memory_guard_decide(cfg, low,
                                                      have_ms ? (avail >> 20) : 0,
                                                      have_ms ? (commit >> 20) : 0,
                                                      /*have_commit=*/have_ms,
                                                      notify_low, in_cooldown, slope);

        // Release the deficit with the soft working-set ceiling: `target = entry_ws - deficit`, so the
        // request is proportional to what is actually missing (the old code asked for a flat quarter of
        // the target).  Priority stays VERY_LOW for the whole low state, so the OS keeps choosing the
        // engine's clean pages over another app's dirty ones for as long as the pressure lasts.
        if (dec.action == GuardAction::Enter) {
            low = true;
            g_pressure_low.store(true, std::memory_order_relaxed);
            sc.base_ws = ws;   // the fallback ceiling is measured against the entry working set
            const bool prio = cfg.priority && set_memory_priority(kMemoryPriorityVeryLow);
            // A cliff goes hard only when asked (`emergency_trim == Hard`); the default sends a larger
            // proportional soft ceiling instead.  EmptyWorkingSet is expensive and throws the whole working
            // set away - it re-faults everything - while the ceiling plus VERY_LOW priority already makes
            // the OS reclaim this process's cheap pages first.
            const bool cliff_hard = dec.emergency && cfg.emergency_trim == MemGuardTrim::Hard;
            if (cfg.trim == MemGuardTrim::Off) {
                std::fprintf(stderr,
                             "strata memory-guard: LOW (RAM %llu MiB, commit %llu MiB); memory priority %s, "
                             "trim off%s\n",
                             (unsigned long long) (avail >> 20), (unsigned long long) (commit >> 20),
                             prio ? "VERY_LOW" : "unchanged", notify_low ? ", OS signalled low" : "");
            } else if (cfg.trim == MemGuardTrim::Hard || cliff_hard) {
                hard_trim(dec.emergency ? "LOW (emergency)" : "LOW (hard)");
            } else {
                const uint64_t before = working_set_bytes();
                const uint64_t ceiling = soft_trim(dec.release_mib << 20);
                std::fprintf(stderr,
                             "strata memory-guard: LOW (RAM %llu MiB, commit %llu MiB, ws %llu MiB); "
                             "soft ceiling -> %llu MiB, memory priority %s%s%s\n",
                             (unsigned long long) (avail >> 20), (unsigned long long) (commit >> 20),
                             (unsigned long long) (before >> 20), (unsigned long long) (ceiling >> 20),
                             prio ? "VERY_LOW" : "unchanged",
                             dec.emergency ? ", cliff" : "",
                             notify_low ? ", OS signalled low" : "");
            }
            std::fflush(stderr);
        } else if (dec.action == GuardAction::Recover) {
            low = false;
            g_pressure_low.store(false, std::memory_order_relaxed);
            clear_soft_ceiling(sc);
            const bool prio = cfg.priority && set_memory_priority(kMemoryPriorityNormal);
            cooldown_until = t + cfg.cooldown_ms;
            std::fprintf(stderr,
                         "strata memory-guard: recovered (RAM %llu MiB, commit %llu MiB); resuming%s\n",
                         (unsigned long long) (avail >> 20), (unsigned long long) (commit >> 20),
                         prio ? ", memory priority NORMAL" : "");
            std::fflush(stderr);
        } else if (dec.action == GuardAction::Stay) {
            // A cliff goes hard only when `emergency_trim == Hard` (and `trim` is not Off); the default
            // soft ceiling then re-trims on the same `retrim_ms` schedule as ordinary pressure.
            const bool cliff_hard = dec.emergency && cfg.emergency_trim == MemGuardTrim::Hard;
            if (cfg.trim != MemGuardTrim::Off && cliff_hard) {
                if (!retrim || t - last_trim >= cfg.retrim_ms) hard_trim("still low (emergency)");
            } else if (cfg.trim == MemGuardTrim::Hard) {
                if (retrim && t - last_trim >= cfg.retrim_ms) hard_trim("still low");
            } else if (cfg.trim == MemGuardTrim::Soft && dec.release_mib > 0) {
                // Respect `retrim_ms` here too: re-applying the same ceiling every poll was pure overhead
                // - it is the measured driver of the monitor thread's own CPU - and `retrim_ms` of 0 means
                // "act on entry only", the same as the hard paths.
                if (retrim && t - last_trim >= cfg.retrim_ms) {
                    soft_trim(dec.release_mib << 20);
                    last_trim = t;
                }
            }
        } else if (cfg.verbose) {
            std::fprintf(stderr,
                         "strata memory-guard: RAM %llu MiB, commit %llu MiB, ws %llu MiB, slope %.0f MiB/s, "
                         "pressure idle\n",
                         (unsigned long long) (avail >> 20), (unsigned long long) (commit >> 20),
                         (unsigned long long) (ws >> 20), slope);
            std::fflush(stderr);
        }

        if (cfg.stats && t - last_stats >= 5000) {
            const double wall_s = (t - t_start) / 1000.0;
            const double cpu_s = monitor_cpu_s();
            std::fprintf(stderr,
                         "strata memory-guard: stats samples=%llu hard=%llu soft=%llu monitor_cpu=%.3fs over %.1fs "
                         "(%.3f%% of one core)\n",
                         (unsigned long long) samples, (unsigned long long) hard_trims,
                         (unsigned long long) soft_trims, cpu_s, wall_s,
                         wall_s > 0.0 ? 100.0 * cpu_s / wall_s : 0.0);
            std::fflush(stderr);
            last_stats = t;
            publish();
        }

        idle(cfg.poll_ms);
    }

    clear_soft_ceiling(sc);
    g_pressure_low.store(false, std::memory_order_relaxed);   // never leave sources suppressing prefetch
    if (cfg.priority) (void) set_memory_priority(kMemoryPriorityNormal);
    if (low_evt != nullptr) CloseHandle(low_evt);
    if (high_evt != nullptr) CloseHandle(high_evt);
    publish();
    const uint64_t cpu_end_ns = (uint64_t) (monitor_cpu_s() * 1e9);
    g_stat_cpu_ns.store(cpu_end_ns, std::memory_order_relaxed);
    if (cfg.stats)
        std::fprintf(stderr, "strata memory-guard: monitor thread exiting, cpu %.6f s over %.1f s\n",
                     (double) cpu_end_ns / 1e9, (now_ms() - t_start) / 1000.0);
}

#else  // !_WIN32 - Linux/POSIX: sample /proc/meminfo, pause cold prefetch under pressure.
//
// Linux has no per-process memory priority and no working-set ceiling, so this backend does the one
// thing the kernel does not do by itself: when free RAM (MemAvailable) falls under the target it
// raises `g_pressure_low`, and the tiered/mmap sources stop prefetching cold experts (see memory.hpp).
// The kernel already reclaims this process's clean, file-backed expert pages without a page-file
// write, so there is no explicit trim - only the prefetch pause and, on recovery, resuming it.

/// The /proc/meminfo fields the guard and `memory_sample` need, from ONE read of the file (it used to be
/// opened once per field, so three times a poll).  Values are KiB; 0 when the field is absent.
/// MemAvailable is the kernel's own estimate of what can be allocated without swapping and it counts
/// reclaimable page cache - the right free-RAM signal; a kernel older than 3.14 has none (`have_avail`).
struct Meminfo {
    uint64_t avail_kib = 0;         // MemAvailable
    uint64_t commit_limit_kib = 0;  // CommitLimit
    uint64_t committed_kib = 0;     // Committed_AS
    bool have_avail = false;
};

Meminfo read_meminfo() {
    Meminfo m;
    std::ifstream f("/proc/meminfo");
    if (!f) return m;
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 13, "MemAvailable:") == 0) {
            m.avail_kib = std::strtoull(line.c_str() + 13, nullptr, 10);
            m.have_avail = true;
        } else if (line.compare(0, 12, "CommitLimit:") == 0) {
            m.commit_limit_kib = std::strtoull(line.c_str() + 12, nullptr, 10);
        } else if (line.compare(0, 13, "Committed_AS:") == 0) {
            m.committed_kib = std::strtoull(line.c_str() + 13, nullptr, 10);
        }
    }
    return m;
}

/// The /proc/self/status fields used here (VmRSS, VmSwap), one read; KiB, 0 when absent.
struct SelfStatus {
    uint64_t vm_rss_kib = 0;   // resident set
    uint64_t vm_swap_kib = 0;  // this process's swapped-out bytes
};

SelfStatus read_self_status() {
    SelfStatus s;
    std::ifstream f("/proc/self/status");
    if (!f) return s;
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) s.vm_rss_kib = std::strtoull(line.c_str() + 6, nullptr, 10);
        else if (line.compare(0, 7, "VmSwap:") == 0) s.vm_swap_kib = std::strtoull(line.c_str() + 7, nullptr, 10);
    }
    return s;
}

void guard_loop(GuardState* s) {
    const MemoryGuardConfig& cfg = s->cfg;

    auto now_ms = [] {
        return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    // The monitor thread's own CPU, straight from the thread clock; cheap and exact, no calibration.
    struct timespec tc0{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tc0);
    auto monitor_cpu_s = [&]() -> double {
        struct timespec tc{};
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tc) != 0) return 0.0;
        return (double) (tc.tv_sec - tc0.tv_sec) + (double) (tc.tv_nsec - tc0.tv_nsec) / 1e9;
    };

    bool low = false;
    long long cooldown_until = 0;
    const long long t_start = now_ms();
    long long last_stats = t_start;
    uint64_t prev_avail = 0;
    long long prev_t = t_start;
    bool have_prev = false;
    double slope = 0.0;              ///< EWMA of free-RAM decline, MiB/s (positive = falling)
    uint64_t samples = 0, hard_trims = 0, soft_trims = 0;

    auto publish = [&] {
        g_stat_samples.store(samples, std::memory_order_relaxed);
        g_stat_hard.store(hard_trims, std::memory_order_relaxed);
        g_stat_soft.store(soft_trims, std::memory_order_relaxed);
    };

    while (!s->stop.load(std::memory_order_relaxed)) {
        const Meminfo mi = read_meminfo();
        const SelfStatus ss = read_self_status();
        const uint64_t avail = mi.avail_kib << 10;
        const uint64_t commit_limit = mi.commit_limit_kib << 10;
        const uint64_t committed = mi.committed_kib << 10;
        const bool have_commit = commit_limit != 0;   // CommitLimit can legitimately be 0 = exhausted
        const uint64_t commit = commit_limit > committed ? commit_limit - committed : 0;
        const uint64_t ws = ss.vm_rss_kib << 10;
        const long long t = now_ms();
        samples++;

        // Predictive: a smoothed decline in free RAM (MiB/s) near the target starts the pause early,
        // before the kernel is forced to reclaim.  Pure arithmetic on samples we already take.
        if (avail != 0) {
            if (have_prev) {
                const double dt = (double) (t - prev_t) / 1000.0;
                if (dt > 0.02) {
                    const double inst = ((double) prev_avail - (double) avail) / 1048576.0 / dt;
                    slope = 0.35 * inst + 0.65 * slope;
                }
            }
            prev_avail = avail;
            prev_t = t;
            have_prev = true;
        }

        const bool in_cooldown = t < cooldown_until;
        const GuardDecision dec = memory_guard_decide(cfg, low,
                                                      avail != 0 ? (avail >> 20) : 0,
                                                      commit >> 20,
                                                      have_commit,
                                                      /*os_low=*/false, in_cooldown, slope);

        if (dec.action == GuardAction::Enter) {
            low = true;
            g_pressure_low.store(true, std::memory_order_relaxed);
            // The kernel reclaims the engine's clean, file-backed expert pages by itself; the guard's job
            // here is the prefetch pause (`g_pressure_low`) plus the same hysteresis as Windows.
            std::fprintf(stderr,
                         "strata memory-guard: LOW (RAM %llu MiB, commit %llu MiB, ws %llu MiB); "
                         "pausing cold prefetch%s\n",
                         (unsigned long long) (avail >> 20), (unsigned long long) (commit >> 20),
                         (unsigned long long) (ws >> 20), dec.emergency ? " (emergency)" : "");
            std::fflush(stderr);
        } else if (dec.action == GuardAction::Recover) {
            low = false;
            g_pressure_low.store(false, std::memory_order_relaxed);
            cooldown_until = t + cfg.cooldown_ms;
            std::fprintf(stderr,
                         "strata memory-guard: recovered (RAM %llu MiB, commit %llu MiB); resuming "
                         "cold prefetch\n",
                         (unsigned long long) (avail >> 20), (unsigned long long) (commit >> 20));
            std::fflush(stderr);
        } else if (cfg.verbose) {
            std::fprintf(stderr,
                         "strata memory-guard: RAM %llu MiB, commit %llu MiB, ws %llu MiB, slope %.0f MiB/s, "
                         "pressure idle\n",
                         (unsigned long long) (avail >> 20), (unsigned long long) (commit >> 20),
                         (unsigned long long) (ws >> 20), slope);
            std::fflush(stderr);
        }

        if (cfg.stats && t - last_stats >= 5000) {
            const double wall_s = (t - t_start) / 1000.0;
            const double cpu_s = monitor_cpu_s();
            std::fprintf(stderr,
                         "strata memory-guard: stats samples=%llu hard=%llu soft=%llu monitor_cpu=%.3fs over "
                         "%.1fs (%.3f%% of one core)\n",
                         (unsigned long long) samples, (unsigned long long) hard_trims,
                         (unsigned long long) soft_trims, cpu_s, wall_s,
                         wall_s > 0.0 ? 100.0 * cpu_s / wall_s : 0.0);
            std::fflush(stderr);
            last_stats = t;
        }
        publish();

        for (int slept = 0; slept < cfg.poll_ms && !s->stop.load(std::memory_order_relaxed); slept += 50)
            std::this_thread::sleep_for(std::chrono::milliseconds(std::min(50, cfg.poll_ms - slept)));
    }

    g_pressure_low.store(false, std::memory_order_relaxed);
    publish();
    g_stat_cpu_ns.store((uint64_t) (monitor_cpu_s() * 1e9), std::memory_order_relaxed);
    if (cfg.stats)
        std::fprintf(stderr, "strata memory-guard: monitor thread exiting, cpu %.6f s over %.1f s\n",
                     monitor_cpu_s(), (now_ms() - t_start) / 1000.0);
}
#endif  // _WIN32
}  // namespace

bool memory_sample(MemorySample& out) {
    out = MemorySample{};
#if defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
        out.avail_phys = ms.ullAvailPhys;
        out.total_phys = ms.ullTotalPhys;
        out.avail_pagefile = ms.ullAvailPageFile;
    }
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        out.pagefile_usage = pmc.PagefileUsage;
        out.working_set = pmc.WorkingSetSize;
    }
    return out.total_phys != 0;
#else
    out.total_phys = total_physical_memory();
    const Meminfo mi = read_meminfo();
    const SelfStatus ss = read_self_status();
    out.avail_phys = mi.avail_kib << 10;
    const uint64_t cl = mi.commit_limit_kib << 10;
    const uint64_t ca = mi.committed_kib << 10;
    out.avail_pagefile = cl > ca ? cl - ca : 0;          // available commit (RAM + swap)
    out.working_set = ss.vm_rss_kib << 10;
    out.pagefile_usage = ss.vm_swap_kib << 10;
    return out.total_phys != 0 || out.avail_phys != 0;
#endif
}

GuardDecision memory_guard_decide(const MemoryGuardConfig& cfg, bool low, uint64_t avail_mib,
                                  uint64_t commit_mib, bool have_commit, bool os_low, bool in_cooldown,
                                  double slope_mib_s) {
    GuardDecision d;
    const bool have_avail = avail_mib != 0;
    // `have_commit` separates "the sample is unavailable" from "commit is exhausted": the old
    // `commit_mib != 0` test excluded the worst case (a genuinely zero commit) from pressure.
    const bool commit_low = have_commit && commit_mib < cfg.min_commit_mib;
    // A fast decline starts the release a little early; the band is a lead, not a floor, so it stays
    // close to the target (the old 2 GiB band triggered at ~2.5 GiB free and then recovered at 1 GiB -
    // an inverted pair that pulsed the guard on every transient dip).
    const bool predictive = cfg.predictive && have_avail && slope_mib_s >= cfg.predict_slope_mib_s &&
                            avail_mib <= cfg.keep_free_mib + cfg.predict_band_mib;
    const bool pressure = have_avail && (avail_mib < cfg.min_avail_mib || avail_mib < cfg.keep_free_mib) ||
                          commit_low || os_low || predictive;
    // A genuine cliff is free RAM under the emergency floor, whether or not the OS notification fired:
    // the notification can be disabled (`notify=0`) or unavailable (CreateMemoryResourceNotification
    // failed), and there is none off-Windows.  The OS's own low signal only widens the floor to twice it.
    const bool emergency = have_avail && (avail_mib < cfg.emergency_mib ||
                                          (os_low && avail_mib < 2 * cfg.emergency_mib));

    // Release up to the exit floor (`keep_free + recover`) so acting lifts free RAM clear of the entry
    // band and the state does not flap; never less than `release_min_mib` once acting, because a
    // sub-poll trim is wasted work.  This is the whole release: the source drops exactly these pages.
    const uint64_t exit_floor = cfg.keep_free_mib + cfg.recover_mib;
    auto deficit_mib = [&]() -> uint64_t {
        const uint64_t gap = exit_floor > avail_mib ? exit_floor - avail_mib : 0;
        return gap > cfg.release_min_mib ? gap : cfg.release_min_mib;
    };

    if (low) {
        // Do not recover while the OS still says memory is low: its low-memory notification can stay
        // signalled for the whole of a sustained load while free RAM is already above the exit floor, and
        // recovering then re-entering every poll flapped the priority.  Hold the yield until the OS clears.
        const bool recovered = have_avail && !os_low && avail_mib > exit_floor &&
                               avail_mib > cfg.min_avail_mib + cfg.recover_mib &&
                               (!have_commit || commit_mib > cfg.min_commit_mib + cfg.recover_mib);
        if (recovered && !emergency) { d.action = GuardAction::Recover; return d; }
        d.action = GuardAction::Stay;
        d.release_mib = have_avail ? deficit_mib() : 0;
        d.emergency = emergency;
        return d;
    }
    // A genuine cliff ignores the cooldown: the OS signal plus free RAM under the emergency floor must
    // release now, not after the settle window (the old loop exempted emergency the same way).
    if ((pressure && !in_cooldown) || emergency) {
        d.action = GuardAction::Enter;
        d.release_mib = have_avail ? deficit_mib() : 0;
        d.emergency = emergency;
    }
    return d;
}

bool memory_guard_start(const MemoryGuardConfig& cfg, std::string& why) {
    std::lock_guard<std::mutex> lk(g_guard_mu);
    if (g_guard) return true;
    if (cfg.poll_ms < 50) { why = "the guard poll interval is under 50 ms"; return false; }
    // The knobs come from the environment (`std::strtoull`, so "-1" becomes UINT64_MAX) and used to be
    // taken unclamped.  An absurd `keep_free + recover` overflows the exit floor, which either makes
    // recovery unreachable or asks the OS for nonsense; and `recover < predict_band` makes enter/exit an
    // inverted pair that pulses.  Validate here, where the failure can be reported, and sanitize the rest.
    MemoryGuardConfig c = cfg;
    if (c.retrim_ms < 0) c.retrim_ms = 0;
    if (c.cooldown_ms < 0) c.cooldown_ms = 0;
    const uint64_t kMibCap = 1ull << 20;   // 1 TiB: far above any real target, and keeps every sum safe
    if (c.keep_free_mib == 0 || c.keep_free_mib > kMibCap ||
        c.recover_mib > kMibCap - c.keep_free_mib) {
        why = "STRATA_MEM_GUARD_KEEP_FREE_MIB must be in [1, 1048576], and keep_free + recover under 1048576";
        return false;
    }
    if (c.min_avail_mib > kMibCap || c.min_commit_mib > kMibCap || c.emergency_mib > kMibCap ||
        c.release_min_mib > kMibCap || c.predict_band_mib > kMibCap) {
        why = "a memory-guard MiB target is out of range (maximum 1048576)";
        return false;
    }
    if (c.predictive && c.predict_band_mib > c.recover_mib) {
        why = "STRATA_MEM_GUARD_RECOVER_MIB must be at least STRATA_MEM_GUARD_PREDICT_BAND_MIB when predictive";
        return false;
    }
    auto s = std::make_unique<GuardState>();
    s->cfg = c;
    g_pressure_low.store(false, std::memory_order_relaxed);
    g_stat_samples.store(0, std::memory_order_relaxed);
    g_stat_hard.store(0, std::memory_order_relaxed);
    g_stat_soft.store(0, std::memory_order_relaxed);
    g_stat_cpu_ns.store(0, std::memory_order_relaxed);
    static const bool registered = (std::atexit(memory_guard_stop), true);
    (void) registered;
    s->thread = std::thread(guard_loop, s.get());
    g_guard = std::move(s);
    why.clear();
    return true;
}

void memory_guard_stop() {
    std::unique_ptr<GuardState> s;
    {
        std::lock_guard<std::mutex> lk(g_guard_mu);
        s = std::move(g_guard);
    }
    if (s) {
        s->stop.store(true, std::memory_order_relaxed);
        if (s->thread.joinable()) s->thread.join();
    }
    g_pressure_low.store(false, std::memory_order_relaxed);
}

bool memory_pressure_low() {
    return g_pressure_low.load(std::memory_order_relaxed);
}

MemoryGuardStats memory_guard_stats() {
    MemoryGuardStats st;
    st.samples = g_stat_samples.load(std::memory_order_relaxed);
    st.hard_trims = g_stat_hard.load(std::memory_order_relaxed);
    st.soft_trims = g_stat_soft.load(std::memory_order_relaxed);
    st.monitor_cpu_ns = g_stat_cpu_ns.load(std::memory_order_relaxed);
    return st;
}

}  // namespace strata::platform
