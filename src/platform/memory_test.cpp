// src/platform/memory_test.cpp - plan v0.3 P0.1: lock_resident on a 256 MiB region (CPU only, no GPU, no model).
#include "strata/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace {
uint64_t gib(double g) { return (uint64_t) (g * 1073741824.0); }

// #577: the file tier's unbuffered choice (true = the file cache keeps its reads, so they stay buffered)
int file_cache_keeps_cases() {
    using strata::platform::file_cache_keeps;
    struct Case { const char* what; double avail, arena, read; bool keeps; };
    const double experts = 71.7;   // UD-Q4_K_XL's routed experts
    const Case cases[] = {
        // the reporter's 96 GB PC (83.0 GiB available at start): 0.1.38 passed the budget asked for and every
        // shard's bytes (103.7 GiB) and went unbuffered
        {"96 GB, 0.1.38's inputs (budget 72, all shards)", 83.0, 72.0, 103.7, false},
        {"96 GB, budget 72, before the copy (arena = all experts)", 83.0, experts, 0.0, true},
        {"96 GB, budget 72, copy built (48.94 resident)", 83.0 - 48.94, 0.0, experts - 48.94, true},
        {"96 GB, budget 40, before the copy", 83.0, 40.0, experts - 40.0, true},
        {"96 GB, budget 40, copy built", 83.0 - 40.0, 0.0, experts - 40.0, true},
        // #357/#362's low-RAM PCs stay unbuffered
        {"64 GB, budget 40, before the copy", 55.0, 40.0, experts - 40.0, false},
        {"64 GB, budget 40, copy built", 15.0, 0.0, experts - 40.0, false},
        {"32 GB, budget 16, before the copy", 26.0, 16.0, experts - 16.0, false},
        {"32 GB, budget 16, copy built", 10.0, 0.0, experts - 16.0, false},
        {"32 GB, Q2_0 pack (34 GiB experts), budget 8", 26.0, 8.0, 34.0 - 8.0, false},
        {"nothing available", 0.0, 0.0, 1.0, false},
        {"arena larger than the RAM", 10.0, 20.0, 0.0, true},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const bool got = file_cache_keeps(gib(c.avail), gib(c.arena), gib(c.read));
        std::printf("file_cache_keeps %-58s -> %s%s\n", c.what, got ? "buffered" : "unbuffered",
                    got == c.keeps ? "" : "   <-- WRONG");
        fail |= got != c.keeps;
    }
    return fail;
}

// The guard's state machine is pure (`memory_guard_decide`, platform/memory.cpp), so its enter/exit and
// release sizing are checked here with no GPU, model or memory hog.  Defaults: keep 1024, recover 1024
// (exit floor 2048), min_avail 512, min_commit 2048, emergency 256, release_min 256; prediction is opt-in
// (band 512, slope 128 when `predictive` is set).
int guard_decision_cases() {
    using namespace strata::platform;
    MemoryGuardConfig c;
    struct Case {
        const char* what; bool low; uint64_t avail, commit; bool os_low; bool cooldown; double slope;
        GuardAction action; uint64_t release; bool emergency;
    };
    const Case cases[] = {
        {"quiet at 3 GiB free",                       false, 3000, 8000, false, false,   0.0, GuardAction::None,    0,          false},
        {"predict off: 1.5 GiB free, falling",        false, 1500, 8000, false, false, 500.0, GuardAction::None,    0,          false},
        {"below target at 900 MiB",                   false,  900, 8000, false, false,   0.0, GuardAction::Enter,   2048 - 900,  false},
        {"low at 1.5 GiB does not recover",           true, 1500, 8000, false, false, 500.0, GuardAction::Stay,  2048 - 1500, false},
        {"below target while low",                    true,   1100, 8000, false, false,   0.0, GuardAction::Stay,    2048 - 1100, false},
        {"recover above the exit floor",              true,   2500, 8000, false, false,   0.0, GuardAction::Recover, 0,          false},
        {"stay: OS still signals low at 2.5 GiB",     true,   2500, 8000, true,  false,   0.0, GuardAction::Stay,    256,        false},
        {"stay: commit still below its floor",        true,   2500, 1000, false, false,   0.0, GuardAction::Stay,    256,        false},
        {"cooldown suppresses a new enter",           false,   900, 8000, false, true,    0.0, GuardAction::None,    0,          false},
        {"emergency breaks the cooldown",             false,   200, 8000, true,  true,    0.0, GuardAction::Enter,   2048 - 200, true},
        {"emergency cliff",                           false,   200, 8000, true,  false,   0.0, GuardAction::Enter,   2048 - 200, true},
        {"emergency holds while low",                 true,    200, 8000, true,  false,   0.0, GuardAction::Stay,    2048 - 200, true},
    };
    int fail = 0;
    for (const Case& k : cases) {
        const GuardDecision d = memory_guard_decide(c, k.low, k.avail, k.commit, k.os_low, k.cooldown, k.slope);
        const char* an = d.action == GuardAction::None ? "none" : d.action == GuardAction::Enter ? "enter"
                       : d.action == GuardAction::Stay ? "stay" : "recover";
        const bool ok = d.action == k.action && d.release_mib == k.release && d.emergency == k.emergency;
        std::printf("guard %-46s -> %-7s release %4llu MiB%s%s\n", k.what, an,
                    (unsigned long long) d.release_mib, d.emergency ? " EMERGENCY" : "", ok ? "" : "  <-- WRONG");
        if (!ok) {
            std::printf("      wanted action=%d release=%llu emergency=%d\n", (int) k.action,
                        (unsigned long long) k.release, (int) k.emergency);
            fail = 1;
        }
    }
    // The predictive trigger is opt-in (`STRATA_MEM_GUARD_PREDICT=1`); these cases pin it and the narrow
    // 512 MiB band, so the default cannot drift back to a wide band unnoticed.
    c.predictive = true;
    struct PCase {
        const char* what; uint64_t avail; double slope; GuardAction action; uint64_t release;
    };
    const PCase pcases[] = {
        {"predict on: 1.5 GiB free, falling",  1500, 500.0, GuardAction::Enter, 2048 - 1500},
        {"predict on: 3 GiB free, falling",    3000, 500.0, GuardAction::None,  0},
    };
    for (const PCase& k : pcases) {
        const GuardDecision d = memory_guard_decide(c, false, k.avail, 8000, false, false, k.slope);
        const char* an = d.action == GuardAction::None ? "none" : d.action == GuardAction::Enter ? "enter"
                       : d.action == GuardAction::Stay ? "stay" : "recover";
        const bool ok = d.action == k.action && d.release_mib == k.release;
        std::printf("guard %-46s -> %-7s release %4llu MiB%s\n", k.what, an,
                    (unsigned long long) d.release_mib, ok ? "" : "  <-- WRONG");
        fail |= !ok;
    }
    return fail;
}
}  // namespace

int main() {
    int fail_keeps = file_cache_keeps_cases();
    const int fail_guard = guard_decision_cases();
    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const strata::platform::LockResult r = strata::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    int fail = !(r.ok && r.locked_bytes == bytes) | fail_keeps | fail_guard;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    std::printf("working set %llu MiB\n", (unsigned long long) (pmc.WorkingSetSize >> 20));
    fail |= pmc.WorkingSetSize < bytes;
#endif
    strata::platform::unlock_resident(p, bytes);
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
