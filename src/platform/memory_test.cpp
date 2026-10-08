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
    // #1194, the Linux file tier: a skewed read set, so a twentieth of it (at least 1.5 GiB) is room enough
    const Case linux_cases[] = {
        {"#1194: 31 GB + 12 GB GPU, budget 18 built (6.4 avail, 13.6 read)", 6.4, 0.0, 13.6, true},
        {"#1194: before the copy (avail 25, arena 18, read 13.6)", 25.0, 18.0, 13.6, true},
        {"32 GB cgroup, budget 18, copy built (12.1 avail, 22 read)", 12.1, 0.0, 22.0, true},
        {"32 GB cgroup, budget 24, copy built (3.8 avail, 16 read)", 3.8, 0.0, 16.0, true},
        {"20 GB cgroup, budget 12, copy built (4.7 avail, 28 read)", 4.7, 0.0, 28.0, true},
        {"12 GB cgroup, budget 6, copy built (3.5 avail, 34 read)", 3.5, 0.0, 34.0, true},
        {"no room beside the margin (1 avail, 47 read)", 1.0, 0.0, 47.0, false},
        {"room under the 1.5 GiB floor (2.4 avail, 47 read)", 2.4, 0.0, 47.0, false},
        {"room a twentieth of the reads (3.4 avail, 47 read)", 3.4, 0.0, 47.0, true},
        {"room just under that (3.3 avail, 47 read)", 3.3, 0.0, 47.0, false},
        {"a big read set needs its twentieth (4 avail, 200 read)", 4.0, 0.0, 200.0, false},
        {"a big read set with that room (12 avail, 200 read)", 12.0, 0.0, 200.0, true},
        {"read less than the floor (2.5 avail, 1 read)", 2.5, 0.0, 1.0, true},
        {"nothing available", 0.0, 0.0, 1.0, false},
        {"arena larger than the RAM", 10.0, 20.0, 5.0, false},
    };
    for (const Case& c : linux_cases) {
        const bool got = file_cache_keeps(gib(c.avail), gib(c.arena), gib(c.read), 1ull << 30, 0.05, 3ull << 29);
        std::printf("file_cache_keeps (Linux file tier) %-52s -> %s%s\n", c.what, got ? "buffered" : "unbuffered",
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
        bool have_commit = true;   // trailing default: the pre-B2 cases all carry a real commit sample
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
        // `have_commit` separates an unavailable sample from a genuinely exhausted commit (0 MiB),
        // which is the worst case and must count as pressure / block recovery.
        {"commit exhausted (0) acts",                 false, 3000,    0, false, false,   0.0, GuardAction::Enter,   256,        false, true},
        {"commit unknown is not pressure",            false, 3000,    0, false, false,   0.0, GuardAction::None,    0,          false, false},
        {"commit exhausted blocks recovery",          true,  3000,    0, false, false,   0.0, GuardAction::Stay,    256,        false, true},
        {"commit unknown allows recovery",            true,  3000,    0, false, false,   0.0, GuardAction::Recover, 0,          false, false},
        // The emergency floor itself is the driver, not the OS notification - so it still fires with
        // `notify=0` and on Linux (`os_low=false`); the OS signal only widens it to 2x the floor.
        {"emergency cliff without OS signal",         false,  200, 8000, false, false,   0.0, GuardAction::Enter,   2048 - 200, true},
        {"emergency holds without OS signal",         true,   200, 8000, false, false,   0.0, GuardAction::Stay,    2048 - 200, true},
        {"OS low widens emergency to 2x floor",       false,  400, 8000, true,  false,   0.0, GuardAction::Enter,   2048 - 400, true},
        {"OS low above 2x floor is not emergency",    false,  600, 8000, true,  false,   0.0, GuardAction::Enter,   2048 - 600, false},
    };
    int fail = 0;
    for (const Case& k : cases) {
        const GuardDecision d = memory_guard_decide(c, k.low, k.avail, k.commit, k.have_commit, k.os_low,
                                                    k.cooldown, k.slope);
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
        const GuardDecision d = memory_guard_decide(c, false, k.avail, 8000, true, false, false, k.slope);
        const char* an = d.action == GuardAction::None ? "none" : d.action == GuardAction::Enter ? "enter"
                       : d.action == GuardAction::Stay ? "stay" : "recover";
        const bool ok = d.action == k.action && d.release_mib == k.release;
        std::printf("guard %-46s -> %-7s release %4llu MiB%s\n", k.what, an,
                    (unsigned long long) d.release_mib, ok ? "" : "  <-- WRONG");
        fail |= !ok;
    }
    return fail;
}

// `memory_guard_start` validates/clamps the env-supplied knobs.  The rejection paths return before a
// thread is created, so they need no GPU; the last case starts and stops a real monitor thread, which is
// the graceful-stop path the guard harness never exercised (it force-kills the engine).
int guard_validation_cases() {
    using namespace strata::platform;
    int fail = 0;
    auto rejects = [&](const char* what, const MemoryGuardConfig& cfg) {
        std::string why;
        const bool ok = !memory_guard_start(cfg, why);
        std::printf("guard cfg %-40s -> %s (%s)%s\n", what, ok ? "rejected" : "ACCEPTED", why.c_str(),
                    ok ? "" : "   <-- WRONG");
        fail |= !ok;
    };
    MemoryGuardConfig c;
    c.keep_free_mib = 0;
    rejects("keep_free 0", c);
    c = MemoryGuardConfig{};
    c.keep_free_mib = ~0ull;                  // what `strtoull("-1")` produces
    rejects("keep_free -1 (UINT64_MAX)", c);
    c = MemoryGuardConfig{};
    c.recover_mib = ~0ull;
    rejects("recover -1 (UINT64_MAX)", c);
    c = MemoryGuardConfig{};
    c.poll_ms = 10;
    rejects("poll 10 ms", c);
    c = MemoryGuardConfig{};
    c.predictive = true;
    c.predict_band_mib = 4096;                // band above recover = an inverted enter/exit pair
    rejects("predictive band > recover", c);

    c = MemoryGuardConfig{};
    std::string why;
    const bool started = memory_guard_start(c, why);
    std::printf("guard cfg %-40s -> %s\n", "the defaults", started ? "accepted" : "REFUSED");
    fail |= !started;
    if (started) memory_guard_stop();
    const MemoryGuardStats st = memory_guard_stats();
    std::printf("guard start/stop: samples=%llu hard=%llu soft=%llu\n", (unsigned long long) st.samples,
                (unsigned long long) st.hard_trims, (unsigned long long) st.soft_trims);
    return fail;
}
}  // namespace

int main() {
    int fail_keeps = file_cache_keeps_cases();
    const int fail_guard = guard_decision_cases();
    const int fail_guard_cfg = guard_validation_cases();
    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const strata::platform::LockResult r = strata::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    int fail = !(r.ok && r.locked_bytes == bytes) | fail_keeps | fail_guard | fail_guard_cfg;
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
