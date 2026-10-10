// src/core/tiered_source.cpp - the VRAM / pinned RAM / SSD expert tiers.  See `TieredExpertSource` in the header.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// STRATA_PREFILL_TIMING: where the tiered source's streamed reads go (see tiered_read_report in expert_source.hpp).
namespace strata::core {
namespace {
std::atomic<int64_t> g_direct_reads{0}, g_direct_bytes{0}, g_direct_us{0};
std::atomic<int64_t> g_fb_reads{0}, g_fb_bytes{0}, g_fb_us{0};
std::atomic<int64_t> g_pin_reads{0}, g_pin_bytes{0}, g_pin_us{0};
inline int64_t us_now() {
    return (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace
std::string tiered_read_report() {
    char b[320];
    std::snprintf(b, sizeof b,
        "tiered streamed reads: direct %lld (%.1f GB, %.0f ms) | mapped-memcpy fallback %lld (%.1f GB, %.0f ms) "
        "| pinned memcpy %lld (%.1f GB, %.0f ms)",
        (long long) g_direct_reads.load(std::memory_order_relaxed), g_direct_bytes.load(std::memory_order_relaxed) / 1e9,
        g_direct_us.load(std::memory_order_relaxed) / 1000.0,
        (long long) g_fb_reads.load(std::memory_order_relaxed), g_fb_bytes.load(std::memory_order_relaxed) / 1e9,
        g_fb_us.load(std::memory_order_relaxed) / 1000.0,
        (long long) g_pin_reads.load(std::memory_order_relaxed), g_pin_bytes.load(std::memory_order_relaxed) / 1e9,
        g_pin_us.load(std::memory_order_relaxed) / 1000.0);
    return b;
}
}  // namespace strata::core

#if defined(_WIN32)
// ---------------------------------------------------------------------------------------------------------------
// THE WINDOWS TIERED SOURCE.  Same three tiers and the same one-time `settle`, but TWO address regions instead of
// one: the COLD tier reads through a file mapping (`base_`) and the PINNED tier lives in its own anonymous region
// (`pin_base_`), because Windows has no `MAP_FIXED` in-place anonymous remap and `cudaHostRegister` refuses
// long-term pins of page-cache pages.  `blob()` dispatches on the tier table, so the kernels still see plain
// pointers and the layout math is unchanged.
// ---------------------------------------------------------------------------------------------------------------
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace strata::core {

namespace {

constexpr uint64_t kAlign = 4096;         // a multiple of every Windows volume's sector size (512 and 4096)
constexpr uint64_t kRegChunk = 1ull << 30; // cudaHostRegister granularity of the PINNED tier: no blob may straddle it
// The driver's page-locked capacity is finite and shared with EVERY later cudaMalloc/cudaHostRegister (the prompt
// path's buffers, the R4 hit path, the draft head).  Upstream measured that registering 30 GiB of a 40 GiB arena
// left the driver unable to page-lock the prompt path's small buffers afterwards, and set 24 GiB as the cap
// (STRATA_PARTIAL_PIN_GIB, expert_source.cpp).  MEASURED HERE: `--host-budget-gib 8` on a 16 GB primary + 12 GB
// helper registered fully ("PINNED 4934 (7.99 GiB, 8 runs, 8.00 GiB locked)") yet the next device allocation still
// failed ("the R4 hit path could not allocate its device buffers") - the registration had consumed the driver's
// headroom even though every chunk succeeded.  So the pin must be capped before the driver is exhausted.
//
// The cap depends on whether a HELPER is present, matching setup's rule (cascade_host_budget): beside a helper the
// primary card's expert cache and the prefill borrow share the driver, so the measured safe ceiling is 6 GiB; on
// one card there is no such sharing and upstream's 24 GiB driver limit applies.  This is NOT derived from free RAM
// here: by the time `settle` runs the model is already resident, so MemAvailable is depleted and subtracting from
// it would collapse the pin to 0 (measured - it did).  The machine-relative RAM bound is already in `budget_bytes`
// (setup writes min(6, avail-8, 0.45*primary); `auto` is avail-minus-reserve read before the pin).  This guard only
// stops a request larger than the driver can safely lock, on any PC, with the ceiling matched to the card layout.
// A chunk that itself fails is still demoted to COLD below; this cap stops the starvation case the demote cannot.
// Override with STRATA_TIERED_PIN_CAP_GIB.
uint64_t tiered_pin_cap_bytes(bool beside_helper) {
    if (const char* v = std::getenv("STRATA_TIERED_PIN_CAP_GIB"); v != nullptr && std::atof(v) > 0)
        return (uint64_t) (std::atof(v) * 1073741824.0);
    return beside_helper ? (6ull << 30) : (24ull << 30);
}

uint64_t page_size() {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    return si.dwPageSize ? si.dwPageSize : 4096;
}

uint64_t alloc_granularity() {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    return si.dwAllocationGranularity ? si.dwAllocationGranularity : (64u << 10);
}

// Free + reclaimable standby physical memory, the closest Windows analog to Linux's MemAvailable.  `ullAvailPhys`
// already counts the standby list, which is where a file mapping's faulted pages live.
int64_t mem_available() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    if (!GlobalMemoryStatusEx(&m)) return -1;
    return (int64_t) m.ullAvailPhys;
}

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
uint64_t align_down(uint64_t v, uint64_t a) { return v / a * a; }

// Converting a native pack: the experts in the layout's blob order, written once to experts.bin.
bool write_experts_bin(const std::string& gguf, const std::string& path, std::string& err) {
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const std::string part = path + ".part";
    HANDLE h = CreateFileA(part.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) { err = "TieredExpertSource: cannot create " + part; return false; }
    LARGE_INTEGER sz{};
    sz.QuadPart = (LONGLONG) lay.total;
    if (!SetFilePointerEx(h, sz, nullptr, FILE_BEGIN) || !SetEndOfFile(h)) {
        CloseHandle(h);
        err = "TieredExpertSource: cannot size " + part;
        return false;
    }
    HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READWRITE, (DWORD) (lay.total >> 32),
                                  (DWORD) (lay.total & 0xffffffffu), nullptr);
    if (m == nullptr) {
        CloseHandle(h);
        err = "TieredExpertSource: cannot map " + part;
        return false;
    }
    void* dst = MapViewOfFile(m, FILE_MAP_ALL_ACCESS, 0, 0, (SIZE_T) lay.total);
    if (dst == nullptr) {
        CloseHandle(m);
        CloseHandle(h);
        err = "TieredExpertSource: cannot map " + part;
        return false;
    }
    std::fprintf(stderr, "strata generate: writing %s from %s (%.2f GiB, one time) ...\n", path.c_str(), gguf.c_str(),
                 (double) lay.total / 1073741824.0);
    const LoadStats st = load_experts_gguf(gguf, (uint8_t*) dst, lay, /*threads=*/6);
    const bool ok = st.seconds >= 0 && FlushViewOfFile(dst, (SIZE_T) lay.total) != 0;
    UnmapViewOfFile(dst);
    CloseHandle(m);
    CloseHandle(h);
    if (!ok || !MoveFileExA(part.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        err = "TieredExpertSource: writing " + path + " failed";
        return false;
    }
    std::fprintf(stderr, "strata generate: wrote %s in %.0f s\n", path.c_str(), st.seconds);
    return true;
}

// One unbuffered read of an aligned superset of [off, off+n).  `FILE_FLAG_NO_BUFFERING` wants aligned offsets,
// lengths and buffers, so the aligned span lands in a per-thread buffer and the blob is copied out - the same
// shape as the Linux `O_DIRECT` path.  Returns false when the span would run past the file (the caller then reads
// through the mapping, which is right for the last unaligned sector).
// Reads the aligned span [a, end) (both multiples of kAlign) from the unbuffered handle into a per-thread buffer;
// returns the buffer base, or null.  `read_direct_span` (one blob) and `read_into_many` (a merged run) both use it -
// the buffer lives until the next read on this thread, so copy the parts out before reading again.
const uint8_t* read_direct_range(HANDLE h, uint64_t a, uint64_t end) {
    if (h == nullptr || h == INVALID_HANDLE_VALUE || end <= a) return nullptr;
    thread_local std::vector<uint8_t> bounce;
    if (bounce.size() < (size_t) (end - a)) bounce.resize((size_t) (end - a));
    uint8_t* buf = bounce.data();
    // An OVERLAPPED with a NULL hEvent is NOT safe with several outstanding reads on one handle: the wait can be
    // signalled by another thread's completion.  Every read gets this thread's own manual-reset event.
    thread_local HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (ev == nullptr) return nullptr;
    for (uint64_t done = 0; done < end - a;) {
        OVERLAPPED ov{};
        const uint64_t at = a + done;
        ov.Offset = (DWORD) (at & 0xffffffffu);
        ov.OffsetHigh = (DWORD) ((at >> 32) & 0xffffffffu);
        ov.hEvent = ev;
        ResetEvent(ev);
        DWORD got = 0;
        const DWORD want = (DWORD) std::min<uint64_t>((end - a) - done, 64u << 20);
        if (!ReadFile(h, buf + done, want, &got, &ov)) {
            if (GetLastError() != ERROR_IO_PENDING || !GetOverlappedResult(h, &ov, &got, TRUE)) return nullptr;
        }
        if (got == 0) return nullptr;
        done += got;
    }
    return buf;
}

// One unbuffered read of an aligned superset of [off, off+n).  `FILE_FLAG_NO_BUFFERING` wants aligned offsets,
// lengths and buffers, so the aligned span lands in a per-thread buffer and the blob is copied out - the same
// shape as the Linux `O_DIRECT` path.  Returns false when the span would run past the file (the caller then reads
// through the mapping, which is right for the last unaligned sector).
bool read_direct_span(HANDLE h, uint64_t file_bytes, uint64_t off, uint8_t* dst, size_t n) {
    if (h == nullptr || h == INVALID_HANDLE_VALUE || n == 0) return false;
    const uint64_t a = align_down(off, kAlign);
    const uint64_t end = align_up(off + n, kAlign);
    if (end > file_bytes) return false;
    const uint8_t* buf = read_direct_range(h, a, end);
    if (buf == nullptr) return false;
    std::memcpy(dst, buf + (size_t) (off - a), n);
    return true;
}

}  // namespace

TieredExpertSource::~TieredExpertSource() { close(); }

bool TieredExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "TieredExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const std::string path = pack_dir + "/experts.bin";
    if (lay.native && !gguf_.empty() && GetFileAttributesA(path.c_str()) == INVALID_FILE_ATTRIBUTES &&
        !write_experts_bin(gguf_, path, err))
        return false;
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        err = "TieredExpertSource: cannot open " + path + " (the tiered source needs a pack with experts.bin)";
        return false;
    }
    LARGE_INTEGER fsz{};
    if (!GetFileSizeEx(h, &fsz) || (uint64_t) fsz.QuadPart != lay.total) {
        char buf[400];
        std::snprintf(buf, sizeof buf, "TieredExpertSource: %s is %llu B but the layout makes %llu B", path.c_str(),
                      (unsigned long long) (fsz.QuadPart < 0 ? 0 : fsz.QuadPart),
                      (unsigned long long) lay.total);
        CloseHandle(h);
        err = buf;
        return false;
    }
    // The whole file is mapped read-only and the address is the OS's choice.  Every read is exactly one blob
    // (`blob_bytes(layer)`) and `lay.total` is the sum of those blobs, so no path ever reads past the file's end.
    // (Linux reserves `max_blob` of anonymous tail for a whole-slot copy that may start at the last expert; the
    // engine does not do that here - prefill and the cache fill both pass `blob_bytes(layer)` - so it is not needed.)
    file_bytes_ = lay.total;
    map_bytes_ = align_up(file_bytes_, alloc_granularity());
    HANDLE map = CreateFileMappingA(h, nullptr, PAGE_READONLY, (DWORD) (file_bytes_ >> 32),
                                    (DWORD) (file_bytes_ & 0xffffffffu), nullptr);
    if (map == nullptr) {
        CloseHandle(h);
        err = "TieredExpertSource: CreateFileMapping failed on " + path + " (error " +
              std::to_string((long long) GetLastError()) + ")";
        return false;
    }
    void* view = MapViewOfFile(map, FILE_MAP_READ, 0, 0, (SIZE_T) file_bytes_);
    if (view == nullptr) {
        CloseHandle(map);
        CloseHandle(h);
        err = "TieredExpertSource: MapViewOfFile failed on " + path + " (error " +
              std::to_string((long long) GetLastError()) + ")";
        return false;
    }
    file_ = h;
    map_ = map;
    view_ = view;
    base_ = (const uint8_t*) view;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    // a second handle for streamed reads that should not go through the page cache (read_into)
    if (std::getenv("STRATA_NO_DIRECT_STREAM") == nullptr) {
        HANDLE d = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (d == INVALID_HANDLE_VALUE) {
            (void) GetLastError();
        } else {
            dfile_ = d;
        }
    }
    // Until `settle` runs every expert is cold: the cache fill reads through the mapping like any other reader.
    tier_.assign((size_t) (n_layers * n_expert), (uint8_t) kCold);
    pin_off_.assign((size_t) (n_layers * n_expert), -1);
    pf_seen_.assign((size_t) (n_layers * n_expert), -1);
    pf_epoch_ = 0;
    note_ = "mapped " + path;
    pf_stop_ = false;
    for (int i = 0; i < 4; ++i)
        pf_threads_.emplace_back([this] {
            for (;;) {
                std::pair<uint64_t, uint64_t> r;
                {
                    std::unique_lock<std::mutex> lk(pf_mu_);
                    pf_cv_.wait(lk, [&] { return pf_stop_ || !pf_q_.empty(); });
                    if (pf_stop_) return;
                    r = pf_q_.front();
                    pf_q_.pop_front();
                }
                WIN32_MEMORY_RANGE_ENTRY e;
                e.VirtualAddress = (PVOID) (base_ + r.first);
                e.NumberOfBytes = (SIZE_T) r.second;
                (void) PrefetchVirtualMemory(GetCurrentProcess(), 1, &e, 0);
            }
        });
    return true;
}

bool TieredExpertSource::settle(const ExpertCache* cache, const uint8_t* also_vram,
                                const std::vector<std::pair<int32_t, int32_t>>& profile,
                                int64_t budget_bytes, int64_t reserve_bytes, int threads, std::string& err) {
    if (base_ == nullptr) { err = "TieredExpertSource::settle: not open"; return false; }
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t gran = alloc_granularity();
    const auto t0 = std::chrono::steady_clock::now();
    other_vram_.assign((size_t) (n_layers_ * n_expert_), 0);

    // ---- 1. the VRAM tier.  Windows has no per-range "drop these clean mapped pages" call (`OfferVirtualMemory`
    // is not it: it can discard contents and the range is inaccessible until `ReclaimVirtualMemory`, and it
    // rejects file mappings).  A mapping's faulted pages sit on the standby list and are reclaimed before
    // anonymous memory, so the drop is best-effort and gated behind STRATA_TRIM_WORKING_SET
    // (SetProcessWorkingSetSize trims the whole process).
    //
    // `cache` is the primary tier and `also_vram` every other VRAM tier (a layer-split stage, a CUDA1..3 helper).
    // Both are VRAM: the point of the mask is that the PINNED loop below must not spend the budget on a blob a GPU
    // already holds, which is exactly what happened when `settle` ran before the CUDA1 helper was filled.
    uint64_t vram_bytes = 0;
    int64_t n_vram = 0, n_shared = 0;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e) {
            const size_t idx = (size_t) (l * n_expert_ + e);
            const bool in_primary = cache != nullptr && cache->slot_of(l, e) != kNotResident;
            const bool in_other = also_vram != nullptr && also_vram[idx] != 0;
            if (!in_primary && !in_other) continue;
            tier_[idx] = kVram;
            if (in_other) { other_vram_[idx] = 1; ++n_shared; }
            vram_bytes += lay.blob_bytes(l);
            ++n_vram;
        }
    if (std::getenv("STRATA_TRIM_WORKING_SET") != nullptr)
        (void) SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T) -1, (SIZE_T) -1);

    // ---- 2. the budget.  Windows caps page-locked memory well under the RAM size, so the registration is done in
    // chunks and a refusal demotes the rest of the profile order rather than failing the run.
    if (budget_bytes < 0) {
        const int64_t avail = mem_available();
        budget_bytes = avail > reserve_bytes ? avail - reserve_bytes : 0;
    }
    int64_t pin_cap = (int64_t) 8 << 30;
    if (const char* v = std::getenv("STRATA_PIN_CAP_GIB")) pin_cap = (int64_t) (std::atof(v) * 1073741824.0);
    if (pin_cap > 0 && budget_bytes > pin_cap) budget_bytes = pin_cap;

    // ---- 3. the PINNED tier: the profile's order past the cache, while the budget lasts.  A pair missing from
    // the profile is never routed in the profiling traces, so it is the right one to leave cold.
    std::vector<std::pair<int32_t, int32_t>> chosen;
    uint64_t pinned_bytes = 0;
    for (const auto& pr : profile) {
        const int64_t l = pr.first, e = pr.second;
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) continue;
        uint8_t& t = tier_[(size_t) (l * n_expert_ + e)];
        if (t != kCold) continue;
        const uint64_t n = lay.blob_bytes(l);
        if (pinned_bytes + n > (uint64_t) budget_bytes) break;
        t = kPinned;
        pinned_bytes += align_up(n, 64);
        chosen.push_back(pr);
    }
    int64_t n_pinned = (int64_t) chosen.size();

    // ---- 4. one contiguous anonymous region for the whole pinned set, filled from the file with unbuffered reads.
    uint64_t locked = 0;
    int64_t failed_chunks = 0;
    std::string first_fail;
    if (pinned_bytes > 0) {
        // Lay the pinned blobs out FIRST, then allocate exactly what the layout needs.  Each blob is kept whole
        // inside one registration chunk (`kRegChunk`): a blob straddling a chunk edge lies partly outside every
        // registered run, and a cudaMemcpy from it fails with "invalid argument".
        struct Fill { uint64_t off, dst, n; };
        std::vector<Fill> fills;
        uint64_t cursor = 0;
        for (const auto& pr : chosen) {
            const int64_t l = pr.first, e = pr.second;
            const size_t idx = (size_t) (l * n_expert_ + e);
            const uint64_t n = lay.blob_bytes(l), need = align_up(n, 64);
            const uint64_t in_chunk = cursor % kRegChunk;
            if (in_chunk != 0 && in_chunk + need > kRegChunk) cursor = align_up(cursor, kRegChunk);
            pin_off_[idx] = (int64_t) cursor;
            fills.push_back({lay.blob_offset(l, e), cursor, n});
            cursor += need;
        }
        pin_bytes_ = align_up(cursor, gran);
        pin_base_ = (uint8_t*) VirtualAlloc(nullptr, (SIZE_T) pin_bytes_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (pin_base_ == nullptr) {
            err = "TieredExpertSource::settle: the pinned region could not be allocated";
            return false;
        }
        std::atomic<size_t> next{0};
        auto worker = [&] {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= fills.size()) return;
                const Fill& f = fills[i];
                if (!read_direct_span((HANDLE) dfile_, file_bytes_, f.off, pin_base_ + f.dst, (size_t) f.n))
                    std::memcpy(pin_base_ + f.dst, base_ + f.off, (size_t) f.n);   // the file's last sector
            }
        };
        std::vector<std::thread> pool;
        for (int i = 1; i < std::max(threads, 1); ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();

        // **A SELF-CHECK ON THE STREAMED FILL.**  These bytes do not come from the mapping, so a bug in the
        // unbuffered read is a plausible-looking token.  Compare a sample against the mapping.
        if (std::getenv("STRATA_TIERED_VERIFY") != nullptr && !fills.empty()) {
            const size_t step = std::max<size_t>(1, fills.size() / 32);
            int64_t bad = 0, checked = 0;
            for (size_t i = 0; i < fills.size() && checked < 32; i += step) {
                const Fill& f = fills[i];
                if (std::memcmp(pin_base_ + f.dst, base_ + f.off, (size_t) f.n) != 0) ++bad;
                ++checked;
            }
            std::fprintf(stderr, "strata generate: tiered fill self-check: %lld of %lld sampled blobs differ\n",
                         (long long) bad, (long long) checked);
        }

        // ---- 5. lock it.  Windows refuses past a limit; 1 GiB chunks mean a refusal demotes the lowest-ranked
        // experts (the tail of the profile order) and keeps the ones decode routes most.  Registration stops at a
        // ceiling matched to the card layout (6 GiB beside a helper, 24 GiB on one card) so it does not starve a
        // later allocation - see tiered_pin_cap_bytes.
        const uint64_t chunk = kRegChunk;
        const uint64_t reg_cap = tiered_pin_cap_bytes(n_shared > 0);
        for (uint64_t off = 0; off < pin_bytes_ && off < reg_cap; off += chunk) {
            const uint64_t n = std::min<uint64_t>(chunk, std::min(pin_bytes_ - off, reg_cap - off));
            void* p = pin_base_ + off;
            const cudaError_t e = cudaHostRegister(p, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped);
            if (e != cudaSuccess) {
                (void) cudaGetLastError();   // consume it: a sticky error lies about the next launch
                if (first_fail.empty()) first_fail = cudaGetErrorString(e);
                ++failed_chunks;
                continue;
            }
            regs_.push_back({(uint8_t*) p, n});
            locked += n;
        }
        if (pin_bytes_ > reg_cap) ++failed_chunks;   // the bytes past the cap were never attempted
        std::sort(regs_.begin(), regs_.end(),
                  [](const std::pair<uint8_t*, uint64_t>& a, const std::pair<uint8_t*, uint64_t>& b) {
                      return a.first < b.first;
                  });
        // A refused chunk means those experts are NOT page-locked, so they must not stay kPinned - the tier table
        // is the contract the rest of the engine reads (device_alias, blob, begin_layer).  Without this they kept
        // their kPinned mark, the engine reported a PINNED tier it did not actually hold, and the driver's reduced
        // page-lock capacity then broke a later allocation (measured: `--host-budget-gib 8` on a 16 GB primary
        // logged "cudaHostRegister refused chunks" and then "the R4 hit path could not allocate its device
        // buffers").  Demoting them to COLD - what the Linux path already does for a refused run - makes the
        // engine report the honest PINNED size and read those experts from the SSD instead of failing later.
        if (failed_chunks > 0) {
            for (int64_t l = 0; l < n_layers_; ++l)
                for (int64_t e = 0; e < n_expert_; ++e) {
                    const size_t idx = (size_t) (l * n_expert_ + e);
                    if (tier_[idx] != kPinned) continue;
                    if (!pinned(l, e)) {
                        tier_[idx] = kCold;
                        pinned_bytes -= lay.blob_bytes(l);
                        --n_pinned;
                    }
                }
        }
    }

    uint64_t cold_bytes = 0;
    int64_t n_cold = 0;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e)
            if (tier_[(size_t) (l * n_expert_ + e)] == kCold) {
                cold_bytes += lay.blob_bytes(l);
                ++n_cold;
            }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto gib = [](uint64_t b) { return (double) b / 1073741824.0; };
    char buf[600];
    std::snprintf(buf, sizeof buf,
                  "tiers: VRAM %lld experts (%.2f GiB, %lld held by another GPU, left to the standby list) | "
                  "PINNED %lld (%.2f GiB, %zu runs, %.2f GiB locked) | COLD %lld (%.2f GiB, paged from SSD) | "
                  "budget %.2f GiB | %.1f s%s%s",
                  (long long) n_vram, gib(vram_bytes), (long long) n_shared, (long long) n_pinned,
                  gib(pinned_bytes), regs_.size(), gib(locked), (long long) n_cold, gib(cold_bytes),
                  gib((uint64_t) budget_bytes), secs,
                  failed_chunks ? "; cudaHostRegister refused chunks: " : "", failed_chunks ? first_fail.c_str() : "");
    note_ = buf;
    return true;
}

// The adaptive swap moved a pair into the primary cache.  If it was PINNED, two tiers now hold the same blob; the
// only one any path reads is the slot, so demote it.  Safe to call off the pool thread: `apply_pending` runs the
// same thread that computes, and a byte store to `tier_` cannot tear.
void TieredExpertSource::promote_to_vram(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return;
    const size_t idx = (size_t) (layer * n_expert_ + expert);
    if (tier_[idx] != kPinned) return;
    // KEEP the pinned copy.  Its bytes stay locked either way, and if an adaptive swap later evicts this pair from
    // VRAM the CPU reads it from locked RAM instead of re-reading the SSD - the cheap RAM home the adaptive tier
    // needs.  The slot stays authoritative while host_res marks the pair resident.
    if (!other_vram_.empty()) other_vram_[idx] = 0;
    ++retained_pins_;
}

void TieredExpertSource::close() {
    {
        std::lock_guard<std::mutex> lk(pf_mu_);
        pf_stop_ = true;
        pf_q_.clear();
    }
    pf_cv_.notify_all();
    for (auto& t : pf_threads_) t.join();
    pf_threads_.clear();
    for (const auto& r : regs_) cudaHostUnregister(r.first);
    regs_.clear();
    if (pin_base_ != nullptr) {
        VirtualFree(pin_base_, 0, MEM_RELEASE);
        pin_base_ = nullptr;
    }
    pin_bytes_ = 0;
    if (view_ != nullptr) {
        UnmapViewOfFile(view_);
        view_ = nullptr;
    }
    if (tail_ != nullptr) {
        (void) VirtualFree(tail_, (SIZE_T) tail_bytes_, MEM_DECOMMIT);
        tail_ = nullptr;
        tail_bytes_ = 0;
    }
    if (map_ != nullptr) {
        CloseHandle((HANDLE) map_);
        map_ = nullptr;
    }
    if (file_ != nullptr) {
        CloseHandle((HANDLE) file_);
        file_ = nullptr;
    }
    if (dfile_ != nullptr) {
        CloseHandle((HANDLE) dfile_);
        dfile_ = nullptr;
    }
    base_ = nullptr;
    map_bytes_ = 0;
    file_bytes_ = 0;
    tier_.clear();
    pin_off_.clear();
}

const uint8_t* TieredExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return nullptr;
    ++reads_;
    const size_t idx = (size_t) (layer * n_expert_ + expert);
    const uint8_t t = tier_[idx];
    if (t == kPinned) {
        ++pinned_reads_;
        return pin_base_ + pin_off_[idx];
    }
    if (t == kCold) {
        ++cold_reads_;
        cold_read_bytes_ += strata::kernels::cpu::expert_layout().blob_bytes(layer);
        if (!pf_seen_.empty() && pf_seen_[idx] == pf_epoch_) ++cold_pf_reads_;
    }
    return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

bool TieredExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    const size_t idx = (size_t) (layer * n_expert_ + expert);
    if (tier_[idx] != kPinned) return false;
    // A pinned blob must lie inside one registered run, so a refused registration can never hand the GPU an
    // unlocked address.  The runs are in address order: binary search.
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint8_t* a = pin_base_ + pin_off_[idx];
    const uint8_t* b = a + lay.blob_bytes(layer);
    auto it = std::upper_bound(regs_.begin(), regs_.end(), a,
                               [](const uint8_t* p, const std::pair<uint8_t*, uint64_t>& r) { return p < r.first; });
    if (it == regs_.begin()) return false;
    --it;
    return a >= it->first && b <= it->first + it->second;
}

const uint8_t* TieredExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert)) return nullptr;
    const uint8_t* h = pin_base_ + pin_off_[(size_t) (layer * n_expert_ + expert)];
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, (void*) h, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return (const uint8_t*) d;
}

void TieredExpertSource::read_into(const uint8_t* src, uint8_t* dst, size_t n) const {
    const int64_t t0 = us_now();
    // Pinned or otherwise anonymous bytes are in RAM already: memcpy.
    if (pin_base_ != nullptr && src >= pin_base_ && src + n <= pin_base_ + pin_bytes_) {
        std::memcpy(dst, src, n);
        g_pin_reads.fetch_add(1, std::memory_order_relaxed); g_pin_bytes.fetch_add((int64_t) n, std::memory_order_relaxed);
        g_pin_us.fetch_add(us_now() - t0, std::memory_order_relaxed);
        return;
    }
    if (base_ != nullptr && src >= base_ && (uint64_t) (src - base_) + n <= file_bytes_) {
        const uint64_t off = (uint64_t) (src - base_);
        // inside a registered run?  then it is page-locked RAM, not a file read
        auto it = std::upper_bound(regs_.begin(), regs_.end(), src,
                                   [](const uint8_t* p, const std::pair<uint8_t*, uint64_t>& r) { return p < r.first; });
        const bool locked = it != regs_.begin() && std::prev(it)->first + std::prev(it)->second >= src + n;
        if (!locked) {
            // A streamed expert is read once per prompt chunk; through the page cache ~20 GB per long prompt would
            // evict the cold-tier pages decode relies on.  One unbuffered read instead of ~400 page faults.
            if (read_direct_span((HANDLE) dfile_, file_bytes_, off, dst, n)) {
                g_direct_reads.fetch_add(1, std::memory_order_relaxed); g_direct_bytes.fetch_add((int64_t) n, std::memory_order_relaxed);
                g_direct_us.fetch_add(us_now() - t0, std::memory_order_relaxed);
                return;
            }
        }
    }
    std::memcpy(dst, src, n);
    g_fb_reads.fetch_add(1, std::memory_order_relaxed); g_fb_bytes.fetch_add((int64_t) n, std::memory_order_relaxed);
    g_fb_us.fetch_add(us_now() - t0, std::memory_order_relaxed);
}

// Batched cold read.  The prompt path batches consecutive experts of one layer and `experts.bin` stores a layer's
// experts back to back, so a batch's file spans are contiguous: one request for the run is far faster on an NVMe
// than one request per blob (measured ~2.3 GB/s at 2 MiB vs ~6.6 GB/s at 8 MiB on the reference rig).  A blob that
// is pinned/locked (or past the file) falls back to a memcpy / the single-blob path, exactly as `read_into` does.
void TieredExpertSource::read_into_many(const uint8_t* const* srcs, uint8_t* const* dsts,
                                        const size_t* ns, size_t count) const {
    if (count == 0) return;
    constexpr uint64_t kMergeTarget = 8ull << 20;   // keep each merged request near the drive's sweet spot
    auto is_file = [&](const uint8_t* s, size_t n) -> bool {
        if (base_ == nullptr || s < base_ || (uint64_t) (s - base_) + n > file_bytes_) return false;
        auto it = std::upper_bound(regs_.begin(), regs_.end(), s,
                                   [](const uint8_t* p, const std::pair<uint8_t*, uint64_t>& r) { return p < r.first; });
        return !(it != regs_.begin() && std::prev(it)->first + std::prev(it)->second >= s + n);
    };
    size_t i = 0;
    while (i < count) {
        if (!is_file(srcs[i], ns[i])) {   // pinned/otherwise-RAM bytes: memcpy, as read_into would
            std::memcpy(dsts[i], srcs[i], ns[i]);
            g_fb_reads.fetch_add(1, std::memory_order_relaxed);
            g_fb_bytes.fetch_add((int64_t) ns[i], std::memory_order_relaxed);
            ++i;
            continue;
        }
        const uint64_t off0 = (uint64_t) (srcs[i] - base_);
        const uint64_t a = align_down(off0, kAlign);
        uint64_t end = off0 + ns[i];
        size_t j = i + 1;
        while (j < count) {   // extend over the following contiguous file blobs, up to the size cap
            if (!is_file(srcs[j], ns[j])) break;
            const uint64_t o = (uint64_t) (srcs[j] - base_);
            const uint64_t e = o + ns[j];
            if (std::max(end, e) - a > kMergeTarget) break;
            if (e > end) end = e;
            ++j;
        }
        const int64_t t0 = us_now();
        const uint64_t end_al = align_up(end, kAlign);
        const uint8_t* buf = end_al <= file_bytes_ ? read_direct_range((HANDLE) dfile_, a, end_al) : nullptr;
        if (buf == nullptr) {
            for (size_t k = i; k < j; ++k) read_into(srcs[k], dsts[k], ns[k]);   // last unaligned sector
        } else {
            for (size_t k = i; k < j; ++k)
                std::memcpy(dsts[k], buf + ((uint64_t) (srcs[k] - base_) - a), ns[k]);
            g_direct_reads.fetch_add(1, std::memory_order_relaxed);
            g_direct_bytes.fetch_add((int64_t) (end - off0), std::memory_order_relaxed);
            g_direct_us.fetch_add(us_now() - t0, std::memory_order_relaxed);
        }
        i = j;
    }
}

void TieredExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    if (base_ == nullptr || ids == nullptr || layer < 0 || layer >= n_layers_) return;
    ++pf_epoch_;   // one epoch per layer; blob() counts a cold read as prefetched only against the current layer
    static const bool no_pf = std::getenv("STRATA_NO_COLD_PREFETCH") != nullptr;   // the A/B arm
    // The memory guard's pause-on-pressure starved decode: with the prefetch off, every cold miss faulted from the
    // SSD synchronously and the CPU pool's per-window time tripled (measured: 0% prefetched, ~20 t/s vs 97%, ~48).
    // Off by default now; STRATA_PREFETCH_PAUSE_ON_PRESSURE=1 restores the old policy.
    static const bool pause_pf = std::getenv("STRATA_PREFETCH_PAUSE_ON_PRESSURE") != nullptr;
    if (no_pf || (pause_pf && strata::platform::memory_pressure_low())) return;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t pg = page_size();
    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        const size_t idx = (size_t) (layer * n_expert_ + e);
        const uint8_t t = tier_[idx];
        if (t == kPinned) continue;
        // A blob another VRAM tier holds is computed there in every window the CPU pool sees, so prefetching it
        // would pull 1.4 MB off the SSD per token to warm pages nothing dereferences.  Prefill never reaches here
        // (it reads through `read_into`), so this costs prefill nothing.
        if (!other_vram_.empty() && other_vram_[idx]) continue;
        // a VRAM-tier expert has no RAM copy; it needs one only once the adaptive swap has evicted it
        if (t == kVram && (res_ == nullptr || res_[idx] >= 0)) continue;
        // One prefetch for the whole blob instead of ~340 page faults in the pool.  Asynchronous: it overlaps the
        // pool's work on the experts already in RAM.
        const uint64_t off = lay.blob_offset(layer, e);
        uint64_t a = align_down(off, pg);
        uint64_t b = std::min(align_up(off + lay.blob_bytes(layer), pg), align_up(file_bytes_, pg));
        if (a >= b) continue;
        if (!pf_seen_.empty()) pf_seen_[idx] = pf_epoch_;
        ++cold_prefetches_;
        WIN32_MEMORY_RANGE_ENTRY ent;
        ent.VirtualAddress = (PVOID) (base_ + a);
        ent.NumberOfBytes = (SIZE_T) (b - a);
        // STRATA_SYNC_PREFETCH: the old arm, on the pool's own thread (it can block while the device queue is
        // full, and the pool's plan phase waited for it).  Otherwise a helper thread takes the range.
        static const bool sync_pf = std::getenv("STRATA_SYNC_PREFETCH") != nullptr;
        if (sync_pf) {
            (void) PrefetchVirtualMemory(GetCurrentProcess(), 1, &ent, 0);
            continue;
        }
        (void) ent;
        {
            std::lock_guard<std::mutex> lk(pf_mu_);
            pf_q_.push_back({a, b - a});
        }
        pf_cv_.notify_one();
    }
}

}  // namespace strata::core
#else
namespace strata::core {

namespace {

// Converting a native pack: the experts in the layout's blob order, written once to experts.bin.  The arena's own
// loader does the per-expert gather (gate | up | down per blob), so the bytes are the ones the arena would hold.
bool write_experts_bin(const std::string& gguf, const std::string& path, std::string& err) {
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const std::string part = path + ".part";
    const int fd = ::open(part.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { err = "TieredExpertSource: cannot create " + part; return false; }
    if (ftruncate(fd, (off_t) lay.total) != 0) { ::close(fd); err = "TieredExpertSource: cannot size " + part; return false; }
    void* dst = mmap(nullptr, (size_t) lay.total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (dst == MAP_FAILED) { ::close(fd); err = "TieredExpertSource: cannot map " + part; return false; }
    std::fprintf(stderr, "strata generate: writing %s from %s (%.2f GiB, one time) ...\n", path.c_str(), gguf.c_str(),
                 (double) lay.total / 1073741824.0);
    const LoadStats st = load_experts_gguf(gguf, (uint8_t*) dst, lay, /*threads=*/6);
    const bool ok = st.seconds >= 0 && msync(dst, (size_t) lay.total, MS_SYNC) == 0;
    munmap(dst, (size_t) lay.total);
    ::close(fd);
    if (!ok || std::rename(part.c_str(), path.c_str()) != 0) {
        err = "TieredExpertSource: writing " + path + " failed";
        return false;
    }
    std::fprintf(stderr, "strata generate: wrote %s in %.0f s\n", path.c_str(), st.seconds);
    return true;
}

uint64_t page_size() {
    static const uint64_t p = (uint64_t) sysconf(_SC_PAGESIZE);
    return p;
}

// MemAvailable, in bytes: the kernel's own estimate of what can be allocated without swapping, which counts the
// reclaimable page cache - including the resident experts' pages this source has just dropped.
int64_t mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    int64_t kb = 0;
    std::string unit;
    while (f >> key >> kb >> unit)
        if (key == "MemAvailable:") return kb * 1024;
    return -1;
}

}  // namespace

TieredExpertSource::~TieredExpertSource() { close(); }

bool TieredExpertSource::open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "TieredExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    const std::string path = pack_dir + "/experts.bin";
    if (lay.native && !gguf_.empty() && !std::ifstream(path, std::ios::binary) && !write_experts_bin(gguf_, path, err))
        return false;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "TieredExpertSource: cannot open " + path + " (the tiered source needs a pack with experts.bin)";
        return false;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || (uint64_t) st.st_size != lay.total) {
        char buf[400];
        std::snprintf(buf, sizeof buf, "TieredExpertSource: %s is %llu B but the layout makes %llu B", path.c_str(),
                      (unsigned long long) st.st_size, (unsigned long long) lay.total);
        ::close(fd);
        err = buf;
        return false;
    }
    // The arena is one largest blob longer than the file (a copy of a whole VRAM slot may start at any expert);
    // past the file's last page a file mapping raises SIGBUS, so the file is mapped over an anonymous reservation
    // and the tail stays anonymous zeros.
    const uint64_t pg = page_size();
    file_bytes_ = lay.total;
    map_bytes_ = (lay.total + lay.max_blob + pg - 1) / pg * pg;
    void* res = mmap(nullptr, (size_t) map_bytes_, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (res == MAP_FAILED) {
        ::close(fd);
        err = "TieredExpertSource: the address reservation failed";
        return false;
    }
    void* view = mmap(res, (size_t) file_bytes_, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0);
    if (view == MAP_FAILED) {
        munmap(res, (size_t) map_bytes_);
        ::close(fd);
        err = "TieredExpertSource: mmap failed on " + path;
        return false;
    }
    fd_ = fd;
    // a second descriptor for streaming reads that should not go through the page cache (read_into)
    dfd_ = std::getenv("STRATA_NO_DIRECT_STREAM") ? -1 : ::open(path.c_str(), O_RDONLY | O_DIRECT);
    base_ = (const uint8_t*) res;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    // Until `settle` runs every expert is cold: the cache fill reads through the mapping like any other reader.
    tier_.assign((size_t) (n_layers * n_expert), (uint8_t) kCold);
    note_ = "mapped " + path;
    pf_stop_ = false;
    for (int i = 0; i < 4; ++i)
        pf_threads_.emplace_back([this] {
            for (;;) {
                std::pair<uint64_t, uint64_t> r;
                {
                    std::unique_lock<std::mutex> lk(pf_mu_);
                    pf_cv_.wait(lk, [&] { return pf_stop_ || !pf_q_.empty(); });
                    if (pf_stop_) return;
                    r = pf_q_.front();
                    pf_q_.pop_front();
                }
                madvise((void*) (base_ + r.first), (size_t) r.second, MADV_WILLNEED);
            }
        });
    return true;
}

bool TieredExpertSource::settle(const ExpertCache* cache, const uint8_t* also_vram,
                                const std::vector<std::pair<int32_t, int32_t>>& profile,
                                int64_t budget_bytes, int64_t reserve_bytes, int threads, std::string& err) {
    if (base_ == nullptr) { err = "TieredExpertSource::settle: not open"; return false; }
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t pg = page_size();
    const auto t0 = std::chrono::steady_clock::now();
    other_vram_.assign((size_t) (n_layers_ * n_expert_), 0);

    // ---- 1. the VRAM tier: drop its pages.  Only whole pages inside the blob, so a neighbour's bytes that share
    // an edge page stay mapped (dropping a clean page is harmless anyway - it would only be read again).  A pair
    // resident in a *different* VRAM tier (a stage, a CUDA1..3 helper) is VRAM too: `also_vram` names those, and
    // the PINNED loop below must not spend the budget on a blob a GPU already holds.
    uint64_t vram_bytes = 0;
    int64_t n_vram = 0, n_shared = 0;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e) {
            const size_t idx = (size_t) (l * n_expert_ + e);
            const bool in_primary = cache != nullptr && cache->slot_of(l, e) != kNotResident;
            const bool in_other = also_vram != nullptr && also_vram[idx] != 0;
            if (!in_primary && !in_other) continue;
            tier_[idx] = kVram;
            if (in_other) { other_vram_[idx] = 1; ++n_shared; }
            const uint64_t off = lay.blob_offset(l, e), n = lay.blob_bytes(l);
            const uint64_t a = (off + pg - 1) / pg * pg, b = (off + n) / pg * pg;
            if (b > a) {
                madvise((void*) (base_ + a), (size_t) (b - a), MADV_DONTNEED);
                posix_fadvise(fd_, (off_t) a, (off_t) (b - a), POSIX_FADV_DONTNEED);
            }
            vram_bytes += n;
            ++n_vram;
        }

    // ---- 2. the budget, read AFTER the drop so the cache fill's pages count as free
    if (budget_bytes < 0) {
        const int64_t avail = mem_available();
        budget_bytes = avail > reserve_bytes ? avail - reserve_bytes : 0;
    }

    // ---- 3. the PINNED tier: the profile's order past the cache, while the budget lasts.  A pair missing from
    // the profile is never routed in the profiling traces, so it is the right one to leave cold.
    uint64_t pinned_bytes = 0;
    int64_t n_pinned = 0;
    for (const auto& pr : profile) {
        const int64_t l = pr.first, e = pr.second;
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) continue;
        uint8_t& t = tier_[(size_t) (l * n_expert_ + e)];
        if (t != kCold) continue;
        const uint64_t n = lay.blob_bytes(l);
        if (pinned_bytes + n > (uint64_t) budget_bytes) break;
        t = kPinned;
        pinned_bytes += n;
        ++n_pinned;
    }

    // ---- 4. merge the pinned blobs into runs in FILE order, across layer boundaries, and widen each to whole
    // pages.  Two registrations must not share a page (`cudaHostRegister` refuses an overlap), and runs split by
    // at least one other blob (>= 1 MB) never do; runs that would touch are merged by construction.
    struct Run { uint64_t a, b; };
    std::vector<Run> runs;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e) {
            if (tier_[(size_t) (l * n_expert_ + e)] != kPinned) continue;
            const uint64_t off = lay.blob_offset(l, e), end = off + lay.blob_bytes(l);
            const uint64_t a = off / pg * pg, b = (end + pg - 1) / pg * pg;
            if (!runs.empty() && a <= runs.back().b) runs.back().b = std::max(runs.back().b, b);
            else runs.push_back({a, b});
        }

    // ---- 5. turn each run into anonymous memory IN PLACE, fill it from the file, then lock it.
    //
    // **THE DRIVER DOES NOT PIN FILE-BACKED PAGES.**  Measured on this machine (driver 610, kernel 7.2):
    // `cudaHostRegister` on the `MAP_SHARED` file mapping returns "operation not supported" for every run, with
    // or without `cudaHostRegisterReadOnly` - the kernel refuses long-term pins of page-cache pages.  So the run's
    // slice of the mapping is replaced (`MAP_FIXED`) by anonymous memory at the SAME address and filled with
    // `pread`: `blob_offset` still lands on the same bytes, and anonymous memory is what the arena always pinned.
    // Edge pages shared with a neighbouring blob are filled from the file too, so the neighbour's bytes stay right.
    std::vector<uint8_t> anon_ok(runs.size(), 0);
    {
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= runs.size()) return;
                uint8_t* p = (uint8_t*) base_ + runs[i].a;
                // the run's last page may extend past the file: only the file's bytes are read, the rest is zero
                const uint64_t end = std::min(runs[i].b, file_bytes_);
                void* m = mmap(p, (size_t) (runs[i].b - runs[i].a), PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
                if (m == MAP_FAILED) continue;
                bool good = true;
                for (uint64_t off = runs[i].a; off < end;) {
                    const ssize_t got = pread(fd_, p + (off - runs[i].a), (size_t) std::min<uint64_t>(end - off, 64u << 20),
                                              (off_t) off);
                    if (got <= 0) { good = false; break; }
                    off += (uint64_t) got;
                }
                if (good) anon_ok[i] = 1;
            }
        };
        std::vector<std::thread> pool;
        for (int i = 1; i < std::max(threads, 1); ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    uint64_t locked = 0;
    int64_t failed_runs = 0;
    std::string first_fail;
    for (size_t i = 0; i < runs.size(); ++i) {
        const Run& r = runs[i];
        void* p = (void*) (base_ + r.a);
        const cudaError_t e = anon_ok[i] ? cudaHostRegister(p, (size_t) (r.b - r.a),
                                                            cudaHostRegisterPortable | cudaHostRegisterMapped)
                                         : cudaErrorInvalidValue;
        if (e != cudaSuccess) {
            // Consume the error (see pinned.cu: a sticky error lies about the next launch), and put the file
            // mapping back: an unpinned anonymous copy would only be swapped (zram) instead of dropped.
            (void) cudaGetLastError();
            if (first_fail.empty()) first_fail = anon_ok[i] ? cudaGetErrorString(e) : "the anonymous fill failed";
            mmap(p, (size_t) (std::min(r.b, file_bytes_) - r.a), PROT_READ, MAP_SHARED | MAP_FIXED, fd_, (off_t) r.a);
            ++failed_runs;
            continue;
        }
        mprotect(p, (size_t) (r.b - r.a), PROT_READ);   // the kernels only read; a stray write should fault
        regs_.push_back({(uint8_t*) p, r.b - r.a});
        locked += r.b - r.a;
    }
    // the fill's reads went through the page cache as well: those copies are now duplicates of the anonymous ones
    for (const Run& r : runs) posix_fadvise(fd_, (off_t) r.a, (off_t) (std::min(r.b, file_bytes_) - r.a), POSIX_FADV_DONTNEED);
    if (failed_runs > 0) {
        // demote every pinned blob that is not inside a registered run
        for (int64_t l = 0; l < n_layers_; ++l)
            for (int64_t e = 0; e < n_expert_; ++e) {
                uint8_t& t = tier_[(size_t) (l * n_expert_ + e)];
                if (t != kPinned) continue;
                if (!pinned(l, e)) {
                    t = kCold;
                    pinned_bytes -= lay.blob_bytes(l);
                    --n_pinned;
                }
            }
    }

    // `pinned()` above consults `regs_` only while settling; from here on the tier table is the answer.
    uint64_t cold_bytes = 0;
    int64_t n_cold = 0;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e)
            if (tier_[(size_t) (l * n_expert_ + e)] == kCold) {
                cold_bytes += lay.blob_bytes(l);
                ++n_cold;
            }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto gib = [](uint64_t b) { return (double) b / 1073741824.0; };
    char buf[600];
    std::snprintf(buf, sizeof buf,
                  "tiers: VRAM %lld experts (%.2f GiB, %lld held by another GPU, dropped from RAM) | "
                  "PINNED %lld (%.2f GiB, %zu runs, %.2f GiB locked) | COLD %lld (%.2f GiB, paged from SSD) | "
                  "budget %.2f GiB | %.1f s%s%s",
                  (long long) n_vram, gib(vram_bytes), (long long) n_shared, (long long) n_pinned,
                  gib(pinned_bytes), regs_.size(), gib(locked), (long long) n_cold, gib(cold_bytes),
                  gib((uint64_t) budget_bytes), secs,
                  failed_runs ? "; cudaHostRegister refused runs: " : "", failed_runs ? first_fail.c_str() : "");
    note_ = buf;
    return true;
}

// The adaptive swap moved a pair into the primary cache; if it was PINNED, two tiers now hold the same blob.  The
// only copy any path reads is the new slot, so demote it.
void TieredExpertSource::promote_to_vram(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return;
    const size_t idx = (size_t) (layer * n_expert_ + expert);
    if (tier_[idx] != kPinned) return;
    // KEEP the pinned copy.  Its bytes stay locked either way, and if an adaptive swap later evicts this pair from
    // VRAM the CPU reads it from locked RAM instead of re-reading the SSD - the cheap RAM home the adaptive tier
    // needs.  The slot stays authoritative while host_res marks the pair resident.
    if (!other_vram_.empty()) other_vram_[idx] = 0;
    ++retained_pins_;
}

void TieredExpertSource::close() {
    {
        std::lock_guard<std::mutex> lk(pf_mu_);
        pf_stop_ = true;
        pf_q_.clear();
    }
    pf_cv_.notify_all();
    for (auto& t : pf_threads_) t.join();
    pf_threads_.clear();
    for (const auto& r : regs_) cudaHostUnregister(r.first);
    regs_.clear();
    if (base_ != nullptr) munmap((void*) base_, (size_t) map_bytes_);
    if (fd_ >= 0) ::close(fd_);
    if (dfd_ >= 0) ::close(dfd_);
    base_ = nullptr;
    fd_ = -1;
    dfd_ = -1;
    tier_.clear();
}

const uint8_t* TieredExpertSource::blob(int64_t layer, int64_t expert) {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return nullptr;
    ++reads_;
    // Linux pins in place (MAP_FIXED anonymous remap), so the pointer is the mapping's either way; the tier only
    // classifies the read for the report.
    const size_t idx = (size_t) (layer * n_expert_ + expert);
    const uint8_t t = tier_[idx];
    if (t == kPinned) {
        ++pinned_reads_;
    } else if (t == kCold) {
        ++cold_reads_;
        cold_read_bytes_ += strata::kernels::cpu::expert_layout().blob_bytes(layer);
        if (!pf_seen_.empty() && pf_seen_[idx] == pf_epoch_) ++cold_pf_reads_;
    }
    return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
}

bool TieredExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (base_ == nullptr || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    if (tier_[(size_t) (layer * n_expert_ + expert)] != kPinned) return false;
    // A pinned blob lies inside one registered run; checked against the runs so a refused registration can never
    // hand the GPU an unlocked address.  The runs are in address order: binary search.
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint8_t* a = base_ + lay.blob_offset(layer, expert);
    const uint8_t* b = a + lay.blob_bytes(layer);
    auto it = std::upper_bound(regs_.begin(), regs_.end(), a,
                               [](const uint8_t* p, const std::pair<uint8_t*, uint64_t>& r) { return p < r.first; });
    if (it == regs_.begin()) return false;
    --it;
    return a >= it->first && b <= it->first + it->second;
}

const uint8_t* TieredExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert)) return nullptr;
    // Unified addressing (every 64-bit Linux CUDA context): a mapped registration's device address is its host
    // address.  Asked of the driver rather than assumed, once per call - this path runs a few times per layer.
    const uint8_t* h = base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, (void*) h, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    return (const uint8_t*) d;
}

void TieredExpertSource::read_into(const uint8_t* src, uint8_t* dst, size_t n) const {
    // Inside the file-backed part and not in a locked run: one pread (a single large request to the NVMe) instead
    // of faulting ~400 pages in through the mapping with 128 KB of readahead.  Pinned or anonymous bytes are in
    // RAM already: memcpy.
    if (base_ != nullptr && src >= base_ && src + n <= base_ + file_bytes_) {
        const uint8_t* p = src;
        auto it = std::upper_bound(regs_.begin(), regs_.end(), p,
                                   [](const uint8_t* q, const std::pair<uint8_t*, uint64_t>& r) { return q < r.first; });
        const bool locked = it != regs_.begin() && p < (std::prev(it))->first + (std::prev(it))->second;
        if (!locked) {
            // O_DIRECT: a streamed expert is read once per prompt chunk, and through the page cache ~20 GB per long
            // prompt would evict the cold-tier pages decode relies on.  Direct I/O wants the offset, length and
            // buffer page-aligned, so the aligned superset lands in a per-thread buffer and the blob is copied out.
            if (dfd_ >= 0) {
                const uint64_t off = (uint64_t) (src - base_), pg = 4096;
                const uint64_t a = off / pg * pg, b = std::min<uint64_t>((off + n + pg - 1) / pg * pg,
                                                                          (file_bytes_ + pg - 1) / pg * pg);
                thread_local uint8_t* bounce = nullptr;
                thread_local size_t cap = 0;
                if (cap < b - a) {
                    std::free(bounce);
                    bounce = (uint8_t*) std::aligned_alloc(pg, (size_t) (b - a));
                    cap = bounce ? (size_t) (b - a) : 0;
                }
                size_t done = 0;
                while (bounce != nullptr && done < b - a) {
                    const ssize_t got = pread(dfd_, bounce + done, (size_t) (b - a) - done, (off_t) (a + done));
                    if (got <= 0) break;
                    done += (size_t) got;
                }
                if (bounce != nullptr && done >= off - a + n) {
                    std::memcpy(dst, bounce + (off - a), n);
                    return;
                }
            }
            size_t done = 0;
            while (done < n) {
                const ssize_t got = pread(fd_, dst + done, n - done, (off_t) (src - base_) + (off_t) done);
                if (got <= 0) break;
                done += (size_t) got;
            }
            if (done == n) return;
        }
    }
    std::memcpy(dst, src, n);
}

void TieredExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    if (base_ == nullptr || ids == nullptr || layer < 0 || layer >= n_layers_) return;
    ++pf_epoch_;   // one epoch per layer; blob() counts a cold read as prefetched only against the current layer
    static const bool no_pf = std::getenv("STRATA_NO_COLD_PREFETCH") != nullptr;   // the A/B arm
    // The memory guard's pause-on-pressure starved decode: with the prefetch off, every cold miss faulted from the
    // SSD synchronously and the CPU pool's per-window time tripled (measured: 0% prefetched, ~20 t/s vs 97%, ~48).
    // Off by default now; STRATA_PREFETCH_PAUSE_ON_PRESSURE=1 restores the old policy.
    static const bool pause_pf = std::getenv("STRATA_PREFETCH_PAUSE_ON_PRESSURE") != nullptr;
    if (no_pf || (pause_pf && strata::platform::memory_pressure_low())) return;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const uint64_t pg = page_size();
    for (int64_t i = 0; i < k; ++i) {
        const int64_t e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        const size_t idx = (size_t) (layer * n_expert_ + e);
        const uint8_t t = tier_[idx];
        if (t == kPinned) continue;
        // A blob another VRAM tier holds is computed there in every window the CPU pool sees, so prefetching it
        // would pull 1.4 MB off the SSD per token to warm pages nothing dereferences.  Prefill never reaches here
        // (it reads through `read_into`), so this costs prefill nothing.
        if (!other_vram_.empty() && other_vram_[idx]) continue;
        // a VRAM-tier expert has no RAM copy; it needs one only once the adaptive swap has evicted it
        if (t == kVram && (res_ == nullptr || res_[idx] >= 0)) continue;
        // One read for the whole blob instead of ~340 page faults in the pool.  Asynchronous: it overlaps the
        // pool's work on the experts already in RAM.
        const uint64_t off = lay.blob_offset(layer, e);
        const uint64_t a = off / pg * pg, b = (off + lay.blob_bytes(layer) + pg - 1) / pg * pg;
        if (!pf_seen_.empty()) pf_seen_[idx] = pf_epoch_;
        ++cold_prefetches_;
        // STRATA_SYNC_PREFETCH: the old arm, madvise on the pool's own thread (it can block while the device queue
        // is full, and the pool's plan phase waited for it).  Otherwise a helper thread takes the range.
        static const bool sync_pf = std::getenv("STRATA_SYNC_PREFETCH") != nullptr;
        if (sync_pf) {
            madvise((void*) (base_ + a), (size_t) (b - a), MADV_WILLNEED);
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(pf_mu_);
            pf_q_.push_back({a, b - a});
        }
        pf_cv_.notify_one();
    }
}

}  // namespace strata::core
#endif  // !_WIN32
