#!/usr/bin/env python3
"""tune_cascade.py - find a good Strata-Cascade tier layout for THIS machine.

Portable core (Windows and Linux). It starts the server once per layout with a
modified config, runs the 32K/5K bench over HTTP, records decode / prefill /
hit-rate / tier split, and writes a CSV matrix plus a winning config.

It is deliberately small: 2-3 --host-budget-gib values around "free RAM minus
reserve", and (with a second GPU) 2 --expert-cache-device1 sizes. Enough to
find a good layout in a few minutes; a full sweep would overfit the bench
prompt (docs/TUNING.md section 4).

Usage (normally through the .ps1/.sh wrapper, which fills in --free-gib etc.):

  python tools/cascade_bench/tune_cascade.py --config strata-<model>.json
         --python .venv/Scripts/python.exe --port 8080 --max-tokens 5000
         --repeats 1 --free-gib 26 --helper-gib 11 --tag tune

`--config` is the config setup wrote (strata-<model>.json in the repo root): it
must use --tiered-experts. The wrapper detects GPUs/RAM/disk; this script only
sweeps and measures.
"""

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # tools/cascade_bench -> repo root

# The pinned tier is capped by the engine (STRATA_PIN_CAP_GIB, 8 GiB) and, with a helper + a ~16 GB primary, the
# pinned budget competes with CUDA0's cache for the prefill borrow - past ~6 GiB a 16 GB card can run out of VRAM
# in prefill (measured: 8 GiB pinned + a 6250 helper died mid-prefill with "only 0 MiB free ... shrinking the
# expert cache").  So when a helper is present the sweep stays at or below 6 GiB, and it is scaled to the machine
# rather than fixed: on a 16 GB-RAM PC the same 4/6/8 GiB would over-pin and page.  See STRATA-CASCADE.md 4.1.
PIN_CAP_GIB = 8.0
PIN_HELPER_CAP_GIB = 6.0        # measured sweet spot beside a 16 GB primary; never sweep above it
MIN_RAM_GIB = 32.0
MIN_VRAM_GIB = 12.0

# The shipped memory-guard knobs.  The tuner writes these into every guard-on run and into the winning
# config, so the result does not depend on whatever the engine's compiled defaults happen to be later.
# They apply on BOTH platforms: the engine reads them on Linux too, where trim / priority / notify are
# Windows-only and ignored (the guard there only pauses cold prefetch).  See STRATA-CASCADE.md
# ("The memory guard") and docs/TUNING.md section 1b.
GUARD_ENV = {
    "STRATA_MEM_GUARD_KEEP_FREE_MIB": "1536",   # free-RAM target to hold
    "STRATA_MEM_GUARD_RECOVER_MIB": "512",      # release/clear this far above keep_free
    "STRATA_MEM_GUARD_RELEASE_MIB": "512",      # smallest release once the guard acts
    "STRATA_MEM_GUARD_PREDICT": "0",            # keep the opt-in early trigger off (the shipped default)
    "STRATA_MEM_GUARD_MIN_MIB": "1024",         # free-RAM floor that also counts as pressure
    "STRATA_MEM_GUARD_EMERGENCY_MIB": "512",    # below this (or once the OS says low), release hard
    "STRATA_MEM_GUARD_COMMIT_MIB": "3072",      # available-commit floor
    "STRATA_MEM_GUARD_POLL_MS": "250",          # sample interval
    "STRATA_MEM_GUARD_RETRIM_MS": "1000",       # re-trim rate while low
}


def apply_guard_env(cfg, on):
    """Merge the shipped guard knobs into `cfg['env']` when the guard is kept on, drop them when it is off.

    Keeps every other env key (AMD GEMM tables, the user's own settings) untouched.  `cfg` is copied, so the
    caller's dict is not mutated."""
    out = dict(cfg)
    env = dict(out.get("env") or {})
    for k in GUARD_ENV:
        env.pop(k, None)
    if on:
        env.update(GUARD_ENV)
    out["env"] = env
    return out


def log(msg):
    print(msg, flush=True)


def in_wsl() -> bool:
    """Linux under WSL (Windows Subsystem for Linux): /proc/version says "microsoft".  The cascade cannot run there -
    its PINNED tier needs the driver to page-lock several GiB, and the WSL driver pins only about 1 GB - so the tuner
    refuses instead of starting engines that will fail at the first adaptive refill."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False


def load_config(path):
    # utf-8-sig: tolerate a BOM (some editors and PowerShell's Set-Content add one).
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def set_flag(args, flag, value):
    """Set `flag value` in the engine args list, replacing an existing pair."""
    out = list(args)
    for i, a in enumerate(out):
        if a == flag:
            if i + 1 < len(out):
                out[i + 1] = str(value)
            else:
                out.append(str(value))
            return out
    out += [flag, str(value)]
    return out


def get_flag(args, flag, default=None):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
    return default


def drop_flag(args, flag):
    out, skip = [], False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == flag:
            skip = True
            continue
        out.append(a)
    return out


def has_flag(args, flag):
    return flag in args


def add_flag(args, flag):
    """Add a bare boolean flag if it is absent (unlike set_flag, which takes a value)."""
    return list(args) if flag in args else list(args) + [flag]


def remove_flag(args, flag):
    """Remove a bare boolean flag.  Unlike drop_flag, it never eats the following argument."""
    return [a for a in args if a != flag]


def wait_ready(port, proc, timeout_s=900):
    deadline = time.time() + timeout_s
    url = "http://127.0.0.1:%d/v1/models" % port
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                obj = json.loads(r.read())
            if obj.get("data"):
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_strata():
    if os.name == "nt":
        subprocess.run(["taskkill", "/f", "/im", "strata.exe"],
                       capture_output=True, text=True)
    else:
        subprocess.run(["pkill", "-f", "strata"],
                       capture_output=True, text=True)
    time.sleep(5)


def run_bench(port, prompt_path, max_tokens, repeats):
    """Call bench_http.py (same numbers as the normal harness)."""
    py = sys.executable
    bench = HERE / "bench_http.py"
    r = subprocess.run([py, str(bench), str(port), str(prompt_path),
                        str(max_tokens), str(repeats)],
                       capture_output=True, text=True)
    decodes, prefills = [], []
    for line in r.stdout.splitlines():
        m = re.search(r"decode=([\d.]+)", line)
        if m:
            decodes.append(float(m.group(1)))
        p = re.search(r"prefill=([\d.]+)", line)
        if p and float(p.group(1)) > 1.0:      # skip the cached follow-ups (~50 t/s)
            prefills.append(float(p.group(1)))
    return (max(decodes) if decodes else 0.0,
            max(prefills) if prefills else 0.0)


def read_tiers(engine_log):
    """Pull the tier line, hit rate, the pack's blob size and the engine's own budget from the log."""
    tiers, hit, misses, blob_mib, budget_gib = "", None, "", None, None
    try:
        text = Path(engine_log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return tiers, hit, misses, blob_mib, budget_gib
    for line in text.splitlines():
        if "tiers:" in line:
            tiers = line.split("tiers:", 1)[1].strip()
            m = re.search(r"budget ([\d.]+) GiB", tiers)
            if m:
                budget_gib = float(m.group(1))
        if "decode expert cache hit rate:" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                hit = float(m.group(1))
        if "tiered decode misses:" in line:
            misses = line.split("decode misses:", 1)[1].strip()
        # "native pack: ... experts (largest blob 2.33 MB), ..."  -- model specific
        m = re.search(r"largest blob ([\d.]+) MB", line)
        if m:
            blob_mib = float(m.group(1))
        # the CUDA1 line gives bytes/expert directly, the most reliable measure
        m = re.search(r"CUDA1: (\d+) additional experts, ([\d.]+) GiB", line)
        if m:
            blob_mib = (float(m.group(2)) * 1024.0) / float(m.group(1))  # MiB per expert
    return tiers, hit, misses, blob_mib, budget_gib


def oom_reason(engine_log):
    """A short reason if the run ran out of VRAM/MEMORY, else None.

    A too-large pinned budget does not always stop the engine from starting: it can start and then die during the
    first prefill, with the engine log showing the cache being shrunk and then a failed allocation.  Or it can start,
    report a PINNED tier, and still be broken - the driver refused part of the page-lock (`cudaHostRegister refused
    chunks`), or a device buffer (the R4 hit path) could not be allocated.  All of those look like a working run to
    the bench, so every signature must be caught.  The strings come from the engine, so this covers Windows and
    Linux."""
    try:
        text = Path(engine_log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        low = line.lower()
        if "out of memory" in low or "cuda error" in low and "memory" in low:
            return line.strip()
        if "shrinking the expert cache" in low or "could not be allocated" in low:
            return line.strip()
        # the expert cache was shrunk because the slots could not be written ("only N MiB free once the slots are
        # written"): the card is out of room, even if a later line does not repeat the OOM wording
        if "once the slots are written" in low:
            return line.strip()
        # the pin could not be fully locked (Windows): the engine carried on partially pinned
        if "hostregister refused" in low or "page-locking refused" in low:
            return line.strip()
        # a device buffer failed to allocate (the R4 hit path, a verify buffer, ...)
        if "could not allocate its device buffers" in low or "could not be allocated" in low:
            return line.strip()
        # Linux: the anonymous fill behind a pin run failed
        if "the anonymous fill failed" in low:
            return line.strip()
    return None


def pin_clamped(engine_log, requested_gib, tolerance_gib=0.5):
    """If the engine pinned LESS than requested, the reason line, else None.

    `--host-budget-gib N` is a request: when the driver refuses part of the page-lock the engine either carries on
    with a smaller PINNED tier (Windows) or demotes the failed blobs to COLD (Linux) and reports the real number on
    the `tiers:` line.  Either way the achieved PINNED can be below the request, which the tuner must not mistake for
    a successful layout - the config it would write back says `N` but the machine was only ever able to hold less.
    Compares the `PINNED ... (X.XX GiB` figure on the tiers line with the requested budget."""
    if requested_gib is None:
        return None
    try:
        text = Path(engine_log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if "tiers:" not in line:
            continue
        m = re.search(r"PINNED[^(]*\(([\d.]+) GiB", line)
        if m:
            achieved = float(m.group(1))
            if achieved + tolerance_gib < requested_gib:
                return "pinned %.2f GiB of the requested %.2f GiB" % (achieved, requested_gib)
    return None


def probe_model(cfg, python_exe, port, tag, out_dir):
    """One start with the BASE config to learn the pack's real per-expert size and the engine's own auto budget.

    Returns (blob_mib_per_expert, auto_budget_gib) - either may be None if unavailable.

    A model's expert count is fixed (24,576 for Qwen3.8-Flash-Next) but the BYTES per expert depend on the quant
    (IQ2_XS < IQ3_XXS < IQ3_S < Coder), so the helper size must be derived from the measured size, never a
    constant.  The engine ALSO computes the correct pinned budget itself from MemAvailable (`--host-budget-gib
    auto`); we bracket around that number rather than guessing OS headroom from total RAM - guessing is how you
    ask a 32 GB PC to pin 21 GiB and make it page."""
    run_cfg = dict(cfg)
    run_cfg["args"] = list(cfg["args"])
    run_log = out_dir / ("tune-%s-probe.engine.log" % tag)
    run_cfg["log"] = str(run_log)
    run_cfg_path = out_dir / ("tune-%s-probe.json" % tag)
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    log("=== probing the model (one start) ===")
    stop_strata()
    proc = subprocess.Popen(
        [python_exe, "-m", "serve.server", "--engine", "strata",
         "--config", str(run_cfg_path), "--port", str(port)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_ready(port, proc):
        log("  probe NOT READY - see %s" % run_log)
        stop_strata()
        return None, None
    _, _, _, blob_mib, auto_budget = read_tiers(run_log)
    stop_strata()
    if blob_mib:
        log("  per-expert blob: %.2f MiB (model specific)" % blob_mib)
    if auto_budget:
        log("  engine auto budget: %.2f GiB (from MemAvailable; the sweep brackets this)" % auto_budget)
    return blob_mib, auto_budget


def sweep_budgets(has_helper, center, avail_gib, primary_gib, reserve):
    """The --host-budget-gib values to sweep, scaled to the machine (not a fixed 4/6/8).

    With a helper the pinned budget shares the machine with CUDA0's cache, so it is capped at PIN_HELPER_CAP_GIB
    (6 GiB, the measured sweet spot on a 16 GB primary) and also bounded by what this PC can actually pin: the real
    RAM the OS reports free minus the reserve (the engine's own `auto` rule - src/core/tiered_source.cpp), and a
    fraction of the primary card.  Without a helper the engine's own `auto` value is the center and the sweep brackets
    it by +/-2 GiB.  The result is always non-empty and >= 2 GiB, so a small PC gets small, feasible budgets instead
    of only 4/6/8."""
    if has_helper:
        # The pinned budget shares the machine with CUDA0's expert cache, so cap it at PIN_HELPER_CAP_GIB (6 GiB,
        # the measured sweet spot on a 16 GB primary), the real free RAM minus the reserve (upstream's auto rule),
        # and ~45% of the primary card.  The three points scale with the cap but stay distinct down to ~2 GiB, so
        # even a tiny PC gets a real (if small) sweep instead of a single value.
        ram_cap = max(0.0, avail_gib - reserve)
        vram_cap = max(2.0, primary_gib * 0.45) if primary_gib else PIN_HELPER_CAP_GIB
        cap = max(2.0, min(PIN_HELPER_CAP_GIB, ram_cap, vram_cap))
        step = min(2.0, max(0.5, round(cap / 3.0, 1)))     # 2 GiB on a big box, less when the cap is small
        vals = {round(cap, 1),
                max(2.0, round(cap - step, 1)),
                max(2.0, round(cap - 2 * step, 1))}
        return sorted(vals)
    return sorted({round(max(2.0, center - 2), 1), round(center, 1), round(center + 2, 1)})


def apply_helpers(eargs, device1, device2=0, device3=0):
    """Set (or drop) the CUDA1..3 helper sizes.  `device1` is the total for the second card; if a third or
    fourth card exists, they are folded into the same helper count via device2/device3 (upstream supports
    --expert-cache-device1..3)."""
    for flag, val in (("--expert-cache-device1", device1),
                      ("--expert-cache-device2", device2),
                      ("--expert-cache-device3", device3)):
        if val and val > 0:
            eargs = set_flag(eargs, flag, int(val))
        else:
            eargs = drop_flag(eargs, flag)
    return eargs


def one_layout(cfg, layout, python_exe, port, max_tokens, repeats, tag, i, total, out_dir, guard_flag=None):
    log("\n[%d/%d] %s" % (i, total, layout))
    eargs = list(cfg["args"])
    eargs = set_flag(eargs, "--host-budget-gib", layout["host_budget_gib"])
    eargs = set_flag(eargs, "--host-reserve-gib", layout["host_reserve_gib"])
    eargs = set_flag(eargs, "--vram-reserve-mib", layout["vram_reserve_mib"])
    eargs = apply_helpers(eargs, layout.get("helper_size") or 0,
                          layout.get("helper_size_2") or 0, layout.get("helper_size_3") or 0)
    # With no helper the adaptive tier has nothing to protect and hurts (measured 35.4 -> 25.1 t/s on one card);
    # with a helper it is the shipped fast path (Phase C).  So it is on only when a helper is present.
    has_helper = bool(layout.get("helper_size") or layout.get("helper_size_2") or layout.get("helper_size_3"))
    if has_helper:
        eargs = set_flag(eargs, "--adapt-every", layout.get("adapt_every", 8))
        eargs = set_flag(eargs, "--adapt-swaps", layout.get("adapt_swaps", 32))
    else:
        eargs = set_flag(eargs, "--adapt-every", 0)
        eargs = set_flag(eargs, "--adapt-swaps", 0)
    # The prefill-chunk A/B sets a ceiling for `auto` (the prompt path still picks the largest chunk whose device
    # buffers fit the expert cache).  A bigger chunk reads each cold expert once per chunk instead of once per 8K
    # chunk, so a long prompt reads far fewer bytes.
    if layout.get("prefill"):
        eargs = set_flag(eargs, "--prefill", layout["prefill"])
    # The guard A/B: force the guard on/off for this run.  Always drop both the canonical and the legacy
    # spelling first, so a base config that already has one cannot leak into the "off" arm.
    if guard_flag is not None:
        eargs = remove_flag(eargs, "--windows-memory-guard")
        eargs = remove_flag(eargs, guard_flag)
        if layout.get("guard"):
            eargs = add_flag(eargs, guard_flag)

    run_cfg = dict(cfg)
    run_cfg["args"] = eargs
    if guard_flag is not None:
        # Force the shipped guard knobs for this arm, so the A/B measures the shipped guard rather than
        # whatever the engine's compiled defaults are - the same numbers on Linux and Windows.
        run_cfg = apply_guard_env(run_cfg, bool(layout.get("guard")))
    run_log = out_dir / ("tune-%s-l%d.engine.log" % (tag, i))
    run_cfg["log"] = str(run_log)
    run_cfg_path = out_dir / ("tune-%s-l%d.json" % (tag, i))
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    stop_strata()
    proc = subprocess.Popen(
        [python_exe, "-m", "serve.server", "--engine", "strata",
         "--config", str(run_cfg_path), "--port", str(port)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not wait_ready(port, proc):
        log("  NOT READY - see %s" % run_log)
        stop_strata()
        return None

    decode, prefill = run_bench(port, HERE / "prompt32k.txt", max_tokens, repeats)
    stop_strata()
    tiers, hit, misses, _, _ = read_tiers(run_log)
    oom = oom_reason(run_log)
    clamped = None
    if not oom and (layout.get("helper_size") or layout.get("helper_size_2") or layout.get("helper_size_3")):
        # the engine reports how much it ACTUALLY pinned on the tiers line; with a helper it must match the request
        clamped = pin_clamped(run_log, float(layout["host_budget_gib"]))
    row = {
        "host_budget_gib": layout["host_budget_gib"],
        "host_reserve_gib": layout["host_reserve_gib"],
        "vram_reserve_mib": layout["vram_reserve_mib"],
        "helper_size": layout.get("helper_size") or 0,
        "helper_size_2": layout.get("helper_size_2") or 0,
        "helper_size_3": layout.get("helper_size_3") or 0,
        "decode_t_s": round(decode, 1),
        "prefill_t_s": round(prefill, 1),
        "hit_pct": hit,
        "tiers": tiers,
        "misses": misses,
        "config": str(run_cfg_path),
    }
    if guard_flag is not None:
        row["memory_guard"] = 1 if layout.get("guard") else 0
    if oom:
        # the engine started but could not hold this layout; a 0 t/s row is NOT a slow layout, and must never
        # win the sweep.  Clear its scores and say why.
        row["decode_t_s"] = 0.0
        row["prefill_t_s"] = 0.0
        row["oom"] = oom
        log("  OOM during prefill (%s) - the pinned budget is too large for this card; skipped"
            % oom)
        log("  decode=0 t/s  prefill=0 t/s  hit=%s%%" % hit)
        return row
    if clamped:
        # the engine pinned less than asked (the driver refused part of the lock, or demoted it): the config this
        # row would write back claims a budget the machine did not actually hold. Never let it win.
        row["decode_t_s"] = 0.0
        row["prefill_t_s"] = 0.0
        row["clamped"] = clamped
        log("  pinned budget not reached (%s) - the driver refused part of it; skipped" % clamped)
        log("  decode=0 t/s  prefill=0 t/s  hit=%s%%" % hit)
        return row
    log("  decode=%s t/s  prefill=%s t/s  hit=%s%%" % (row["decode_t_s"], row["prefill_t_s"], hit))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="base config json from setup (strata-<model>.json); must use --tiered-experts")
    ap.add_argument("--python", default=sys.executable, help="the .venv python that runs serve.server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=1, help="bench repeats per layout (1 is enough; 2-3 for confidence)")
    ap.add_argument("--free-gib", type=float, default=None,
                    help="physical RAM available to the model; wrapper sets this")
    ap.add_argument("--available-gib", type=float, default=None,
                    help="the OS's real free RAM at boot (MemAvailable / ullAvailPhys); the helper budget is "
                         "sized from this minus the reserve (the engine's own auto rule). Wrapper sets this")
    ap.add_argument("--helper-gib", default=None,
                    help="usable VRAM of the SECOND/THIRD/FOURTH gpus as a comma list (GiB); the core converts "
                         "each to experts with the measured per-expert size, so it works for any model size")
    ap.add_argument("--primary-gib", type=float, default=None,
                    help="the main GPU's total VRAM (GiB); read from the config when absent. Used only for the "
                         "low-memory warning - the sweep still runs")
    ap.add_argument("--helper-size", type=int, default=None,
                    help="second-GPU expert count to test (0 = no helper); overrides --helper-gib")
    ap.add_argument("--helper-size-2", type=int, default=None, help="third-GPU expert count")
    ap.add_argument("--helper-size-3", type=int, default=None, help="fourth-GPU expert count")
    ap.add_argument("--reserve-gib", type=float, default=8.0,
                    help="RAM kept free beside the pin (the engine's own --host-reserve-gib default is 8)")
    ap.add_argument("--out", default=None,
                    help="where the CSV, winning config and engine logs go (default: tools/cascade_bench)")
    ap.add_argument("--tag", default="tune")
    ap.add_argument("--guard", action="store_true",
                    help="after the sweep, A/B the winning layout with the memory guard on and off, and keep "
                         "it if it does not cost decode throughput. The shipped knobs (keep 1536 / "
                         "recover 512 / release 512 / band 512 MiB) are written into the run and the winning "
                         "config, on Windows and Linux")
    ap.add_argument("--guard-flag", default="--memory-guard",
                    help="the engine flag the guard A/B toggles (default --memory-guard; the alias "
                         "--windows-memory-guard always works and is dropped first)")
    ap.add_argument("--prefill-ceilings", default="auto,16384,auto:16384",
                    help="comma list of --prefill values to A/B on the winning layout (a bigger chunk reads each "
                         "cold expert once per chunk, so a long prompt reads fewer bytes; measured 605 -> 900 t/s "
                         "on a 32 GB two-card cascade). Both the forced 16384 and the ceiling form auto:16384 are "
                         "tried: auto:16384 only raises the ceiling and the engine picks the largest chunk whose "
                         "buffers fit (15616 here), while the forced 16384 measured faster. The chunk's host "
                         "staging buffers scale with it, so on a big-RAM PC add auto:32768; empty string skips the step")
    args = ap.parse_args()

    if in_wsl():
        log("ERROR: the cascade cannot run under WSL: its PINNED tier needs the driver to page-lock several GiB, and"
            " the WSL driver pins only about 1 GB, so the adaptive refill fails at the first prompt.")
        log("       Use native Windows/Linux, or upstream's low-RAM mode on one card (--low-ram resident).")
        return 2

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path) if (ROOT / cfg_path).exists() else (HERE / cfg_path)
    if not cfg_path.exists():
        log("ERROR: config not found: %s" % cfg_path)
        return 2
    try:
        cfg = load_config(cfg_path)
    except (OSError, ValueError) as e:
        log("ERROR: cannot read %s as JSON: %s" % (cfg_path, e))
        return 2

    eargs = cfg.get("args", [])
    if "--tiered-experts" not in eargs:
        log("ERROR: the config does not use --tiered-experts; nothing to tune.")
        return 2
    if "--expert-profile" not in eargs:
        log("ERROR: --tiered-experts needs --expert-profile. See docs/TUNING.md section 1.")
        return 2

    # The cascade's CUDA0 is its primary expert cache; upstream's #1352 "faster card last" would reorder the cards
    # and shrink it (measured: prefill 907 -> 456 t/s).  Pin the order, exactly as setup's write_config does, so the
    # sweep and the winning config keep the intended primary card even when the base config predates the key.
    cfg["gpu_order"] = "as_given"

    # Normalize the memory-guard knobs to the shipped values whenever the base config has the guard
    # on, so the sweep and the winning config use them too - the same numbers on Linux and Windows.
    cfg = apply_guard_env(cfg, ("--memory-guard" in eargs) or ("--windows-memory-guard" in eargs))
    eargs = cfg.get("args", [])

    out_dir = Path(args.out) if args.out else HERE
    out_dir.mkdir(parents=True, exist_ok=True)

    free_gib = args.free_gib
    if free_gib is None:
        log("WARNING: --free-gib not given; assuming 24 GiB available for the model.")
        free_gib = 24.0
    reserve = args.reserve_gib

    # A low-memory warning, never a stop: the cascade targets ~32 GB RAM with a >=12 GB card, but a smaller model or
    # a lucky layout can still work, so the sweep runs - it just says plainly that results may fail.
    primary_gib = args.primary_gib
    if primary_gib is None:
        # fall back to the largest card in the config's "gpu" list, as on some setups the wrapper is not used
        try:
            primary_gib = max((float(g) for g in cfg.get("gpu", []) if isinstance(g, (int, float))), default=None)
        except (TypeError, ValueError):
            primary_gib = None
    # `free_gib` is what the wrapper measured as available to the model (physical RAM minus ~3 GiB for the OS), so
    # compare it against the target minus that same headroom - never the total.
    avail_floor_gib = MIN_RAM_GIB - 3.0
    low_ram = free_gib < avail_floor_gib
    low_vram = primary_gib is not None and primary_gib < MIN_VRAM_GIB
    if low_ram or low_vram:
        log("=" * 78)
        log("WARNING: INSUFFICIENT MEMORY - RESULTS MAY FAIL")
        if low_ram:
            log("  RAM: about %.0f GiB would be available to the model; the cascade targets %d GB of system RAM."
                % (free_gib, int(MIN_RAM_GIB)))
            log("       It will still run, but expect paging or slow sweeps on the larger pinned budgets.")
        if low_vram:
            log("  VRAM: the main card has about %.0f GB; the cascade targets %d GB or more.  With little VRAM"
                % (primary_gib, int(MIN_VRAM_GIB)))
            log("       there is little room for CUDA0's expert cache, and prefill can run out of VRAM.")
        log("  Continuing anyway.  Try a smaller model (Q2_0 / IQ2_XS) or fewer layouts if it is too slow.")
        log("=" * 78)
        log("")

    # ALWAYS probe once: it gives the model's real per-expert size AND the engine's own pinned budget, which the
    # engine computes correctly from MemAvailable. Bracketing that is far safer than guessing OS headroom from
    # total RAM - a wrong guess makes a 32 GB PC try to pin 21 GiB and page.
    blob_mib, auto_budget = probe_model(cfg, args.python, args.port, args.tag, out_dir)

    if auto_budget:
        center = auto_budget
        log("  the engine's auto pinned budget: %.1f GiB" % center)
    else:
        # fallback: the RAM available to the model minus reserve (used only if the probe could not report one)
        center = max(2.0, free_gib - reserve)
        log("  a RAM-derived pinned estimate: %.1f GiB" % center)

    # The model's expert count is fixed, but the BYTES per expert depend on the quant (IQ2_XS < IQ3_XXS <
    # IQ3_S < Coder), so the helper size is derived from the MEASURED per-expert size, never a constant.
    # --helper-gib is a comma list of the 2nd/3rd/4th cards' usable VRAM (GiB); --expert-cache-device1..3 take
    # the corresponding expert counts, all scaled by the model's measured MiB per expert.
    helper_base = args.helper_size
    hb2 = args.helper_size_2
    hb3 = args.helper_size_3
    if helper_base is None and args.helper_gib:
        if not blob_mib:
            log("  WARNING: could not measure the per-expert size; skipping the helper sweep.")
        else:
            gibs = [float(x) for x in str(args.helper_gib).split(",") if x.strip()]
            counts = [int((g * 1024.0) / blob_mib) for g in gibs[:3]]
            helper_base = counts[0] if len(counts) > 0 else None
            hb2 = counts[1] if len(counts) > 1 else None
            hb3 = counts[2] if len(counts) > 2 else None
            log("  helper sizes from VRAM (%.2f MiB/expert): %s experts"
                % (blob_mib, ", ".join(str(c) for c in counts)))
    has_helper = bool(helper_base and helper_base > 0)

    # The pinned budget to sweep, scaled to this machine: with a helper it stays <= 6 GiB and is bounded by the real
    # free RAM minus the reserve and the primary card (a fixed 4/6/8 would over-pin a 16 GB-RAM PC); without a helper
    # it brackets the engine's own auto value by +/-2 GiB.  See sweep_budgets().
    avail_gib = args.available_gib if args.available_gib is not None else free_gib
    budgets = sweep_budgets(has_helper, center, avail_gib, primary_gib, reserve)
    if has_helper:
        log("  helper present: sweeping pinned budgets %s GiB (scaled to this PC; the auto value %.1f over-commits CUDA0)"
            % (", ".join("%g" % b for b in budgets), center))
    else:
        log("  one card: bounding the auto value by +/-2 GiB -> %s GiB"
            % ", ".join("%g" % b for b in budgets))

    helper_sets = []
    if helper_base and helper_base > 0:
        for f in (0.75, 1.0, 1.1):    # 1.1 catches the over-budget edge
            helper_sets.append({
                "helper_size": int(helper_base * f),
                "helper_size_2": int(hb2 * f) if hb2 else 0,
                "helper_size_3": int(hb3 * f) if hb3 else 0,
            })
    if not helper_sets:
        helper_sets = [{"helper_size": 0, "helper_size_2": 0, "helper_size_3": 0}]

    vram_reserve = int(get_flag(eargs, "--vram-reserve-mib", "1000") or 1000)

    layouts = []
    for b in budgets:
        for h in helper_sets:
            layouts.append({
                "host_budget_gib": b,
                "host_reserve_gib": reserve,
                "vram_reserve_mib": vram_reserve,
                "helper_size": h["helper_size"],
                "helper_size_2": h["helper_size_2"],
                "helper_size_3": h["helper_size_3"],
            })

    log("Strata-Cascade layout tuner")
    log("  base config : %s" % cfg_path)
    log("  free RAM    : %.1f GiB (reserve %.1f)" % (free_gib, reserve))
    if blob_mib:
        log("  per-expert  : %.2f MiB (measured; helper sizes scale with it)" % blob_mib)
    log("  layouts     : %d" % len(layouts))
    log("  NOTE: the bench prompt is synthetic and overfits - confirm the winner on your real workload (docs/TUNING.md).")

    rows = []
    for i, layout in enumerate(layouts, 1):
        row = one_layout(cfg, layout, args.python, args.port, args.max_tokens,
                         args.repeats, args.tag, i, len(layouts), out_dir)
        if row:
            rows.append(row)

    if not rows:
        log("\nNo layout completed. Check the engine logs in %s." % out_dir)
        return 1

    usable = [r for r in rows if r.get("decode_t_s", 0) > 0]
    if not usable:
        log("\nEvery layout failed (out of memory or not ready). Lower --host-budget-gib and try again; see the")
        log("engine logs in %s. On a 16 GB primary with a helper, 6 GiB pinned is a good starting point." % out_dir)
        return 1
    best = max(usable, key=lambda r: r["decode_t_s"])

    # The winning layout, reused by the guard and prefill A/B steps below.
    best_layout = {
        "host_budget_gib": best["host_budget_gib"],
        "host_reserve_gib": best["host_reserve_gib"],
        "vram_reserve_mib": best["vram_reserve_mib"],
        "helper_size": best.get("helper_size") or 0,
        "helper_size_2": best.get("helper_size_2") or 0,
        "helper_size_3": best.get("helper_size_3") or 0,
    }

    # --- the guard step: A/B the winning layout with the memory guard on and off -----------------------
    # The guard trades a little engine throughput for a responsive system: it pauses cold prefetch under
    # pressure and (on Windows) trims its own working set.  On Linux only the prefetch pause applies, so
    # the delta should be near zero - this A/B proves it on THIS machine instead of trusting the bench.
    guard_rows = []
    guard_helps = None
    if args.guard:
        log("\n==== guard A/B on the winning layout ====")
        for j, use_guard in enumerate((False, True), 1):
            gl = dict(best_layout)
            gl["guard"] = use_guard
            row = one_layout(cfg, gl, args.python, args.port, args.max_tokens, args.repeats,
                             args.tag + "-guard", j, 2, out_dir, guard_flag=args.guard_flag)
            if row:
                guard_rows.append(row)
        g_off = next((r for r in guard_rows if not r.get("memory_guard")), None)
        g_on = next((r for r in guard_rows if r.get("memory_guard")), None)
        if g_off and g_on:
            d_dec = g_on["decode_t_s"] - g_off["decode_t_s"]
            d_pre = g_on["prefill_t_s"] - g_off["prefill_t_s"]
            # Keep the guard unless it clearly costs decode.  One run per arm has a few percent of noise (two
            # identical runs differed by ~1 t/s decode, ~10% prefill here), so a 1 t/s / 2% band stops a noisy
            # single run from dropping a guard that is really free.  See docs/TUNING.md section 1b.
            band = max(1.0, 0.02 * g_off["decode_t_s"])
            guard_helps = d_dec >= -band
            log("  guard off: decode %s t/s  prefill %s t/s  hit %s%%"
                % (g_off["decode_t_s"], g_off["prefill_t_s"], g_off["hit_pct"]))
            log("  guard on : decode %s t/s  prefill %s t/s  hit %s%%"
                % (g_on["decode_t_s"], g_on["prefill_t_s"], g_on["hit_pct"]))
            log("  delta    : decode %+.1f t/s  prefill %+.1f t/s -> keep %s in the winning config"
                % (d_dec, d_pre, "the guard" if guard_helps else "no guard"))
            log("             (a within %.1f t/s decode is run-to-run noise)" % band)
            log("  NOTE: on Linux the guard only pauses cold prefetch, so a near-zero delta is expected;")
            log("        keep it for the responsive-system headroom.")
        else:
            log("  guard A/B did not complete; see the engine logs in %s" % out_dir)

    # --- the prefill step: A/B the chunk ceiling on the winning layout ---------------------------------
    # A bigger prompt chunk reads each cold expert once per chunk instead of once per 8K chunk, so a long prompt
    # reads far fewer bytes (measured on a 32 GB two-card cascade: 129.4 -> 71.2 GB of cold reads, 605 -> 900 t/s).
    # The chunk's host staging buffers scale with it, so on a low-RAM PC the bigger ceiling can lose - the bench
    # decides on THIS machine.  `auto:N` only raises the ceiling; the engine still picks the largest chunk whose
    # device buffers fit the expert cache, so a short prompt is unaffected.
    prefill_rows = []
    prefill_helps = None
    ceilings = [c.strip() for c in (args.prefill_ceilings or "").split(",") if c.strip()]
    if ceilings:
        log("\n==== prefill-chunk A/B on the winning layout ====")
        gflag = args.guard_flag if args.guard else None
        for j, ceiling in enumerate(ceilings, 1):
            pl = dict(best_layout)
            pl["prefill"] = ceiling
            if args.guard and guard_helps is not None:
                pl["guard"] = guard_helps
            row = one_layout(cfg, pl, args.python, args.port, args.max_tokens, args.repeats,
                             args.tag + "-prefill", j, len(ceilings), out_dir, guard_flag=gflag)
            if row:
                row["prefill_ceiling"] = ceiling
                prefill_rows.append(row)
        ok_rows = [r for r in prefill_rows if r.get("decode_t_s", 0) > 0]
        if ok_rows:
            for r in ok_rows:
                log("  --prefill %s: decode %s t/s  prefill %s t/s  hit %s%%"
                    % (r["prefill_ceiling"], r["decode_t_s"], r["prefill_t_s"], r["hit_pct"]))
            pbest = max(ok_rows, key=lambda r: r["prefill_t_s"])
            prefill_helps = pbest["prefill_ceiling"]
            log("  -> keep --prefill %s in the winning config" % prefill_helps)
        else:
            log("  prefill A/B did not complete; see the engine logs in %s" % out_dir)

    stamp = _dt.date.today().isoformat()
    csv_path = out_dir / ("tune-%s.csv" % stamp)
    all_rows = rows + guard_rows + prefill_rows
    # union of keys across rows, so an `oom` row's field appears even if the first row lacks it
    keys = []
    for row in all_rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in all_rows:
            f.write(",".join('"%s"' % str(row.get(k, "")).replace('"', "'") for k in keys) + "\n")

    win_cfg = dict(cfg)
    win_cfg["args"] = set_flag(set_flag(set_flag(list(cfg["args"]),
                            "--host-budget-gib", best["host_budget_gib"]),
                            "--host-reserve-gib", best["host_reserve_gib"]),
                            "--vram-reserve-mib", best["vram_reserve_mib"])
    win_cfg["args"] = apply_helpers(win_cfg["args"], best["helper_size"],
                                    best.get("helper_size_2") or 0, best.get("helper_size_3") or 0)
    # carry the adaptive flags that actually won: on with a helper, off without (one_layout's rule)
    best_has_helper = bool(best.get("helper_size") or best.get("helper_size_2") or best.get("helper_size_3"))
    win_cfg["args"] = set_flag(win_cfg["args"], "--adapt-every", 8 if best_has_helper else 0)
    win_cfg["args"] = set_flag(win_cfg["args"], "--adapt-swaps", 32 if best_has_helper else 0)
    # the guard A/B's verdict (or, when --guard was not given, the base config's own choice, untouched)
    if args.guard and guard_helps is not None:
        win_cfg["args"] = remove_flag(win_cfg["args"], "--windows-memory-guard")
        win_cfg["args"] = (add_flag(win_cfg["args"], args.guard_flag) if guard_helps
                           else remove_flag(win_cfg["args"], args.guard_flag))
    # the prefill A/B's verdict (or, when it did not run, the base config's own choice, untouched)
    if prefill_helps:
        win_cfg["args"] = set_flag(win_cfg["args"], "--prefill", prefill_helps)
    # Ship the guard knobs whenever the guard is on in the final config (the A/B's verdict, or the
    # base config's own choice when --guard was not given) and drop them when it is off - so the config is
    # self-describing and portable instead of relying on the engine's compiled defaults.
    guard_on = (args.guard_flag in win_cfg["args"]) or ("--windows-memory-guard" in win_cfg["args"])
    win_cfg = apply_guard_env(win_cfg, guard_on)
    win_path = out_dir / ("tune-%s-best.json" % stamp)
    win_path.write_text(json.dumps(win_cfg, indent=2), encoding="utf-8")

    helper_str = ""
    if best["helper_size"]:
        helper_str = " --expert-cache-device1 %d" % best["helper_size"]
        if best.get("helper_size_2"):
            helper_str += " --expert-cache-device2 %d" % best["helper_size_2"]
        if best.get("helper_size_3"):
            helper_str += " --expert-cache-device3 %d" % best["helper_size_3"]

    log("\n==== best layout ====")
    log("  --host-budget-gib %s --host-reserve-gib %s --vram-reserve-mib %s%s"
        % (best["host_budget_gib"], best["host_reserve_gib"], best["vram_reserve_mib"], helper_str))
    if best_has_helper:
        log("  --adapt-every 8 --adapt-swaps 32   (helper present: the adaptive tier helps)")
    else:
        log("  --adapt-every 0                    (one card: the adaptive tier hurts)")
    log("  decode %s t/s, prefill %s t/s, hit %s%%" % (best["decode_t_s"], best["prefill_t_s"], best["hit_pct"]))
    if prefill_helps:
        log("  --prefill %s   (the chunk A/B's winner)" % prefill_helps)
    if args.guard and guard_helps is not None:
        log("  memory guard: %s%s" % ("on" if guard_helps else "off",
            "" if guard_helps else " (the A/B measured a decode cost on this machine)"))
        if guard_helps:
            log("  guard knobs: keep %s / recover %s / release %s MiB  (written to the config)"
                % (GUARD_ENV["STRATA_MEM_GUARD_KEEP_FREE_MIB"], GUARD_ENV["STRATA_MEM_GUARD_RECOVER_MIB"],
                   GUARD_ENV["STRATA_MEM_GUARD_RELEASE_MIB"]))
    if any(r.get("oom") for r in rows):
        log("  note: at least one layout ran out of memory in prefill - see the CSV's oom column")
    log("  CSV   : %s" % csv_path)
    log("  config: %s" % win_path)
    log("  To use it: copy %s over your %s (or pass it as --config next time)." % (win_path.name, cfg_path.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
