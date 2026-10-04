# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

## This checkout is Strata-Cascade (a fork of upstream Strata)

Everything here is **upstream Strata** plus multiple additions, built around the **cascading expert source** (`--tiered-experts`),
which sorts the experts across VRAM, a pinned RAM budget and the SSD so a PC whose RAM cannot hold every expert
still runs the model, and combines that with the CUDA1..3 helper (a combination upstream rejects). The port lives
on branch `cascade` as a single commit on top of upstream.

- **Do not "fix" the cascade away.** Upstream's own `--resident-cpu-experts` is not a replacement: it hard-rejects
  the helper. The port is the point of this fork.
- The port touches 8 engine surfaces: `src/core/tiered_source.cpp`, `expert_source.hpp`, `remote_experts.hpp`,
  `generate.cpp`, `prefill.cpp`, `expert_cache.cpp`, `serve/server.py`, `CMakeLists.txt` (+ `native_mmap_test.cpp`).
  What each does and where: [`docs/strata-cascade-implementation-guide.md`](docs/strata-cascade-implementation-guide.md).
- Plus two non-engine surfaces so a fresh clone actually runs the cascade: **`setup.py`** (`--low-ram tiered`, and
  the default on a low-RAM PC with a second GPU) and **`tools/cascade_bench/`** (the shipped layout tuner). Both
  degrade to upstream behaviour when the ported engine is absent; see `docs/strata-cascade-implementation-guide.md`
  section 8.
- Results and measured numbers: [`STRATA-CASCADE.md`](STRATA-CASCADE.md). Tuning for a user's own
  RAM/CPU/GPU: [`docs/TUNING.md`](docs/TUNING.md) (+ `tools/cascade_bench/tune_cascade.py`).
- The optional `--memory-guard` (alias `--windows-memory-guard`; for a page file on a slow disk): details and measured
  numbers in [`STRATA-CASCADE.md`](STRATA-CASCADE.md) ("The memory guard"); the knobs and testing in
  [`docs/TUNING.md`](docs/TUNING.md) section 1b; the test harness in `tools/cascade_bench/guard/` (`guard-test.ps1`,
  `mem-hog.py`, `guard-summary.ps1`). Off by default. On Windows it trims the working set + lowers memory priority;
  on Linux it only pauses cold prefetch (implemented, unmeasured); the harness is Windows-only.
- Re-landing on a new upstream: **rebase** the port commit on `origin/main` (not merge) so conflicts stay isolated.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

**This fork adds the cascade, and setup turns it on by itself.** On a PC whose RAM cannot hold the model (the
existing low-RAM test) *and* that has a second usable GPU, `setup.py` selects `--tiered-experts` automatically and
writes a `--host-budget-gib` sized from the real free RAM at start (the engine's own `MemAvailable`/`ullAvailPhys`
minus the 8 GiB reserve), capped at the measured 6 GiB beside a helper, floored at 2 GiB, and bounded by ~45% of the
primary card (`auto` on one card), plus a `--expert-cache-device1..3` per helper card. On **one** card (RAM still too
small) setup **asks** and recommends the cascade (default **yes**); a PC whose RAM holds the whole model is never
offered it. You do not need to pass anything; just run setup as usual. Notes for an agent:

- **Force it / turn it off:** `--low-ram tiered` forces the cascade; `--low-ram resident` or `--low-ram off` turns
  it off. On an engine that is not this fork's, `--low-ram tiered` falls back to the usual low-RAM mode with a
  warning (detection is `strata --help` for `--tiered-experts`).
- **The second GPU is what makes it much faster.** One card still runs the cascade (prefill ~2x, decode about the
  same as upstream's resident mode) but the big win needs the helper; a one-card PC that says no keeps upstream's
  resident mode.
- **After install, offer to tune it:** setup now does this itself - `START-HERE.bat --tune-cascade` (or
  `./setup.sh --tune-cascade`) runs `tools/cascade_bench/tune_cascade.py` (wrappers `tune-cascade.ps1` /
  `tune-cascade.sh`) and copies the winning layout over the config; an interactive install also offers it. It sweeps
  a few layouts on the 32K/5K bench (at the config's context and KV quant - the measured runs use 131072 /
  `--kv int8 --kv-resident 20480`). See [docs/TUNING.md](docs/TUNING.md); tell the user the bundled bench prompt
  overfits, so confirm on their real work.
- **Verify it is active:** the engine log shows a `tiers: VRAM ... | PINNED ... | COLD ...` line. No `tiers:` line
  means it did not settle - check the flags.
- **Compatibility is upstream's:** Windows and Linux, NVIDIA-only or AMD-only. The cascade's Linux path is
  implemented but unmeasured; its AMD path is unmeasured too ([docs/TUNING.md](docs/TUNING.md) section 6).

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
