# AGENTS.md

Version 0.2.6, engine 0.1.51. The public model families are DeepSeekV4.1-Flash-1017,
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. DeepSeek V4.1-Flash runs in the app and in the Terminal
(`strata launch --family deepseek41 --model ist-das-2.00bpw`; sets in
[docs/MODELS.md](docs/MODELS.md#deepseek-v41-flash)). Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
source-backed claims. Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

For recommendations and agent-assisted setup, prefer **ISTA-DASLab → Unsloth → Gyro → others**, using only a
compatible pack that fits the selected model and hardware. ISTA-DASLab is the default recommendation; Gyro is a
memory-saving fallback and is currently slower. Honor an explicit user choice. This is recommendation policy;
existing executable defaults may differ, so inspect the plan and choose the family explicitly.

Strata has three parts:

- **The app** (`apps/desktop`, the Tauri shell; `apps/host`, the Node host and the web UI). It is for people who do
  not want a terminal. One installer per system ([latest release](https://github.com/Niko1221/Strata/releases/latest)):
  `Strata-0.2.6-windows-x64-setup.exe`, `Strata-0.2.6-linux-x86_64.AppImage`, `Strata-0.2.6-macos-arm64.dmg`.
  The Windows and Linux installers carry no engine: the app downloads the engine for the graphics card on first
  start, or installs a file the user already has. The Mac dmg keeps its small Metal engine inside.
- **The Terminal** (`strata.py`, `strata_cli/`, `strata_control/`): the `strata` command. It sets up a model, serves
  it and watches it. Agents use this part.
- **The engine and the server** (`src/`, `include/`, `metal/`, `sycl/` for the C++/CUDA/HIP/SYCL/Metal engine;
  `serve/` for the Python server with an OpenAI- and Anthropic-compatible API and a web page). `setup.py`, started by
  `START-HERE.bat` or `setup.sh`, is setup from a source checkout, for developers. The release paths are the
  installer and `strata launch`.

## This checkout is Strata-Cascade (a fork of upstream Strata)

Everything here is **upstream Strata** plus multiple additions, built around the **cascading expert source** (`--tiered-experts`),
which sorts the experts across VRAM, a pinned RAM budget and the SSD so a PC whose RAM cannot hold every expert
still runs the model, and combines that with the CUDA1..3 helper. The port lives on branch `cascade`
(`forestautomation/Strata-Cascade`).

- **Do not "fix" the cascade away.** Upstream used to hard-reject the helper beside a RAM tier; 0.1.45 lifts that
  rejection (see `docs/strata-cascade-implementation-guide.md` section 0), but upstream still has no three-tier
  `--tiered-experts` source and no SSD cold tier, so its `--resident-experts` mode is not a replacement. The port
  is the point of this fork.
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

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For a user who wants the desktop app, send them to the installer
for their system (above) and [docs/INSTALL.md](docs/INSTALL.md). For an agent that has a shell, use the Terminal; the
short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh                  # Linux, Mac
   irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --family qwen --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --family qwen --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--speed faster|full` (DeepSeek), `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). Source setup (`START-HERE.bat`, `setup.sh`) is the fallback for a developer or for a system with no release package; it is
documented in [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md). Those wrappers exist in a source checkout, not in the
released app. Do not tell an app user to find a `.bat` file.
Source setup does not make an unsupported model or backend compatible. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

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
- **WSL: never use the cascade.** The WSL driver pins only ~1 GB, so the PINNED tier fails at the first adaptive
  refill. Setup detects WSL (`is_wsl()`), does not select or ask for the cascade, downgrades an explicit
  `--low-ram tiered` to the usual low-RAM mode, and rewrites an existing cascade config to upstream's
  `--resident-experts`/`--mmap-experts` on its next start. The cascade tuner refuses on WSL. Do not pass
  `--low-ram tiered` or `--tune-cascade` on WSL.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](packaging/README.md) and [packaging/UPDATES.md](packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `tests/terminal` (`python -m pytest tests/terminal`). The app's UI
  tests run with `npm test` in `apps/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- Never submit measurements, logs or reproduction steps you did not produce. Mark a finding from reading code as
  unconfirmed static analysis.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
