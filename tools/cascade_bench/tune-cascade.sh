#!/usr/bin/env bash
# tune-cascade.sh - find a good Strata-Cascade tier layout for THIS Linux PC.
# Detects the GPUs, physical RAM and the pack's disk, then calls the portable
# core (tune_cascade.py), which sweeps a few layouts on the 32K/5K bench.
#
# NOTE: the Linux path of the cascade is implemented but not yet measured. If you
# run this, please report your numbers (docs/TUNING.md section 5).
#
# Usage:
#   tools/cascade_bench/tune-cascade.sh --config strata-<model>.json
#   tools/cascade_bench/tune-cascade.sh --config <path> --port 8080 --repeats 2
#   tools/cascade_bench/tune-cascade.sh --guard    # A/B the memory guard on the winning layout
#
# --config is the config setup wrote (strata-<model>.json in the repo root).
# See docs/TUNING.md. The bench prompt overfits - confirm the winner on your real workload.
set -euo pipefail

config=""
port=8080
max_tokens=5000
repeats=1
tag=tune
guard=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --max-tokens) max_tokens="$2"; shift 2 ;;
    --repeats) repeats="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --guard) guard=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$(dirname "$here")")"          # tools/cascade_bench -> repo root
py="$root/.venv/bin/python"
core="$here/tune_cascade.py"

[[ -f "$py" ]] || { echo "no python venv at $py - run ./setup.sh once." >&2; exit 1; }

# default: the config setup wrote, picked by newest
if [[ -z "$config" ]]; then
  config="$(ls -t "$root"/strata-*.json 2>/dev/null | head -n1 || true)"
  [[ -n "$config" ]] || { echo "no config found: run ./setup.sh first, then pass --config <strata-*.json>" >&2; exit 1; }
fi

# a relative config resolves against the repo root, then this folder
if [[ "$config" != /* ]]; then
  if [[ -f "$root/$config" ]]; then config="$root/$config"
  else config="$here/$config"; fi
fi

echo "=== detecting this PC ==="
ram_gib=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)
# RAM the model may use: total minus what the OS + desktop hold (~3 GiB).
free_gib=$(awk -v r="$ram_gib" 'BEGIN {printf "%.1f", r-3}')
# the OS's REAL free RAM right now (MemAvailable): the honest input for the pinned budget, the same figure the
# engine reads for --host-budget-gib auto.  Falls back to the total-3 estimate on a kernel without MemAvailable.
avail_gib=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
[ -n "$avail_gib" ] || avail_gib="$free_gib"

gpu_count=0
helper_gibs=""
primary_gib=""
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS=',' read -r name mem free; do
    name="$(echo "$name" | xargs)"; mem="$(echo "$mem" | xargs)"; free="$(echo "$free" | xargs)"
    if [[ "$free" =~ ^[0-9]+$ ]]; then
      echo "  GPU: $name  (${mem} MiB, ${free} MiB free)"
    else
      echo "  GPU: $name  (${mem} MiB)"; free=""
    fi
    gpu_count=$((gpu_count+1))
    if [[ $gpu_count -eq 1 ]]; then
      primary_gib=$(awk -v m="$mem" 'BEGIN {printf "%.1f", m/1024.0}')
    fi
    if [[ $gpu_count -ge 2 && $gpu_count -le 4 ]]; then
      # Usable VRAM is hybrid, like the engine: the driver's real free minus a ~0.5 GiB reserve, capped at 82% of
      # the card's total (an over-large count makes the engine refuse to start, not clamp).
      if [[ -n "$free" ]]; then
        gib=$(awk -v m="$mem" -v f="$free" 'BEGIN {c=(m/1024.0)*0.82; u=f/1024.0-0.5; if(u<0)u=0; printf "%.2f", (u<c?u:c)}')
      else
        gib=$(awk -v m="$mem" 'BEGIN {printf "%.2f", (m/1024.0)*0.82}')
      fi
      helper_gibs="${helper_gibs:+$helper_gibs,}$gib"
    fi
  done < <(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || true)
fi

# AMD via rocm-smi when present (the cascade's AMD path is unmeasured - warn)
if command -v rocm-smi >/dev/null 2>&1; then
  echo "  NOTE: AMD/HIP + the cascade is not measured (docs/TUNING.md section 6)."
fi

echo "  RAM: ${ram_gib} GiB  (${avail_gib} GiB free right now)"

args=(--config "$config" --python "$py" --port "$port"
      --max-tokens "$max_tokens" --repeats "$repeats"
      --free-gib "$free_gib" --available-gib "$avail_gib" --tag "$tag")
if [[ "$guard" == "1" ]]; then args+=(--guard); fi
if [[ -n "$helper_gibs" ]]; then
  echo "  helper GPU(s) -> usable VRAM ${helper_gibs} GiB"
  args+=(--helper-gib "$helper_gibs")
else
  echo "  single GPU -> no --expert-cache-device1 sweep"
fi
if [[ -n "$primary_gib" ]]; then
  args+=(--primary-gib "$primary_gib")
fi

echo ""
exec "$py" "$core" "${args[@]}"
