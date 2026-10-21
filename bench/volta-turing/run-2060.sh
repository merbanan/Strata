#!/usr/bin/env bash
# Run only after the GPU is visible. Stages: micro, modes, wide.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
workspace=$(dirname "$repo")
bench="$workspace/Strata-data/bench"
stage=${1:-micro}
case "$stage" in micro|modes|wide) ;; *) echo 'usage: run-2060.sh [micro|modes|wide]' >&2; exit 2;; esac
out=${OUT:-"$bench/2026-10-21-volta-turing/$(date +%Y%m%d-%H%M%S)-$stage"}
mkdir -p "$out"
# Fail before any server launch when device access is unavailable.
if ! nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$out/gpu.csv" 2>&1; then
    cat "$out/gpu.csv" >&2
    echo "GPU tests NOT RUN. Diagnostic: $out/gpu.csv" >&2
    exit 1
fi
if ! grep -q 'RTX 2060 SUPER' "$out/gpu.csv"; then
    echo 'RTX 2060 SUPER not found; refusing to benchmark a different device.' >&2
    exit 1
fi
count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$count" -ne 1 ]; then echo 'Multiple GPUs: select the RTX 2060 SUPER explicitly before adapting this runner.' >&2; exit 1; fi
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$apps" ]; then echo 'GPU has active compute processes; stop them before benchmarking.' >&2; exit 1; fi
git -C "$repo" rev-parse HEAD > "$out/revision.txt"
git -C "$repo" diff > "$out/changes.patch"
python3 "$repo/bench/volta-turing/prepare.py" "$out"
if [ "$stage" = micro ]; then
    "$repo/build/mmvq_il_parity" > "$out/parity.log" 2>&1
    for repeat in 1 2; do
        "$repo/build/mmvq_il_parity" --bench --emit-table --pipeline --group16-bench > "$out/pipeline-$repeat.log" 2>&1
    done
    compute-sanitizer --tool memcheck --error-exitcode 1 "$repo/build/mmvq_il_parity" > "$out/memcheck.log" 2>&1
else
    if [ "$stage" = modes ]; then candidates=(mode0 mode4 mode8); else candidates=(v20-default wide1 wide2); fi
    if [ "$stage" = wide ]; then
        "$repo/build/mmvq_il_parity" > "$out/parity.log" 2>&1
    fi
    # Gate on correctness; the older ab4 wrapper logs a failed probe but still benchmarks.
    for name in base "${candidates[@]}"; do
        label="vt-$(basename "$out")-$name"
        bash "$bench/kv1188.sh" "$label" "$out/$name.json" > "$out/$name-probe.log" 2>&1
        if ! grep -q '156 tok, finish=stop, consecutive repeats=0' "$out/$name-probe.log"; then
            echo "Correctness probe failed for $name; stopping. See $out/$name-probe.log" >&2
            exit 1
        fi
    done
    for name in "${candidates[@]}"; do
        label="vt-$(basename "$out")-$name"
        bash "$bench/ab4.sh" "$label" "$out/base.json" "$out/$name.json" \
            "$workspace/Strata-data/packs/iq3_xxs" "${TARGETS:-1024,4096,16384,32768}" "${PAIRS:-4}" > "$out/$name-ab.log" 2>&1
        if grep -qE 'FAILED|PROBE FAILED' "$out/$name-ab.log"; then
            echo "Benchmark failed: $out/$name-ab.log" >&2; exit 1
        fi
    done
fi
echo "Completed $stage: $out"
