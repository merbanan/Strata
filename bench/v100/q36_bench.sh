#!/bin/bash
# bench/v100/q36_bench.sh GGUF [IDS_FILE]: the V100 tree's strata-q36 checks - decode / prompt at a short and a 16K
# prompt (--max-context 131072, as the setup writes it), MTP 2, and a 16K greedy run whose tokens are compared across
# builds (md5).  GGUF: a Qwen3.6-35B-A3B *-MTP-* file (strata-q36 needs its MTP block even for plain decode).
# IDS_FILE: comma-separated prompt ids (default: 16,000 synthetic ids).
set -e
M=${1:?usage: q36_bench.sh GGUF [IDS_FILE]}
IDS=${2:-/tmp/q36_bench_ids.txt}
[ -f "$IDS" ] || python3 -c "print(','.join(str(1000 + (i * 7919) % 60000) for i in range(16000)))" > "$IDS"
B=${STRATA_BUILD:-build}
for pp in 512 16384; do
  $B/strata-q36 --native "$M" --max-context 131072 --bench-pp $pp --bench-tg 256 2>&1 | grep "bench 2" | sed "s/^/plain  /"
done
$B/strata-q36 --native "$M" --max-context 131072 --mtp 2 --bench-pp 512 --bench-tg 256 2>&1 | grep "bench 2" | sed "s/^/mtp2   /"
$B/strata-q36 --native "$M" --max-context 32768 --tokens "$(cat "$IDS")" --n 128 > /tmp/q36_bench_gen.txt 2> /tmp/q36_bench_gen.err
grep -o "generated in [0-9.]* ms ([0-9.]* tok/s)" /tmp/q36_bench_gen.err | sed "s/^/16K greedy /"
md5sum /tmp/q36_bench_gen.txt | cut -c1-12 | sed "s/^/tokens md5 /"
