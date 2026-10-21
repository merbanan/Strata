#!/usr/bin/env bash
# Usage: compare.sh <unique-label> <base.json> <candidate.json> [targets] [pairs]
# Same existing engine benchmark, with a fail-closed probe and no duplicate probe load.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
workspace=$(dirname "$repo")
bench="$workspace/Strata-data/bench"
label=$1; base=$2; candidate=$3; targets=${4:-4096,16384}; pairs=${5:-2}
out="$bench/2026-10-21-volta-turing/$label"
mkdir "$out"  # Unique results; do not overwrite an earlier measurement.
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$out/gpu.csv"
grep -q 'RTX 2060 SUPER' "$out/gpu.csv"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
cp "$base" "$out/base.json"
cp "$candidate" "$out/candidate.json"
git -C "$repo" diff > "$out/changes.patch"
python3 - "$repo" "$out" <<'PY'
import hashlib, json, shutil, subprocess, sys
from pathlib import Path
repo, out = map(Path, sys.argv[1:])
manifest = {'head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
            'binaries': {}, 'changed_sources': {}}
for arm in ('base', 'candidate'):
    exe = Path(json.loads((out / f'{arm}.json').read_text())['exe'])
    manifest['binaries'][arm] = {'path': str(exe), 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}
# A normal git diff omits a new CUDA header. Preserve modified and new source
# files as well, without copying build symlinks or large benchmark artifacts.
paths = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '--modified', '--others', '--exclude-standard', '-z']).decode().split('\0')
for name in sorted(set(paths)):
    if not (name.startswith(('src/', 'include/')) or name == 'CMakeLists.txt'): continue
    source = repo / name
    if not source.is_file(): continue
    target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest['changed_sources'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
python3 "$repo/bench/volta-turing/telemetry.py" "$$" "$out/telemetry.jsonl" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
bash "$bench/kv1188.sh" "$label-probe" "$out/candidate.json" > "$out/probe.log" 2>&1
if ! grep -q '156 tok, finish=stop, consecutive repeats=0' "$out/probe.log"; then
    echo "Correctness probe failed: $out/probe.log" >&2; exit 1
fi
for pair in $(seq 1 "$pairs"); do
    if [ $((pair % 2)) -eq 1 ]; then arms=(base candidate); else arms=(candidate base); fi
    for arm in "${arms[@]}"; do
        leg="vt-$label-$arm-$pair"
        TARGETS="$targets" RUNS=1 bash "$bench/cfg_ab.sh" "$leg" "$out/$arm.json" \
            "$workspace/Strata-data/packs/iq3_xxs" > "$out/$arm-$pair.log" 2>&1
        python3 - "$bench/2026-10-06-pr-$leg/summary.json" "$arm" "$pair" "$out/runs.jsonl" <<'PY'
import json, math, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
assert summary, "empty benchmark summary"
with open(sys.argv[4], 'a') as out:
    for target, stats in summary.items():
        pp, dec = stats['prefill_tok_s']['median'], stats['decode_tok_s']['median']
        assert all(math.isfinite(x) and x > 0 for x in (pp, dec))
        row = dict(arm=sys.argv[2], pair=int(sys.argv[3]), target=int(target),
                   prefill=pp, decode=dec, source=sys.argv[1])
        out.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
PY
    done
done
python3 - "$out/runs.jsonl" <<'PY'
import json, statistics, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
report = {}
for target in sorted({r['target'] for r in rows}):
    paired = {}
    for r in rows:
        if r['target'] == target: paired.setdefault(r['pair'], {})[r['arm']] = r
    assert all(set(p) == {'base', 'candidate'} for p in paired.values())
    report[target] = {}
    for metric in ('prefill', 'decode'):
        base = [p['base'][metric] for p in paired.values()]
        cand = [p['candidate'][metric] for p in paired.values()]
        delta = [c-b for b,c in zip(base,cand)]
        report[target][metric] = dict(base=statistics.median(base), candidate=statistics.median(cand),
            paired_delta=statistics.median(delta), delta_range=[min(delta),max(delta)], pairs=len(delta))
print(json.dumps(report, indent=2))
Path(sys.argv[1]).with_name('summary.json').write_text(json.dumps(report, indent=2)+'\n')
PY
