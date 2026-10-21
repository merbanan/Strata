#!/usr/bin/env bash
# Usage: lookup-compare.sh <unique-label> <base.json> <candidate.json>
# ABBA: one edit/fresh-code/prose pass per model load, using the existing lookup workload.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
workspace=$(dirname "$repo")
bench="$workspace/Strata-data/bench"
label=$1; base=$2; candidate=$3
out="$bench/2026-10-21-volta-turing/$label"
mkdir "$out"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$out/gpu.csv"
grep -q 'RTX 2060 SUPER' "$out/gpu.csv"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
cp "$base" "$out/base.json"
cp "$candidate" "$out/candidate.json"
sha256sum "$repo/build/strata" > "$out/executable.sha256"
git -C "$repo" diff > "$out/changes.patch"
unit=
cleanup() { if [ -n "$unit" ]; then systemctl --user stop "$unit" || true; fi; }
trap cleanup EXIT
leg=0
for arm in base candidate candidate base; do
    leg=$((leg + 1))
    dir="$out/$leg-$arm"
    mkdir "$dir"
    python3 - "$out/$arm.json" "$dir" <<'PY'
import json, sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text()); d=Path(sys.argv[2])
c['port']=18080; c['host']='127.0.0.1'; c['log']=str(d/'engine.log')
(d/'config.json').write_text(json.dumps(c,indent=2)+'\n')
PY
    unit="strata-vt-$label-$leg"
    systemd-run --user --collect --unit="$unit" -p "WorkingDirectory=$repo" \
        -p "StandardOutput=file:$dir/server.log" -p "StandardError=file:$dir/server.log" \
        "$workspace/Strata/.venv/bin/python" "$repo/serve/server.py" --engine strata \
        --config "$dir/config.json" --port 18080 --host 127.0.0.1
    ready=0
    for attempt in $(seq 1 120); do
        if curl -fsS -m 3 http://127.0.0.1:18080/v1/status 2>/dev/null | grep -q '"loaded": true'; then ready=1; break; fi
        systemctl --user is-active -q "$unit" || { echo "Server failed: $dir/server.log" >&2; exit 1; }
        sleep 5
    done
    [ "$ready" -eq 1 ] || { echo 'Server readiness timed out' >&2; exit 1; }
    python3 "$bench/2026-10-21-lookup/probe.py" http://127.0.0.1:18080 \
        "$dir/engine.log" "$dir/results.json" 1 > "$dir/probe.log" 2>&1
    python3 - "$dir/results.json" "$arm" "$leg" <<'PY'
import json, sys
from pathlib import Path
rows=json.loads(Path(sys.argv[1]).read_text())
assert len(rows)==4
for row in rows:
    assert row.get('gen',0)>0 and row.get('dec_tok_s',0)>0, row
    assert not Path(sys.argv[1]+'.0-'+row['probe']+'.txt').read_text().startswith('ERROR ')
    print(sys.argv[2],sys.argv[3],json.dumps(row),flush=True)
PY
    systemctl --user stop "$unit"
    unit=
    sleep 3
done
echo "Completed lookup ABBA: $out"
