#!/usr/bin/env bash
# Matched production-config screen for opt-in inference candidates. Writes only under TMP.
set -euo pipefail

if (( $# != 2 )); then
    echo "usage: $0 BASE.json CANDIDATE.json" >&2
    exit 2
fi
repo=$(cd "$(dirname "$0")/../.." && pwd)
workspace=$(dirname "$repo")
root=${OUT:-/run/media/benjamin/TMP/strata-v20-abba/full-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$root"
base=$1
candidate=$2
python_bin="$workspace/Strata/.venv/bin/python"
benchmark="$workspace/Strata-data/bench/2026-10-04-profile-q2_0-128k-prtest/benchmark.py"
pack="$workspace/Strata-data/packs/iq3_xxs"
port=18080
unit=

cleanup() {
    if [[ -n "$unit" ]]; then
        systemctl --user stop "$unit" >/dev/null 2>&1 || true
        unit=
    fi
}
trap cleanup EXIT INT TERM

git -C "$repo" rev-parse HEAD > "$root/revision.txt"
sha256sum "$repo/build/strata" "$benchmark" > "$root/sha256.txt"
cp "$base" "$root/base.json"
cp "$candidate" "$root/candidate.json"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU has an active compute process" >&2
    exit 1
fi

arms=(base candidate candidate base)
for i in "${!arms[@]}"; do
    arm=${arms[$i]}
    leg="$root/$((i + 1))-$arm"
    mkdir -p "$leg"
    python3 - "$root/$arm.json" "$leg/config.json" "$leg/engine.log" "$port" <<'PY'
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
config['log'] = sys.argv[3]
config['port'] = int(sys.argv[4])
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + '\n')
PY
    unit="strata-vt-$i-$$"
    systemd-run --user --collect --unit="$unit" \
        -p "WorkingDirectory=$repo" \
        -p "StandardOutput=append:$leg/server.log" \
        -p "StandardError=append:$leg/server.log" \
        -E TMPDIR=/run/media/benjamin/TMP \
        "$python_bin" "$repo/serve/server.py" --engine strata \
        --config "$leg/config.json" --host 127.0.0.1 --port "$port" > "$leg/systemd-run.log"
    ready=0
    for ((attempt=0; attempt<120; attempt++)); do
        if curl -fsS -m 3 "http://127.0.0.1:$port/health" > "$leg/health.json" 2>/dev/null &&
           "$python_bin" - "$leg/health.json" <<'PY' >/dev/null 2>&1
import json, sys
sys.exit(not json.load(open(sys.argv[1])).get('loaded'))
PY
        then
            ready=1
            break
        fi
        if ! systemctl --user is-active --quiet "$unit"; then
            tail -n 30 "$leg/server.log" >&2
            exit 1
        fi
        sleep 5
    done
    if (( ! ready )); then
        echo "$arm leg $i did not become ready" >&2
        exit 1
    fi
    curl -fsS -m 300 "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"strata","messages":[{"role":"user","content":"Write the numbers from 1 to 60 in words, separated by commas, with no other text."}],"max_tokens":400,"temperature":0,"reasoning_effort":"none"}' \
        > "$leg/probe.json"
    "$python_bin" - "$leg/probe.json" <<'PY'
import json, sys
response = json.load(open(sys.argv[1]))
choice = response['choices'][0]
words = [part.strip() for part in (choice['message']['content'] or '').split(',')]
repeats = sum(a == b for a, b in zip(words, words[1:]))
assert response['usage']['completion_tokens'] == 156, response['usage']
assert choice['finish_reason'] == 'stop' and repeats == 0, (choice['finish_reason'], repeats)
PY
    "$python_bin" "$benchmark" --root "$repo" --pack "$pack" \
        --url "http://127.0.0.1:$port" --out "$leg" \
        --targets "${TARGETS:-4096,16384}" --runs 1 > "$leg/benchmark.log" 2>&1
    cleanup
    for ((attempt=0; attempt<60; attempt++)); do
        [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
        sleep 2
    done
    echo "completed $((i + 1))/4: $arm ($leg)"
done

python3 - "$root" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
legs = [json.loads((root / f'{i}-{arm}' / 'results.json').read_text())
        for i, arm in enumerate(('base', 'candidate', 'candidate', 'base'), 1)]
report = {}
for label in sorted({row['label'] for row in legs[0]} - {'warmup'}):
    rows = [next(row for row in leg if row['label'] == label) for leg in legs]
    work_keys = ('prompt_tokens', 'reused', 'engine_generated', 'drafts_offered',
                 'drafts_accepted', 'ram_blobs', 'file_blobs', 'hit_rate', 'pcie_share')
    report[label] = {
        'same_text_all_legs': len({row['text'] for row in rows}) == 1,
        'same_request_all_legs': len({row['request_sha256'] for row in rows}) == 1,
        'same_work_all_legs': all(len({row['engine'].get(key) for row in rows}) == 1
                                  for key in work_keys),
        'prefill_tok_s': [(row['engine']['prompt_tokens'] - (row['engine'].get('reused') or 0)) /
                          (row['engine']['prompt_ms'] / 1000) for row in rows],
        'decode_tok_s': [row['engine']['engine_generated'] /
                         (row['engine']['decode_ms'] / 1000) for row in rows],
        'generated_tokens': [row['engine']['engine_generated'] for row in rows],
    }
    for metric in ('prefill_tok_s', 'decode_tok_s'):
        a, b, b2, a2 = report[label][metric]
        report[label][metric + '_paired_ratios'] = [b / a, b2 / a2]
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
