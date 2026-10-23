#!/usr/bin/env bash
# Eight fresh v19 engine legs in symmetric order. Screening only: a winning mode
# still needs a separate allocation/quality-matched ABBA before adoption.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
workspace=$(dirname "$repo")
bench="$workspace/Strata-data/bench"
out="$bench/2026-10-21-volta-turing/dispatch-v19-20261023"
mkdir "$out"  # Never overwrite a previous screen.
cleanup() {
    if [[ -n "${leg:-}" ]]; then systemctl --user stop "strata-pr-$leg" 2>/dev/null || true; fi
    bash "$workspace/Strata/start-current.sh" > "$out/restore.log" 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
systemctl --user stop strata-server
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$out/gpu.csv"
python3 - "$bench/2026-10-21-volta-turing/configs/base.json" "$out" <<'PY'
import json, sys
from pathlib import Path
base = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
args = base['args']
args[args.index('--vram-reserve-mib') + 1] = '1024'
for mode in ('default', '0', '4', '8'):
    config = json.loads(json.dumps(base))
    config['env'].pop('STRATA_EXP_MODE', None)
    if mode != 'default': config['env']['STRATA_EXP_MODE'] = mode
    (out / f'{mode}.json').write_text(json.dumps(config, indent=2) + '\n')
PY
for spec in default-1 0-1 4-1 8-1 8-2 4-2 0-2 default-2; do
    mode=${spec%-*}
    leg="dispatch-v19-20261023-$spec"
    echo "START $spec $(date -u +%FT%TZ)" | tee -a "$out/progress.log"
    TARGETS=4096,16384 RUNS=1 bash "$bench/cfg_ab.sh" "$leg" "$out/$mode.json" \
        "$workspace/Strata-data/packs/iq3_xxs" > "$out/$spec.log" 2>&1
    python3 - "$bench/2026-10-06-pr-$leg/summary.json" "$out/$spec.summary.json" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
assert set(data) == {'4096', '16384'}
Path(sys.argv[2]).write_text(json.dumps(data, indent=2) + '\n')
PY
    echo "DONE $spec $(date -u +%FT%TZ)" | tee -a "$out/progress.log"
    leg=
done
python3 - "$out" <<'PY'
import json, statistics, sys
from pathlib import Path
out = Path(sys.argv[1])
report = {}
for mode in ('default', '0', '4', '8'):
    runs = [json.loads((out / f'{mode}-{rep}.summary.json').read_text()) for rep in (1, 2)]
    report[mode] = {target: {metric: [run[target][metric]['median'] for run in runs]
                             for metric in ('prefill_tok_s', 'decode_tok_s')}
                    for target in ('4096', '16384')}
for mode in ('0', '4', '8'):
    for target in ('4096', '16384'):
        report[mode][target]['prefill_ratio_to_default'] = (
            statistics.median(report[mode][target]['prefill_tok_s']) /
            statistics.median(report['default'][target]['prefill_tok_s']))
        report[mode][target]['decode_ratio_to_default'] = (
            statistics.median(report[mode][target]['decode_tok_s']) /
            statistics.median(report['default'][target]['decode_tok_s']))
(out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
