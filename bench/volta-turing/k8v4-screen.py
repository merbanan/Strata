#!/usr/bin/env python3
"""Compare opt-in K8V4 staging to existing mode 3, including FP64/graph gates."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('binary', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--cases', default='33:33,1500:128,4096:256,16384:512,32768:512',
               help='comma-separated context:query counts')
p.add_argument('--reference-binary', type=Path, help='original engine kernel parity executable')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
records = []
with tempfile.TemporaryDirectory(prefix='strata-k8v4-') as tmp:
    for ctx, nq in (map(int, case.split(':')) for case in a.cases.split(',')):
        pair_rows = []
        for pair in range(2):
            outputs = {}
            hashes = {}
            timings = {}
            arms = ['base', 'candidate'] if pair == 0 else ['candidate', 'base']
            if a.reference_binary:
                arms.append('reference')
            for arm in arms:
                label = f'{ctx}-{nq}-{pair}-{arm}'
                output = Path(tmp) / (label + '.f32')
                env = dict(os.environ, STRATA_PA_K8V4_STAGE=str(int(arm == 'candidate')),
                           QA_GRAPH='1', QA_EMPTY='1', QA_OUTPUT=str(output))
                binary = a.reference_binary if arm == 'reference' else a.binary
                proc = subprocess.run([str(binary.resolve()), str(ctx), str(nq), '10', '3'],
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                (a.out / (label + '.log')).write_text(proc.stdout)
                if proc.returncode or 'PASS k8v4' not in proc.stdout:
                    raise RuntimeError(f'{label} failed: {proc.stdout}')
                timings[arm] = float(re.search(r'-> ([0-9.]+) ms per chunk', proc.stdout)[1])
                data = array.array('f')
                raw = output.read_bytes()
                hashes[arm] = hashlib.sha256(raw).hexdigest()
                data.frombytes(raw)
                if not data or not all(math.isfinite(x) for x in data):
                    raise RuntimeError(f'{label}: nonfinite or empty output')
                outputs[arm] = data
            base, candidate = outputs['base'], outputs['candidate']
            assert len(base) == len(candidate)
            scale = max(map(abs, base))
            delta = max(abs(b - c) for b, c in zip(base, candidate))
            # Preserve the original scale FMA order and Q4-to-half boundary.
            # Byte hashes also detect signed-zero changes hidden by max_abs.
            if len(set(hashes.values())) != 1:
                raise RuntimeError(f'{ctx}/{nq}: output bits differ: {hashes}; max_abs={delta}')
            row = dict(ctx=ctx, nq=nq, pair=pair, ms=timings, max_abs=delta,
                       sha256=hashes, bitwise_equal=True,
                       relative_to_scale=delta / scale,
                       speedup=timings['base'] / timings['candidate'])
            records.append(row)
            pair_rows.append(row)
            (a.out / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        print(f'ctx {ctx}: median speedup {statistics.median(r["speedup"] for r in pair_rows):.3f}x', flush=True)
