#!/usr/bin/env python3
"""Compare K8V4 V layouts in one binary with output and graph parity gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('binary', type=Path)
p.add_argument('out', type=Path)
p.add_argument('--cases', default='33:33,1500:128,4096:256,16384:512,32768:512')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
records = []
with tempfile.TemporaryDirectory(prefix='strata-k8v4-layout-') as tmp:
    for ctx, nq in (map(int, case.split(':')) for case in a.cases.split(',')):
        for pair in range(2):
            results = {}
            for arm in (('base', 'transpose') if pair == 0 else ('transpose', 'base')):
                label = f'{ctx}-{nq}-{pair}-{arm}'
                output = Path(tmp) / (label + '.f32')
                env = dict(os.environ, STRATA_PA_K8V4_STAGE='1',
                           STRATA_PA_K8V4_V_TRANSPOSE=str(int(arm == 'transpose')),
                           QA_GRAPH='1', QA_EMPTY='1', QA_OUTPUT=str(output))
                proc = subprocess.run([str(a.binary.resolve()), str(ctx), str(nq), '10', '3'],
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                (a.out / (label + '.log')).write_text(proc.stdout)
                if proc.returncode or 'PASS k8v4' not in proc.stdout:
                    raise RuntimeError(f'{label} failed: {proc.stdout}')
                match = re.search(r'-> ([0-9.]+) ms per chunk', proc.stdout)
                if not match:
                    raise RuntimeError(f'{label}: missing kernel timing: {proc.stdout}')
                raw = output.read_bytes()
                results[arm] = {'ms': float(match[1]), 'sha256': hashlib.sha256(raw).hexdigest(),
                                'output_bytes': len(raw)}
            if results['base']['sha256'] != results['transpose']['sha256']:
                raise RuntimeError(f'{ctx}/{nq}: output bits differ: {results}')
            row = {'ctx': ctx, 'nq': nq, 'pair': pair, 'results': results,
                   'speedup': results['base']['ms'] / results['transpose']['ms'],
                   'bitwise_equal': True}
            records.append(row)
            a.out.joinpath('results.json').write_text(json.dumps(records, indent=2) + '\n')
            print(json.dumps(row), flush=True)
        print(f'ctx {ctx}: median speedup '
              f'{statistics.median(row["speedup"] for row in records[-2:]):.3f}x', flush=True)
