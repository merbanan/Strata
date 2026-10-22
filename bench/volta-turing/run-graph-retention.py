#!/usr/bin/env python3
"""Build and run a fresh-process retention matrix on an idle CUDA device."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--arch', default='sm_75')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out.resolve()
    source = Path(__file__).with_name('graph-retention-probe.cu')
    shutil.copy2(source, out / source.name)
    shutil.copy2(__file__, out / Path(__file__).name)
    env = dict(os.environ, CCACHE_DISABLE='1')
    env.pop('LD_PRELOAD', None)
    state = {'status': 'build', 'pid': os.getpid(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'steps': [], 'scope': 'Synthetic sequential kernel graphs; no engine policy or throughput claim.'}

    def save():
        (out / 'run.json').write_text(json.dumps(state, indent=2) + '\n')

    def run(label, cmd):
        state['status'] = label
        save()
        start = time.time()
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['steps'].append({'name': label, 'command': cmd, 'returncode': result.returncode,
                               'start': start, 'end': time.time()})
        save()
        if result.returncode:
            raise RuntimeError(f'{label}: exit {result.returncode}')

    try:
        run('build', ['nvcc', '-O2', '-std=c++17', '-arch=' + args.arch, '-Xcompiler=-Wall,-Wextra',
                      str(out / source.name), '-o', str(out / 'probe')])
        state['binary_sha256'] = hashlib.sha256((out / 'probe').read_bytes()).hexdigest()
        results = []
        for repetition, caps in enumerate(([5, 1, 2, 3], [3, 2, 1, 5]), 1):
            for headroom in [0, 512]:
                for cap in caps:
                    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                      '--format=csv,noheader'], text=True).strip()
                    if active:
                        raise RuntimeError('GPU has active compute processes: ' + active)
                    label = f'r{repetition}-headroom{headroom}-cap{cap}'
                    run(label, [str(out / 'probe'), str(cap), str(headroom)])
                    rows = [json.loads(line) for line in (out / (label + '.log')).read_text().splitlines()]
                    assert rows[-1]['event'] == 'pass' and rows[-1]['checked'] == 60 and rows[-1]['launches'] == 180
                    samples = [row for row in rows if 'free' in row]
                    start_free = next(row['free'] for row in samples if row['event'] == 'start')
                    assert max(row['live'] for row in samples) == cap
                    phases = []
                    for phase in range(3):
                        selected = [row for row in samples if row['phase'] == phase]
                        end = next(row for row in selected if row['event'] == 'phase_destroyed')
                        assert end['live'] == 0
                        phases.append({'phase': phase,
                                       'peak_decrease_bytes': start_free - min(row['free'] for row in selected),
                                       'after_destroy_decrease_bytes': start_free - end['free'],
                                       'instantiate_ms': sum(row['duration_ms'] for row in selected if row['event'] == 'instantiate')})
                    results.append({'label': label, 'cap': cap, 'headroom_mib': headroom, 'phases': phases,
                                    'max_graph_allocator_reserved': max(row['pool_reserved'] for row in samples),
                                    'creations': rows[-1]['creations'],
                                    'log_sha256': hashlib.sha256((out / (label + '.log')).read_bytes()).hexdigest()})
                    (out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
                    print(json.dumps(results[-1]), flush=True)
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
