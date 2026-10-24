#!/usr/bin/env python3
"""Trace actual prompt-attention routes for short pinned prefixes.

The extra token after each pinned prefix is read through a verifier window.
This is route evidence, not a performance or model-quality benchmark.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time


def set_arg(args, name, value):
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args.extend([name, str(value)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('cases', type=Path, help='prefill-teacher cases.json with token IDs')
    parser.add_argument('out', type=Path)
    parser.add_argument('--prefixes', default='2,4,8,16,64,256')
    a = parser.parse_args()
    prefixes = [int(x) for x in a.prefixes.split(',')]
    assert prefixes and all(x > 0 for x in prefixes)
    cfg = json.loads(a.config.read_text())
    case = json.loads(a.cases.read_text())[0]
    assert len(case['ids']) >= max(prefixes) + 1
    for key, value in [('--adapt-swaps', 0), ('--adapt-async', 0), ('--prompt-cache', 0),
                       ('--prompt-cache-root', 0), ('--turn-token', -1), ('--short-read', 1)]:
        set_arg(cfg['args'], key, value)
    cfg['env'].update(STRATA_PA_ROUTE_TRACE='1', STRATA_TRACE='1',
                      STRATA_PREFIX_SNAPSHOTS='0', STRATA_CACHE_MESSAGE_BOUNDARY='0')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (a.out / 'source.sha256').write_text(hashlib.sha256(a.cases.read_bytes()).hexdigest() + '\n')
    spec = importlib.util.spec_from_file_location('decode_compare', Path(__file__).with_name('decode-compare.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = module.Engine(cfg, a.out)
    rows = []
    try:
        engine.ready()
        for n in prefixes:
            ids = case['ids'][:n + 1]
            engine.process.stdin.write((f'GEN 1 pin={n} ' + ','.join(map(str, ids)) + '\n').encode())
            engine.process.stdin.flush()
            deadline = time.monotonic() + 1200
            while True:
                line = engine.line(deadline)
                if line.startswith(b'DONE '):
                    done = line.decode().split()
                    break
            assert int(done[2]) == len(ids) and int(done[8]) == 0, done
            rows.append({'prefix': n, 'done': done})
            print(n, done, flush=True)
    finally:
        engine.close()
    assert engine.process.returncode == 0
    log = (a.out / 'engine.log').read_text()
    routes = [dict(pos=int(pos), qsa=int(qsa), queries=int(n), route=route)
              for pos, qsa, n, route in re.findall(
                  r'strata pa route: prefill pos=(\d+) qsa=(\d+) queries=(\d+) route=(\w+)', log)]
    summary = {'requests': rows, 'routes': routes,
               'limits': 'Pinned-prefix route trace; no performance or quality claim.'}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(routes)} prompt-attention route records', flush=True)


if __name__ == '__main__':
    main()
