#!/usr/bin/env python3
"""Replay every archived decode task with graph diagnostics; not a speed test."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('warmup_reference', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cases = json.loads(args.cases.read_text())
    warmup = json.loads(args.warmup_reference.read_text())
    assert config['env']['STRATA_GRAPH_MEMORY_LOG'] == '-'
    assert config['env']['STRATA_VERIFY_NODES'] == '1'
    assert config['env']['LD_PRELOAD']
    assert cases[0]['name'] == 'warmup' and len(cases) == 7
    assert warmup['request_sha256'] == cases[0]['request_sha256']
    args.out.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).with_name('decode-compare.py')
    spec = importlib.util.spec_from_file_location('decode_runner', runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {'config': config, 'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(Path(config['exe']).read_bytes()).hexdigest(),
                'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
                'scope': 'Archived full workload on the installed release with graph diagnostics; not a throughput comparison.'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.out / 'cases.json').write_bytes(args.cases.read_bytes())
    engine = module.Engine(config, args.out)
    rows = []
    try:
        engine.ready()
        for case in cases:
            tokens, done, timing = engine.generate(case['ids'], case['body']['max_tokens'])
            row = {'name': case['name'], 'request_sha256': case['request_sha256'], 'tokens': tokens,
                   'done': done, 'diagnostic_timing_only': timing}
            rows.append(row)
            (args.out / (case['name'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
            if case['name'] == 'warmup':
                assert tokens == warmup['tokens'] and done[5] == 'stop', 'warmup output differs'
            print(json.dumps({key: value for key, value in row.items() if key != 'tokens'}), flush=True)
    finally:
        engine.close()
    assert engine.process.returncode == 0
    (args.out / 'summary.json').write_text(json.dumps({'completed_tasks': len(rows), 'engine_exit_code': 0,
                                                       'task_quality': 'Not qualified by this diagnostic runner; original tasks retained.'},
                                                      indent=2) + '\n')


if __name__ == '__main__':
    main()
