#!/usr/bin/env python3
"""Audit a completed attention ABBA: matched requests, memory budgets and latency."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--targets', default='1024,4096,16384,32768')
p.add_argument('--pairs', type=int, default=2)
p.add_argument('--switch', default='STRATA_PA_K8V4_STAGE',
               help='the sole environment switch allowed to differ between arms')
a = p.parse_args()
targets = list(map(int, a.targets.split(',')))
rows = [json.loads(line) for line in (a.directory / 'runs.jsonl').read_text().splitlines()]
expected = {(arm, pair, target) for arm in ('base', 'candidate')
            for pair in range(1, a.pairs + 1) for target in targets}
if len(rows) != len(expected) or {(r['arm'], r['pair'], r['target']) for r in rows} != expected:
    raise SystemExit('missing or duplicate arm/pair/target results')
manifest = json.loads((a.directory / 'manifest.json').read_text())
if len({b['sha256'] for b in manifest['binaries'].values()}) != 1:
    raise SystemExit('attention comparison must use one binary with only the switch changed')
for binary in manifest['binaries'].values():
    if hashlib.sha256(Path(binary['path']).read_bytes()).hexdigest() != binary['sha256']:
        raise SystemExit('measured binary has changed')
for name, digest in manifest['changed_sources'].items():
    if hashlib.sha256((a.directory / 'source' / name).read_bytes()).hexdigest() != digest:
        raise SystemExit('saved source has changed: ' + name)
configs = [json.loads((a.directory / f'{arm}.json').read_text()) for arm in ('base', 'candidate')]
for ordinal, cfg in enumerate(configs):
    if cfg['env'].pop(a.switch) != str(ordinal):
        raise SystemExit('wrong attention switch')
    cfg.pop('log', None)
if configs[0] != configs[1]:
    raise SystemExit('other configuration differences')

details = []
budgets = {}
for row in rows:
    directory = Path(row['source']).parent
    results = json.loads((directory / 'results.json').read_text())
    case = next(r for r in results if r['label'] == f'tokens-{row["target"]}-run-1')
    engine = case['engine']
    if engine['reused'] or engine['prompt_tokens'] != case['expected_prompt_tokens']:
        raise SystemExit('reused or mismatched input: ' + str(directory))
    if engine['output_tokens'] != 256 or case['finish_reason'] != 'length':
        raise SystemExit('benchmark output allowance changed: ' + str(directory))
    if str(directory) not in budgets:
        log = (directory / 'engine.log').read_text()
        budgets[str(directory)] = {
            'cache_slots': sorted(set(map(int, re.findall(r'expert cache (\d+) slots', log)))),
            'chunk_tokens': sorted(set(map(int, re.findall(r'prompt chunk auto: (\d+) tokens', log)))),
            'pinned_expert_gib': sorted(set(map(float, re.findall(r'allocating ([\d.]+) GiB page-locked cache complement', log)))),
        }
        if not all(budgets[str(directory)].values()):
            raise SystemExit('missing memory-budget evidence: ' + str(directory))
    details.append(dict(row, ttft=case['client_ttft_s'], elapsed=case['client_elapsed_s'],
                        request_sha256=case['request_sha256'],
                        text_sha256=hashlib.sha256(case['text'].encode()).hexdigest(),
                        drafts_accepted=engine['drafts_accepted'], drafts_offered=engine['drafts_offered']))
if len({json.dumps(b, sort_keys=True) for b in budgets.values()}) != 1:
    raise SystemExit('memory budgets differ across arms')
report = {}
for target in targets:
    cases = [r for r in details if r['target'] == target]
    if len({r['request_sha256'] for r in cases}) != 1:
        raise SystemExit(f'different input requests at {target}')
    entry = {'distinct_output_texts': len({r['text_sha256'] for r in cases}), 'metrics': {}}
    for metric in ('prefill', 'decode', 'ttft', 'elapsed'):
        base = [next(r[metric] for r in cases if r['arm'] == 'base' and r['pair'] == pair)
                for pair in range(1, a.pairs + 1)]
        cand = [next(r[metric] for r in cases if r['arm'] == 'candidate' and r['pair'] == pair)
                for pair in range(1, a.pairs + 1)]
        entry['metrics'][metric] = {'base_median': statistics.median(base),
                                    'candidate_median': statistics.median(cand),
                                    'paired_percent_change': [100 * (c / b - 1) for b, c in zip(base, cand)]}
    report[target] = entry
output = {'budgets': budgets, 'report': report, 'details': details,
          'limits': '256-token capped throughput requests; text equality is reported, not required or a broad quality score. Negative TTFT/elapsed changes mean lower latency.'}
(a.directory / 'audit.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(report, indent=2))
