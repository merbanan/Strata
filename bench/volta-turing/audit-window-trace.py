#!/usr/bin/env python3
"""Audit full-model-abba.sh output for the STRATA_TRACE_WINDOWS=0/1 gate."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def traces(path):
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.startswith('strata window trace:'):
            continue
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        groups[int(fields['req'])].append(fields)
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    assert not a.output.exists()
    root = a.root.resolve()
    arms = ('base', 'candidate', 'candidate', 'base')
    legs = [root / f'{i}-{arm}' for i, arm in enumerate(arms, 1)]
    cfgs = [json.loads((leg / 'config.json').read_text()) for leg in legs]
    normalized = []
    for cfg, arm in zip(cfgs, arms):
        cfg = json.loads(json.dumps(cfg))
        assert cfg['env'].pop('STRATA_TRACE_WINDOWS') == ('1' if arm == 'candidate' else '0')
        cfg.pop('log', None)
        normalized.append(cfg)
    assert all(cfg == normalized[0] for cfg in normalized)
    results = [json.loads((leg / 'results.json').read_text()) for leg in legs]
    summaries = json.loads((root / 'summary.json').read_text())
    for label, entry in summaries.items():
        assert entry['same_text_all_legs'] and entry['same_request_all_legs'] and entry['same_work_all_legs'], label
    parsed = [traces(leg / 'engine.log') for leg in legs]
    assert parsed[0] == parsed[3] == {}
    assert sorted(parsed[1]) == sorted(parsed[2]) == [1, 2, 3, 4]
    # This runner sends an unrecorded 156-token probe, then benchmark.py's
    # warmup, 4K request and 16K request in that order.
    mapping = {2: 'warmup', 3: 'tokens-4096-run-1', 4: 'tokens-16384-run-1'}
    trace_report = {}
    decision_keys = ('T', 'mtp', 'chain', 'first', 'suffix', 'accepted', 'offered', 'pipeline')
    for req in (1, 2, 3, 4):
        left, right = parsed[1][req], parsed[2][req]
        assert len(left) == len(right) > 0
        for rows in (left, right):
            assert [int(x['round']) for x in rows] == list(range(1, len(rows) + 1))
            assert all(int(x['req']) == req for x in rows)
        assert [tuple(x[key] for key in decision_keys) for x in left] == [
            tuple(x[key] for key in decision_keys) for x in right]
        if req in mapping:
            label = mapping[req]
            for leg, rows in ((1, left), (2, right)):
                engine = next(x['engine'] for x in results[leg] if x['label'] == label)
                assert sum(int(x['accepted']) for x in rows) == engine['drafts_accepted']
                assert sum(int(x['offered']) for x in rows) == engine['drafts_offered']
        trace_report[str(req)] = {
            'label': mapping.get(req, 'probe'), 'windows': len(left),
            'widths': dict(sorted(Counter(int(x['T']) for x in left).items())),
            'pipeline': dict(sorted(Counter(int(x['pipeline']) for x in left).items())),
            'chain': dict(sorted(Counter(int(x['chain']) for x in left).items())),
            'accepted': sum(int(x['accepted']) for x in left),
            'offered': sum(int(x['offered']) for x in left),
            'cpu_misses': sum(int(x.get('cpu_misses', 0)) for x in left),
            'vram_hits': sum(int(x.get('vram_hits', 0)) for x in left),
            'pcie_experts': sum(int(x.get('pcie_experts', 0)) for x in left),
        }
    budgets = []
    for leg in legs:
        log = (leg / 'engine.log').read_text()
        budget = {
            'slots': sorted(set(re.findall(r'expert cache (\d+) slots', log))),
            'chunks': sorted(set(re.findall(r'prompt chunk auto: (\d+) tokens', log))),
            'pinned_gib': sorted(set(re.findall(r'allocating ([\d.]+) GiB page-locked cache complement', log))),
        }
        assert all(budget.values())
        budgets.append(budget)
    assert all(value == budgets[0] for value in budgets)
    report = {
        'status': 'complete', 'arms': arms, 'config_only_trace_switch': True,
        'same_text_and_logged_work': True, 'same_trace_decisions_between_candidate_repeats': True,
        'budgets': budgets[0], 'requests': trace_report,
        'timing': {label: {'prefill_ratios': entry['prefill_tok_s_paired_ratios'],
                           'decode_ratios': entry['decode_tok_s_paired_ratios']}
                   for label, entry in summaries.items()},
        'candidate_trace_bytes': [(leg / 'engine.log').stat().st_size for leg in legs[1:3]],
        'inputs': {str(path): digest(path) for path in [root / 'summary.json'] +
                   [leg / 'results.json' for leg in legs] + [leg / 'engine.log' for leg in legs]},
        'limits': 'Two pairs at 4K/16K. Serial windows only here. Per-window counters are complete-window totals; pipeline dispatch requires separate attribution. Timing ratios cannot establish a speed gain for tracing.',
    }
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    main()
