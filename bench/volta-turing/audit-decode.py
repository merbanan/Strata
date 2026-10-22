#!/usr/bin/env python3
"""Audit saved decode ABBA outputs, acceptance and budgets without running a model.

Can snapshot an incomplete run. Task success, output parity and timing are
separate findings; a fast failed task never becomes a successful-task result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, help='directory with independent code/prose review JSON')
    args = parser.parse_args()
    raw = (args.run / 'runs.jsonl').read_bytes()
    # A writer can be midway through its final line. Never count a partial row.
    complete = raw.rsplit(b'\n', 1)[0] if not raw.endswith(b'\n') else raw
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    indexed = {(row['leg'], row['name']): row for row in rows}
    assert len(indexed) == len(rows), 'duplicate leg/task records'
    cases = json.loads((args.run / 'cases.json').read_text())
    names = [case['name'] for case in cases]
    arms = {1: 'base', 2: 'candidate', 3: 'candidate', 4: 'base'}
    assert all(row['arm'] == arms[row['leg']] and row['name'] in names for row in rows)
    reviews = {}
    if args.reviews:
        for filename in ('fresh-code-independent-review.json', 'prose-independent-review.json'):
            path = args.reviews / filename
            if path.exists():
                review = json.loads(path.read_text())
                reviews[review.get('source_sha256', review.get('sha256'))] = review
    report = {'complete': len(rows) == len(names) * 4,
              'rows': len(rows), 'expected_rows': len(names) * 4,
              'runs_snapshot_sha256': hashlib.sha256(complete).hexdigest(),
              'tasks': {}, 'legs': {},
              'limits': 'Partial pairs are provisional. Failed tasks are diagnostic timings only. '
                        'Matching acceptance and budgets do not rule out thermal, memory or I/O confounding.'}
    for name in names:
        selected = [indexed[(leg, name)] for leg in arms if (leg, name) in indexed]
        if not selected:
            continue
        qualifications = []
        for row in selected:
            assert int(row['done'][1]) == len(row['tokens'])
            text = args.run / f"{row['leg']}-{row['arm']}" / f'{name}.txt'
            review = reviews.get(hashlib.sha256(text.read_bytes()).hexdigest())
            qualifications.append(bool(review['qualified'] if review else row['qualified_task'])
                                  and row['done'][5] == 'stop')
        task = {'completed_legs': [row['leg'] for row in selected],
                'output_parity_so_far': (all(row['tokens'] == selected[0]['tokens'] for row in selected)
                                        if len(selected) > 1 else None),
                'qualified_so_far': all(qualifications), 'pairs': [], 'within_arm_repeats': {}}
        for arm, first, second in [('base', 1, 4), ('candidate', 2, 3)]:
            if (first, name) not in indexed or (second, name) not in indexed:
                task['within_arm_repeats'][arm] = None
                continue
            a, b = indexed[(first, name)], indexed[(second, name)]
            task['within_arm_repeats'][arm] = {
                'tokens_identical': a['tokens'] == b['tokens'],
                'acceptance_identical': a['done'][6:8] == b['done'][6:8],
                'prefix_reuse_identical': a['done'][8] == b['done'][8],
                'request_identical': a['request_sha256'] == b['request_sha256']}
        for base, candidate in ((1, 2), (4, 3)):
            if (base, name) not in indexed or (candidate, name) not in indexed:
                continue
            a, b = indexed[(base, name)], indexed[(candidate, name)]
            same = {label: a['done'][index] == b['done'][index]
                    for label, index in [('generated', 1), ('prompt', 2), ('finish', 5),
                                         ('accepted', 6), ('offered', 7), ('prefix_reuse', 8)]}
            mismatch = next((i for i, (x, y) in enumerate(zip(a['tokens'], b['tokens'])) if x != y), None)
            if mismatch is None and len(a['tokens']) != len(b['tokens']):
                mismatch = min(len(a['tokens']), len(b['tokens']))
            task['pairs'].append({'base_leg': base, 'candidate_leg': candidate,
                                  'first_token_mismatch': mismatch, 'matched': same,
                                  'acceptance': {'base': a['done'][6:8], 'candidate': b['done'][6:8]},
                                  'timing': {metric: {'base': a['timing'][metric], 'candidate': b['timing'][metric],
                                              'candidate_change_percent': 100 * (b['timing'][metric] / a['timing'][metric] - 1)}
                                             for metric in ('ttft_s', 'wall_s', 'decode_tok_s')}})
        report['tasks'][name] = task
    for leg, arm in arms.items():
        directory = args.run / f'{leg}-{arm}'
        if not directory.exists():
            continue
        log = (directory / 'engine.log').read_text(errors='replace')
        free = [int(n) for n in re.findall(r'(\d+) MiB of VRAM free', log)]
        record = {'minimum_logged_free_mib': min(free) if free else None,
                  'captured_windows': re.findall(r'captured the (\d+)-token window', log),
                  'budget_complete': (directory / 'budget.json').exists()}
        if record['budget_complete']:
            record['budget'] = json.loads((directory / 'budget.json').read_text())
        report['legs'][leg] = record
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': report['complete'], 'rows': len(rows),
                      'tasks': {name: {k: v for k, v in value.items() if k != 'pairs'}
                                for name, value in report['tasks'].items()}}, indent=2))


if __name__ == '__main__':
    main()
