"""Screen paired ABBA workload timing only when successful work is identical.

The independent workload qualification report must be produced first. A partial
screen is informational; a promotion decision requires all four reviewed legs.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workload', type=Path)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()

    raw = args.workload / 'runs.jsonl'
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    review = json.loads(args.qualification.read_text())
    assert review['saved_text_matches_emitted_tokens'] is True
    assert review['pending_independent_reviews'] == 0
    if args.partial:
        assert len(rows) >= review['requests_reviewed']
        rows = rows[:review['requests_reviewed']]
    else:
        assert review['requests_reviewed'] == len(rows)
        assert review['complete'] is True and len(rows) == review['planned_requests'] == 76

    by_key = {}
    for row in rows:
        key = (row['leg'], row['name'])
        assert key not in by_key
        assert row['arm'] == ('base' if row['leg'] in (1, 4) else 'candidate')
        by_key[key] = row
    qualifications = {}
    for result in review['results']:
        key = (result['leg'], result['name'])
        assert key in by_key and key not in qualifications
        assert result['arm'] == by_key[key]['arm']
        qualifications[key] = result['qualified']
    assert set(qualifications) == set(by_key)

    cases = json.loads((args.workload / 'cases.json').read_text())
    names = [case['name'] for case in cases if case['name'] != 'warmup']
    assert len(names) == 18 and len(set(names)) == 18
    items = []
    for name in names:
        item = {'name': name, 'pairs': [], 'excluded': [], 'policy_pairs': [], 'policy_excluded': []}
        for pair, (base_leg, candidate_leg) in enumerate(((1, 2), (4, 3)), start=1):
            base = by_key.get((base_leg, name))
            candidate = by_key.get((candidate_leg, name))
            if base is None or candidate is None:
                item['excluded'].append({'pair': pair, 'reason': 'pair incomplete'})
                item['policy_excluded'].append({'pair': pair, 'reason': 'pair incomplete'})
                continue
            if not (qualifications[base_leg, name] and qualifications[candidate_leg, name]):
                item['excluded'].append({'pair': pair, 'reason': 'task unqualified'})
                item['policy_excluded'].append({'pair': pair, 'reason': 'task unqualified'})
                continue
            if base['tokens'] != candidate['tokens']:
                item['excluded'].append({'pair': pair, 'reason': 'continuation changed'})
                item['policy_excluded'].append({'pair': pair, 'reason': 'continuation changed'})
                continue
            bd, cd = base['done'], candidate['done']
            assert float(bd[3]) > 0 and float(cd[3]) > 0
            assert base['timing']['decode_tok_s'] > 0 and candidate['timing']['decode_tok_s'] > 0
            result = {
                'pair': pair,
                'decode_candidate_over_base': candidate['timing']['decode_tok_s'] / base['timing']['decode_tok_s'],
                'prompt_processing_candidate_over_base': float(bd[3]) / float(cd[3]),
                'ttft_base_over_candidate': base['timing']['ttft_s'] / candidate['timing']['ttft_s'],
                'wall_base_over_candidate': base['timing']['wall_s'] / candidate['timing']['wall_s'],
            }
            item['policy_pairs'].append(result)
            if bd[:3] + bd[5:] != cd[:3] + cd[5:]:
                item['excluded'].append({'pair': pair, 'reason': 'logged work changed'})
                continue
            item['pairs'].append(result)
        items.append(item)

    def screen(key):
        repeatable = [item for item in items if len(item[key]) == 2]
        label = 'repeatable_exact_work_tasks' if key == 'pairs' else 'repeatable_equal_output_tasks'
        result = {label: len(repeatable)}
        for metric in ('decode_candidate_over_base', 'prompt_processing_candidate_over_base',
                       'ttft_base_over_candidate', 'wall_base_over_candidate'):
            result[metric] = {
                'pair_geometric_means': [geometric_mean([pair[metric] for item in repeatable for pair in item[key]
                                                        if pair['pair'] == index]) for index in (1, 2)],
                'tasks_faster_in_both_pairs': sum(all(pair[metric] > 1 for pair in item[key])
                                                 for item in repeatable),
                'provisional_pair_counts': [sum(any(pair['pair'] == index for pair in item[key])
                                                for item in items) for index in (1, 2)],
                'provisional_pair_geometric_means': [geometric_mean([pair[metric] for item in items
                                                                    for pair in item[key] if pair['pair'] == index])
                                                    for index in (1, 2)],
            }
        return result

    summary = screen('pairs')
    summary['policy_equal_output'] = screen('policy_pairs')
    report = {
        'status': 'partial_qualified_timing_screen' if args.partial else 'complete_qualified_timing_screen',
        'requests': len(rows), 'items': items, 'summary': summary,
        'inputs': {str(path.resolve()): digest(path) for path in
                   (raw, args.workload / 'cases.json', args.qualification)},
        'limits': 'Both screens require independently qualified tasks and identical tokens within each pair. '
                  'The strict screen also requires identical non-timing DONE fields. The policy equal-output screen '
                  'includes adaptation-induced work changes and measures whole-policy effects, not isolated QSA speed. '
                  'A partial screen cannot establish a repeatable full-model speedup. Budget audits and cross-repeat '
                  'continuation checks are separate requirements; no adoption claim.',
    }
    assert not args.output.exists(), args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'requests': len(rows), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
