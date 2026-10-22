#!/usr/bin/env python3
"""Compare observed single-GPU prefill budgets, including per-request loans.

Use STRATA_TRACE=1 in separate diagnostic runs to expose request loans. Matching
startup cache sizes alone cannot prove equal residency during prefill. This
audit does not establish expert identity, numerical equivalence or throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def parse_log(path, expected_requests):
    raw = path.read_bytes()
    text = raw.decode(errors='replace')
    patterns = {
        'expert_slots': r'expert cache (\d+) slots',
        'chunk_and_ring': r'prompt chunk auto: (\d+) tokens, a (\d+)-slot ring',
        'maximum_borrowed_slots': r'the prompt path borrows (\d+) CUDA0 cache slots',
        'pinned_expert_gib': r'allocating ([\d.]+) GiB page-locked cache complement',
        'mtp_room': r'STRATA_MTP_ROOM_MIN=(\d+):[^\n]*drafter\x27s (\d+) MiB[^\n]*\((\d+) slots past its own\)',
    }
    startup = {key: sorted(set(re.findall(pattern, text))) for key, pattern in patterns.items()}
    issues = [f'{key}: expected one observed value, got {len(values)}'
              for key, values in startup.items() if len(values) != 1]
    # Multiple devices or dynamic replanning need a different audit, not a
    # flattened aggregate that happens to match the primary device.
    if re.search(r'CUDA[1-9]\d* prompt path borrows', text):
        issues.append('multiple prefill devices are outside this audit')
    reductions = re.findall(r'prompt chunk \d+ -> \d+ tokens[^\n]*', text)
    if reductions:
        issues.append('chunk reduction observed; actual per-stage chunk needs review')
    requests = []
    current = None
    for line in text.splitlines():
        match = re.search(r'strata trace: request (\d+) (-?\d+)$', line)
        if match:
            current = {'prompt_tokens': int(match[1]), 'mode': int(match[2]),
                       'loans': [], 'prefill_done': False}
            requests.append(current)
            continue
        match = re.search(r'strata trace: lent (\d+) slots for (\d+) tokens in [\d.]+ ms', line)
        if match:
            if current is None:
                issues.append('loan before request boundary')
            else:
                current['loans'].append({'resident_slots_lent': int(match[1]),
                                         'requested_chunk_tokens': int(match[2])})
        if 'strata trace: prompt done (slots refilled)' in line:
            if current is None:
                issues.append('prefill completion before request boundary')
            else:
                current['prefill_done'] = True
    if len(requests) != expected_requests:
        issues.append(f'expected {expected_requests} traced requests, got {len(requests)}')
    for ordinal, request in enumerate(requests, 1):
        if not request['loans'] or not request['prefill_done']:
            issues.append(f'request {ordinal}: loan trace or prefill completion missing')
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
            'startup': startup, 'requests': requests, 'chunk_reductions': reductions,
            'trace_complete': not issues, 'issues': issues}


def audit(paths, expected_requests):
    logs = [parse_log(path, expected_requests) for path in paths]
    base = logs[0]
    comparisons = []
    for candidate in logs[1:]:
        startup_equal = {key: candidate['startup'][key] == values
                         for key, values in base['startup'].items()}
        requests_equal = candidate['requests'] == base['requests']
        comparisons.append({'base': base['path'], 'candidate': candidate['path'],
                            'startup_equal': startup_equal, 'requests_equal': requests_equal,
                            'observed_budgets_match': (base['trace_complete'] and candidate['trace_complete']
                                                      and all(startup_equal.values()) and requests_equal)})
    return {'logs': logs, 'comparisons': comparisons,
            'observed_budgets_match': (bool(comparisons) and
                                      all(c['observed_budgets_match'] for c in comparisons)),
            'limits': 'Single GPU, borrowed buffers and MTP expert-room configuration only. '
                      'Loan counts are observed resident mappings removed, not owned allocation bytes '
                      'or expert identities. Trace timings are excluded. Missing evidence never passes. '
                      'This is not numerical, task-quality or throughput qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    parser.add_argument('--expected-requests', type=int, default=4)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if len(args.logs) < 2 or args.expected_requests < 1:
        parser.error('at least two logs and a positive expected request count are required')
    report = audit(args.logs, args.expected_requests)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'observed_budgets_match': report['observed_budgets_match'],
                      'issues': {log['path']: log['issues'] for log in report['logs']},
                      'comparisons': report['comparisons']}, indent=2))
    return 0 if report['observed_budgets_match'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
