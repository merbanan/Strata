#!/usr/bin/env python3
"""Review full-workload budgets, including proven window-only short prompts.

The retrieval auditor requires every request to lend prompt buffers. Full task
suites also contain short prompts read entirely through verifier windows.
Those may omit a loan only with complete read/completion evidence. This does
not relax the existing retrieval auditor or establish expert identity/quality.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def window_only(segment, prompt_tokens, reused):
    """Require positive evidence that all prefill tokens used verifier windows."""
    parts = segment.split('strata trace: prompt done (slots refilled)')
    if len(parts) != 2:
        return False
    prefill = parts[0]
    if 'strata trace: lent ' in segment or 'strata trace: prompt chunk ' in prefill:
        return False
    reads = re.findall(r'^strata trace: read (\d+) tokens \((windows|batched)\) in [\d.]+ ms$', prefill, re.M)
    resumes = re.findall(r'^RESUME (\d+)$', prefill, re.M)
    return (resumes == [str(reused)] and bool(reads)
            and all(kind == 'windows' and int(n) > 0 for n, kind in reads)
            and sum(int(n) for n, _ in reads) == prompt_tokens - 1 - reused)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workload', type=Path)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve prior audit'
    inputs = {}
    def read(path):
        data = path.read_bytes(); inputs[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data
    qualification = json.loads(read(args.review))
    assert qualification['complete'] and qualification['requests_reviewed'] == 76
    assert qualification['saved_text_matches_emitted_tokens']
    auditor_path = Path(__file__).with_name('audit-prefill-allocation.py'); read(auditor_path)
    spec = importlib.util.spec_from_file_location('allocation', auditor_path)
    auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
    records = []
    for leg, arm in enumerate(('base', 'candidate', 'candidate', 'base'), 1):
        path = args.workload / f'{leg}-{arm}' / 'engine.log'
        raw = read(path)
        assert qualification['inputs'][str(path.resolve())] == dict(
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), prefix_snapshot=False)
        parsed = auditor.parse_log(path, 19)
        segments = re.split(r'^strata trace: request \d+ -?\d+\n', raw.decode(), flags=re.M)[1:]
        rows = [row for row in qualification['results'] if row['leg'] == leg]
        assert len(segments) == len(rows) == len(parsed['requests']) == 19
        proven_windows = []
        for ordinal, (segment, row, request) in enumerate(zip(segments, rows, parsed['requests']), 1):
            assert row['arm'] == arm and request['mode'] == 0
            assert request['prompt_tokens'] == row['prompt_tokens']
            if not request['loans'] and window_only(segment, row['prompt_tokens'], row['prompt_tokens_reused']):
                assert request['prefill_done']
                issue = f'request {ordinal}: loan trace or prefill completion missing'
                assert issue in parsed['issues']; parsed['issues'].remove(issue)
                proven_windows.append(ordinal)
        parsed['trace_complete'] = not parsed['issues']
        parsed['proven_window_only_requests'] = proven_windows
        records.append(parsed)
    base = records[0]
    matched = all(item['trace_complete'] and item['startup'] == base['startup']
                  and item['requests'] == base['requests']
                  and item['proven_window_only_requests'] == base['proven_window_only_requests']
                  for item in records)
    read(Path(__file__))
    report = dict(status='workload_allocations_reviewed', observed_budgets_match=matched,
        requests=76, logs=records, inputs=inputs, adopted=False,
        limits='Single-GPU startup budgets and per-request borrowed mappings only. Window-only requests require complete read counts, prefix position and refill completion. Does not establish expert identities, byte ownership, numerical equality, quality or throughput.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(observed_budgets_match=matched, issues=[r['issues'] for r in records],
                         window_only_requests_by_leg=[r['proven_window_only_requests'] for r in records]), indent=2))
    return 0 if matched else 2


if __name__ == '__main__':
    raise SystemExit(main())
