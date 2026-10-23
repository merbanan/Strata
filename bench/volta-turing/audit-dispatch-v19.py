#!/usr/bin/env python3
"""Audit the eight-leg v19 dispatch screen's work and memory before timing claims."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def require(pattern: str, log: str, label: str) -> str:
    match = re.search(pattern, log)
    if match is None:
        raise ValueError(f'{label}: missing {pattern}')
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('screen', type=Path)
    args = parser.parse_args()
    out = args.screen
    bench = out.parents[1]
    modes = ('default', '0', '4', '8')
    base = json.loads((out / 'default.json').read_text())
    exe = Path(base['exe'])
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    for mode in modes:
        config = json.loads((out / f'{mode}.json').read_text())
        value = config['env'].pop('STRATA_EXP_MODE', None)
        if value != (None if mode == 'default' else mode):
            raise ValueError(f'{mode}: wrong mode switch {value}')
        baseline = json.loads(json.dumps(base))
        baseline['env'].pop('STRATA_EXP_MODE', None)
        if config != baseline:
            raise ValueError(f'{mode}: configuration differs beyond mode')
    report = {'binary_sha256': digest, 'legs': {}, 'warnings': []}
    for mode in modes:
        for repeat in (1, 2):
            label = f'{mode}-{repeat}'
            leg = bench / f'2026-10-06-pr-dispatch-v19-20261023-{label}'
            rows = json.loads((leg / 'results.json').read_text())
            log = (leg / 'engine.log').read_text()
            if len(rows) != 3:
                raise ValueError(f'{label}: expected warmup, 4K and 16K requests')
            cache_slots = int(require(r'expert cache (\d+) slots', log, label))
            prompt_chunk = int(require(r'prompt chunk auto: (\d+) tokens', log, label))
            pinned_gib = float(require(r'allocating ([\d.]+) GiB page-locked cache complement', log, label))
            lends = {int(tokens): float(ms) for tokens, ms in
                     re.findall(r'prompt room: (\d+) tokens to read[^\n]*lent in ([\d.]+) ms', log)}
            requests = {}
            for row in rows[1:]:
                target = int(row['expected_prompt_tokens'])
                e = row['engine']
                if (target not in (4096, 16384) or e['prompt_tokens'] != target or
                    e['reused'] != 0 or e['output_tokens'] != 256 or row['finish_reason'] != 'length'):
                    raise ValueError(f'{label}: request shape or completion mismatch at {target}')
                requests[target] = {
                    'response_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                    'prompt_ms': e['prompt_ms'], 'decode_ms': e['decode_ms'],
                    'drafts_offered': e['drafts_offered'], 'drafts_accepted': e['drafts_accepted'],
                    'ram_blobs': e['ram_blobs'], 'pcie_share': e['pcie_share'],
                    'lend_ms': lends.get(target),
                }
            report['legs'][label] = {'cache_slots': cache_slots, 'prompt_chunk': prompt_chunk,
                                     'pinned_gib': pinned_gib, 'requests': requests,
                                     'low_vram_warning': 'WARNING: only' in log}
            if any(v['lend_ms'] is None for v in requests.values()):
                report['warnings'].append(f'{label}: missing prompt-room timing')
            if any(v['lend_ms'] is not None and v['lend_ms'] > 2000 for v in requests.values()):
                report['warnings'].append(f'{label}: prompt-room lending exceeds 2 seconds')
    for field in ('cache_slots', 'prompt_chunk', 'pinned_gib'):
        values = {leg[field] for leg in report['legs'].values()}
        if len(values) != 1:
            report['warnings'].append(f'{field} differs across legs: {sorted(values)}')
    for target in (4096, 16384):
        values = {leg['requests'][target]['response_sha256'] for leg in report['legs'].values()}
        report[f'{target}_distinct_response_hashes'] = len(values)
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'legs': len(report['legs']), 'warnings': report['warnings'],
                      '4096_distinct_response_hashes': report['4096_distinct_response_hashes'],
                      '16384_distinct_response_hashes': report['16384_distinct_response_hashes']}, indent=2))


if __name__ == '__main__':
    main()
