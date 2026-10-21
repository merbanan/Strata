#!/usr/bin/env python3
"""ABBA snapshot timing: separate the first loan from repeated cold requests."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.repeats < 2:
        p.error('at least two repeats are required')
    repo = Path(__file__).resolve().parents[2]
    workspace = repo.parent
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    configs = {arm: json.loads(path.read_text()) for arm, path in [('base', a.base), ('candidate', a.candidate)]}
    comparable = []
    for arm, value in [('base', '0'), ('candidate', '1')]:
        cfg = json.loads(json.dumps(configs[arm]))
        if cfg['env'].pop('STRATA_MTP_EXPERT_SNAPSHOT') != value:
            raise ValueError('incorrect snapshot switch')
        cfg.pop('log', None)
        comparable.append(cfg)
    if comparable[0] != comparable[1]:
        raise ValueError('configurations differ beyond snapshot switch and log')
    binary = Path(configs['base']['exe'])
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 18080)) == 0:
            raise RuntimeError('port 18080 busy')
    sys.path[:0] = [str(repo), str(repo / 'tools')]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    directory = Path(configs['base']['tokenizer'])
    vocab = json.loads((directory / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, number in vocab.items():
        tokens[number] = token
    tokenizer = Tokenizer(tokens, (directory / 'merges.txt').read_text().splitlines(),
                          json.loads((directory / 'token_type.json').read_text()))
    template = ChatTemplate(directory / 'chat_template.jinja')
    def count(body):
        messages, tools, kwargs = openai_to_messages(body)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))
    # Retain the task template qualified by prefill-quality.py. An earlier
    # end-marker-only timing prompt failed on the baseline and is archived.
    filler = '\n'.join(f'Record {i:05d}: item count {i % 91 + 7}; category standard; review complete.' for i in range(2000))
    cases = []
    for repeat in range(a.repeats):
        for target in (2048, 4096):
            marker = f'MORNING-RIVER-{target + 9271 + repeat}'
            def body(n):
                label = f'Qualification case {target}' + (f', repeat {repeat}' if repeat else '')
                text = (label + '. Read these records.\n' + filler[:n // 2] +
                        f'\nThe audit marker is {marker}.\n' + filler[n // 2:n] +
                        '\nReturn only the exact audit marker, with no explanation.')
                return {'model': 'strata', 'messages': [{'role': 'user', 'content': text}],
                        'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': 64,
                        'stream': True, 'stream_options': {'include_usage': True}}
            lo, hi = 0, len(filler)
            while lo < hi:
                middle = (lo + hi + 1) // 2
                if count(body(middle)) <= target:
                    lo = middle
                else:
                    hi = middle - 1
            payload = body(lo)
            actual = count(payload)
            if not target - 20 <= actual <= target:
                raise RuntimeError('could not construct target')
            cases.append({'target': target, 'repeat': repeat, 'tokens': actual,
                          'expected': marker, 'body': payload})
    (a.out / 'requests.json').write_text(json.dumps(cases, indent=2) + '\n')
    manifest = {'binary': str(binary), 'binary_sha256': digest,
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'source': {}}
    (a.out / 'snapshot-compare.py').write_bytes(Path(__file__).read_bytes())
    for name in ('src/core/mtp.cpp', 'include/strata/core/mtp.hpp'):
        saved = a.out / 'source' / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes((repo / name).read_bytes())
        manifest['source'][name] = hashlib.sha256(saved.read_bytes()).hexdigest()
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    budgets = []
    for leg, arm in enumerate(('base', 'candidate', 'candidate', 'base'), 1):
        if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
            raise RuntimeError('binary changed during measurement')
        out = a.out / f'{leg}-{arm}'
        out.mkdir()
        cfg = json.loads(json.dumps(configs[arm]))
        cfg.update(host='127.0.0.1', port=18080, log=str(out / 'engine.log'))
        (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        unit = 'strata-snapshot-' + hashlib.sha256(str(out).encode()).hexdigest()[:12]
        try:
            subprocess.run(['systemd-run', '--user', '--collect', f'--unit={unit}',
                            '-p', f'WorkingDirectory={repo}', '-p', f'StandardOutput=file:{out}/server.log',
                            '-p', f'StandardError=file:{out}/server.log',
                            str(workspace / 'Strata/.venv/bin/python'), str(repo / 'serve/server.py'),
                            '--engine', 'strata', '--config', str(out / 'config.json'),
                            '--host', '127.0.0.1', '--port', '18080'], check=True)
            deadline = time.monotonic() + 900
            while True:
                if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode:
                    raise RuntimeError('server exited during load')
                try:
                    if get('http://127.0.0.1:18080/health').get('loaded'):
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('model load')
                time.sleep(5)
            for index, case in enumerate(cases):
                offset = (out / 'engine.log').stat().st_size
                wire = json.dumps(case['body']).encode()
                started = time.perf_counter()
                first = None
                text = ''
                finish = None
                with urllib.request.urlopen(urllib.request.Request(
                        'http://127.0.0.1:18080/v1/chat/completions', wire,
                        {'Content-Type': 'application/json'}), timeout=600) as response:
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        payload = line[6:].strip()
                        if payload == b'[DONE]':
                            break
                        event = json.loads(payload)
                        for choice in event.get('choices', []):
                            content = choice.get('delta', {}).get('content') or ''
                            if content:
                                if first is None:
                                    first = time.perf_counter() - started
                                text += content
                            if choice.get('finish_reason'):
                                finish = choice['finish_reason']
                elapsed = time.perf_counter() - started
                metrics = get('http://127.0.0.1:18080/metrics')['requests'][0]
                with (out / 'engine.log').open('rb') as f:
                    f.seek(offset)
                    delta_log = f.read().decode(errors='replace')
                loans = re.findall(r'lent in ([\d.]+) ms', delta_log)
                reclaims = re.findall(r'experts are the decode.s again in ([\d.]+) ms', delta_log)
                row = {'leg': leg, 'arm': arm, 'index': index, 'target': case['target'],
                       'repeat': case['repeat'], 'first_loan': index == 0,
                       'request_sha256': hashlib.sha256(wire).hexdigest(), 'text': text,
                       'finish': finish, 'ttft_s': first, 'elapsed_s': elapsed, 'metrics': metrics,
                       'loan_ms': float(loans[0]) if len(loans) == 1 else None,
                       'reclaim_ms': float(reclaims[0]) if len(reclaims) == 1 else None}
                rows.append(row)
                with (a.out / 'runs.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                if text.strip() != case['expected'] or finish != 'stop' or first is None:
                    raise RuntimeError('retrieval/termination failed; see saved row')
                if metrics['prompt_tokens'] != case['tokens'] or metrics.get('reused', 0):
                    raise RuntimeError('token count or prefix reuse differs')
                if row['loan_ms'] is None or row['reclaim_ms'] is None:
                    raise RuntimeError('expected one completed loan and reclaim per request')
                print(json.dumps(row), flush=True)
            log = (out / 'engine.log').read_text()
            budget = {'slots': sorted(set(re.findall(r'expert cache (\d+) slots', log))),
                      'chunks': sorted(set(re.findall(r'prompt chunk auto: (\d+) tokens', log))),
                      'pinned_gib': sorted(set(re.findall(r'allocating ([\d.]+) GiB page-locked cache complement', log)))}
            if not all(budget.values()):
                raise RuntimeError('missing memory-budget evidence')
            budgets.append(budget)
        finally:
            subprocess.run(['systemctl', '--user', 'stop', unit], check=False)
            time.sleep(3)
    if any(b != budgets[0] for b in budgets):
        raise RuntimeError('memory budgets differ')
    summary = {'budgets': budgets, 'first_loan': {}, 'repeated': {}, 'repeated_by_target': {},
               'limits': 'Small retrieval workload; report first-loan allocation separately. No broad decode claim.'}
    for first_loan, label in ((True, 'first_loan'), (False, 'repeated')):
        for metric in ('loan_ms', 'reclaim_ms', 'ttft_s', 'elapsed_s'):
            entry = {}
            for arm in ('base', 'candidate'):
                values = [r[metric] for r in rows if r['arm'] == arm and r['first_loan'] == first_loan]
                entry[arm] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
            entry['paired_deltas'] = [
                next(r[metric] for r in rows if r['leg'] == right and r['index'] == index) -
                next(r[metric] for r in rows if r['leg'] == left and r['index'] == index)
                for left, right in ((1, 2), (4, 3)) for index in range(len(cases))
                if (index == 0) == first_loan]
            summary[label][metric] = entry
    # Do not pool 2K and 4K medians when deciding whether latency improved.
    # The first 2K request is excluded, so the repeated groups differ in size.
    for target in (2048, 4096):
        target_report = {}
        for metric in ('loan_ms', 'reclaim_ms', 'ttft_s', 'elapsed_s'):
            paired = []
            for left, right in ((1, 2), (4, 3)):
                base = [r[metric] for r in rows if r['leg'] == left and r['target'] == target and not r['first_loan']]
                candidate = [r[metric] for r in rows if r['leg'] == right and r['target'] == target and not r['first_loan']]
                b, c = statistics.median(base), statistics.median(candidate)
                paired.append({'base_median': b, 'candidate_median': c,
                               'delta': c - b, 'percent_change': 100 * (c / b - 1) if b else None})
            target_report[metric] = paired
        summary['repeated_by_target'][str(target)] = target_report
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
