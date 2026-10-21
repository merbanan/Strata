#!/usr/bin/env python3
"""ABBA prefill qualification: retrieval tasks and full first-token distributions.

This measures first-token distributions after batched prefill, not teacher-forced
KL of whole continuations. The older --short-read KL harness would bypass the
prompt-attention kernel being tested. No speed claims use these instrumented runs.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1200 if body else 3) as response:
        return json.load(response)


def compare(a, b):
    def read(path):
        v = array.array('f'); v.frombytes(path.read_bytes())
        if not v or not all(math.isfinite(x) for x in v):
            raise ValueError(f'empty/nonfinite logits: {path}')
        return v
    x, y = read(a), read(b)
    assert len(x) == len(y)
    def distribution(v):
        maximum = max(v)
        z = maximum + math.log(sum(math.exp(t - maximum) for t in v))
        return [t - z for t in v]
    lx, ly = distribution(x), distribution(y)
    topx = sorted(range(len(x)), key=x.__getitem__, reverse=True)[:10]
    topy = sorted(range(len(y)), key=y.__getitem__, reverse=True)[:10]
    return {'vocab': len(x), 'kl': sum(math.exp(p) * (p - q) for p, q in zip(lx, ly)),
            'top1_equal': topx[0] == topy[0], 'top10_overlap': len(set(topx) & set(topy)),
            'max_logit_abs_diff': max(abs(p - q) for p, q in zip(x, y))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path); p.add_argument('candidate', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--targets', default='1024,4096,16384,32768')
    p.add_argument('--arms', default='base,candidate,candidate,base',
                   help='run order; candidate alone can check an isolated release build')
    p.add_argument('--reference-leg', type=Path,
                   help='also compare logits with a completed earlier leg using identical requests')
    args = p.parse_args()
    arms = args.arms.split(',')
    if not arms or any(arm not in ('base', 'candidate') for arm in arms):
        p.error('--arms must contain base and/or candidate')
    repo = Path(__file__).resolve().parents[2]; workspace = repo.parent
    args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=False)
    # Never attach a quality request to an unrelated existing service.
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if apps.strip(): raise RuntimeError('GPU is busy')
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 18080)) == 0: raise RuntimeError('port 18080 is busy')
    cfgs = {arm: json.loads(path.read_text()) for arm, path in [('base', args.base), ('candidate', args.candidate)]}
    sys.path[:0] = [str(repo), str(repo / 'tools')]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate, openai_to_messages
    directory = Path(cfgs['base']['tokenizer'])
    vocab = json.loads((directory / 'vocab.json').read_text()); tokens = [None] * len(vocab)
    for token, number in vocab.items(): tokens[number] = token
    tokenizer = Tokenizer(tokens, (directory / 'merges.txt').read_text().splitlines(),
                          json.loads((directory / 'token_type.json').read_text()))
    template = ChatTemplate(directory / 'chat_template.jinja')
    def count(body):
        messages, tools, kwargs = openai_to_messages(body)
        return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))
    filler = '\n'.join(f'Record {i:05d}: item count {i % 91 + 7}; category standard; review complete.' for i in range(14000))
    requests = []
    for target in map(int, args.targets.split(',')):
        marker = f'MORNING-RIVER-{target + 9271}'
        def body(n):
            content = (f'Qualification case {target}. Read these records.\n' + filler[:n // 2] +
                       f'\nThe audit marker is {marker}.\n' + filler[n // 2:n] +
                       '\nReturn only the exact audit marker, with no explanation.')
            return {'messages': [{'role': 'user', 'content': content}], 'temperature': 0,
                    'reasoning_effort': 'none', 'max_tokens': 64}
        lo, hi = 0, len(filler)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(body(mid)) <= target: lo = mid
            else: hi = mid - 1
        payload = body(lo); actual = count(payload)
        assert target - 20 <= actual <= target
        requests.append({'target': target, 'tokens': actual, 'expected': marker, 'body': payload})
    (args.out / 'requests.json').write_text(json.dumps(requests, indent=2) + '\n')
    if args.reference_leg:
        reference_requests = json.loads((args.reference_leg.parent / 'requests.json').read_text())
        if requests != reference_requests:
            raise ValueError('reference requests differ; logits would not be comparable')
    legs = []
    for ordinal, arm in enumerate(arms, 1):
        directory = args.out / f'{ordinal}-{arm}'; directory.mkdir()
        cfg = json.loads(json.dumps(cfgs[arm]))
        cfg.update(port=18080, host='127.0.0.1', log=str(directory / 'engine.log'))
        cfg['env']['STRATA_DUMP_FIRST_LOGITS'] = str(directory / 'first')
        (directory / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        (directory / 'binary.sha256').write_text(hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest() + '\n')
        unit = 'strata-quality-' + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
        try:
            subprocess.run(['systemd-run', '--user', '--collect', f'--unit={unit}',
                '-p', f'WorkingDirectory={repo}', '-p', f'StandardOutput=file:{directory}/server.log',
                '-p', f'StandardError=file:{directory}/server.log',
                str(workspace / 'Strata/.venv/bin/python'), str(repo / 'serve/server.py'),
                '--engine', 'strata', '--config', str(directory / 'config.json'),
                '--port', '18080', '--host', '127.0.0.1'], check=True)
            deadline = time.monotonic() + 900
            while True:
                if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode:
                    raise RuntimeError(f'server exited: {directory}')
                try:
                    if request('http://127.0.0.1:18080/v1/status').get('loaded'): break
                except (OSError, ValueError): pass
                if time.monotonic() > deadline: raise TimeoutError(f'model load: {directory}')
                time.sleep(5)
            rows = []
            for i, case in enumerate(requests):
                response = request('http://127.0.0.1:18080/v1/chat/completions', case['body'])
                (directory / f'response-{i}.json').write_text(json.dumps(response, indent=2) + '\n')
                choice = response['choices'][0]; text = choice['message'].get('content') or ''
                metrics = request('http://127.0.0.1:18080/metrics')['requests'][0]
                assert metrics['prompt_tokens'] == case['tokens'], metrics
                assert (directory / f'first.{i}').is_file(), 'first logits missing'
                row = {'target': case['target'], 'expected': case['expected'], 'text': text,
                       'finish': choice['finish_reason'], 'metrics': metrics,
                       'passed': text.strip() == case['expected'] and choice['finish_reason'] == 'stop'}
                rows.append(row)
                (directory / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(ordinal, arm, json.dumps(row), flush=True)
            legs.append({'arm': arm, 'directory': str(directory), 'rows': rows})
        finally:
            subprocess.run(['systemctl', '--user', 'stop', unit], check=False)
            time.sleep(3)
    comparisons = {}
    pairs = ((0, 1), (3, 2), (0, 3), (1, 2)) if arms == ['base', 'candidate', 'candidate', 'base'] else (
        (left, right) for left in range(len(legs)) for right in range(left + 1, len(legs)))
    for left, right in pairs:
        comparisons[f'{left+1}-vs-{right+1}'] = [compare(Path(legs[left]['directory']) / f'first.{i}',
                                                       Path(legs[right]['directory']) / f'first.{i}')
                                                for i in range(len(requests))]
    if args.reference_leg:
        for ordinal, leg in enumerate(legs, 1):
            comparisons[f'reference-vs-{ordinal}'] = [compare(args.reference_leg / f'first.{i}',
                                                             Path(leg['directory']) / f'first.{i}')
                                                     for i in range(len(requests))]
    summary = {'legs': legs, 'comparisons': comparisons,
               'limits': 'First-token distributions and four retrieval tasks; not full-continuation KL or broad model quality.'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2), flush=True)
    return 0 if all(row['passed'] for leg in legs for row in leg['rows']) else 1

if __name__ == '__main__':
    raise SystemExit(main())
