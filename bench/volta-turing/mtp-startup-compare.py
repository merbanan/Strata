#!/usr/bin/env python3
"""Measure engine startup and first-request latency for eager MTP snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('base', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('requests', type=Path, help='qualified prefill-quality.py requests.json')
p.add_argument('out', type=Path)
p.add_argument('--target', type=int, default=4096)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=False)
configs = {arm: json.loads(path.read_text()) for arm, path in [('base', a.base), ('candidate', a.candidate)]}
comparable = []
for arm, value in [('base', '0'), ('candidate', '1')]:
    cfg = json.loads(json.dumps(configs[arm]))
    assert cfg['env'].pop('STRATA_MTP_EXPERT_SNAPSHOT_EAGER') == value
    assert cfg['env']['STRATA_MTP_EXPERT_SNAPSHOT'] == '1'
    assert 'STRATA_DUMP_FIRST_LOGITS' not in cfg['env'], 'quality instrumentation is not a timing run'
    cfg.pop('log', None)
    comparable.append(cfg)
assert comparable[0] == comparable[1], 'configuration differences beyond priming switch/log'
case = next(c for c in json.loads(a.requests.read_text()) if c['target'] == a.target)
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
messages, tools, kwargs = openai_to_messages(case['body'])
ids = tokenizer.encode(ChatTemplate(directory / 'chat_template.jinja').render(messages, tools, **kwargs), parse_special=True)
assert len(ids) == case['tokens']
eos = {vocab['<|endoftext|>'], vocab['<|im_end|>']}
assert eos == {248044, 248046}, 'this diagnostic uses the Flash-Next engine EOS defaults'
assert '--eos-ids' not in configs['base']['args'], 'custom EOS needs an explicit comparison policy'
maximum = int(case['body']['max_tokens'])
binary = Path(configs['base']['exe'])
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
manifest = {'binary': str(binary), 'binary_sha256': digest,
            'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sources': {}}
for name in ('src/core/mtp.cpp', 'include/strata/core/mtp.hpp', 'src/program/generate.cpp'):
    dest = a.out / 'source' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes((repo / name).read_bytes())
    manifest['sources'][name] = hashlib.sha256(dest.read_bytes()).hexdigest()
(a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(a.out / 'request.json').write_text(json.dumps(case, indent=2) + '\n')
(a.out / 'mtp-startup-compare.py').write_bytes(Path(__file__).read_bytes())
rows = []
for leg, arm in enumerate(('base', 'candidate', 'candidate', 'base'), 1):
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == digest, 'binary changed'
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU is busy')
    out = a.out / f'{leg}-{arm}'
    out.mkdir()
    cfg = configs[arm]
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    env = dict(os.environ, **cfg.get('env', {}))
    if cfg.get('lib_dirs'):
        env['LD_LIBRARY_PATH'] = ':'.join(cfg['lib_dirs']) + ':' + env.get('LD_LIBRARY_PATH', '')
    started = time.perf_counter()
    proc = subprocess.Popen([cfg['exe'], '--serve', *cfg['args']], cwd=cfg.get('cwd'), env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    ready = first = done_at = None
    done = None
    output_ids = []
    pending = b''
    try:
        with selectors.DefaultSelector() as selector, (out / 'engine.log').open('wb') as log:
            selector.register(proc.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 900
            while selector.get_map():
                if time.monotonic() > deadline:
                    raise TimeoutError('engine protocol timeout')
                for key, _ in selector.select(timeout=1):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(data)
                    log.flush()
                    pending += data
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        now = time.perf_counter()
                        if line.startswith(b'READY ') and ready is None:
                            ready = now
                            proc.stdin.write((f'GEN {maximum} ' + ','.join(map(str, ids)) + '\n').encode())
                            proc.stdin.flush()
                        elif line.startswith(b'T ') and ready is not None:
                            if first is None:
                                first = now
                            output_ids.append(int(line.split()[1]))
                        elif line.startswith(b'DONE ') and ready is not None:
                            done = line.decode().split()
                            done_at = now
                            proc.stdin.close()
                        elif line.startswith((b'ERR ', b'FATAL ')):
                            raise RuntimeError(line.decode(errors='replace'))
            status = proc.wait(timeout=30)
        if status or ready is None or first is None or done_at is None:
            raise RuntimeError(f'incomplete engine run, exit={status}')
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        raise
    assert done[5] == 'stop' and int(done[8]) == 0, done
    assert int(done[1]) == len(output_ids) and int(done[2]) == len(ids), done
    # The engine emits the terminal EOS token; remove only that final token.
    text = tokenizer.decode(output_ids[:-1] if output_ids and output_ids[-1] in eos else output_ids)
    assert text.strip() == case['expected'], (text, case['expected'])
    log = (out / 'engine.log').read_text()
    primes = re.findall(r'primed (\d+) MiB pinned MTP expert snapshot before READY in ([\d.]+) ms', log)
    assert len(primes) == (1 if arm == 'candidate' else 0), 'expected priming path did not execute'
    loans = re.findall(r'lent in ([\d.]+) ms', log)
    reclaims = re.findall(r'experts are the decode.s again in ([\d.]+) ms', log)
    assert len(loans) == len(reclaims) == 1, 'expected one loan/reclaim'
    budget = {'slots': sorted(set(re.findall(r'expert cache (\d+) slots', log))),
              'chunks': sorted(set(re.findall(r'prompt chunk auto: (\d+) tokens', log))),
              'pinned_expert_gib': sorted(set(re.findall(r'allocating ([\d.]+) GiB page-locked cache complement', log)))}
    assert all(budget.values()), 'missing budget evidence'
    row = {'leg': leg, 'arm': arm, 'startup_s': ready - started,
           'ready_to_first_token_s': first - ready, 'ready_to_done_s': done_at - ready,
           'start_to_first_token_s': first - started, 'start_to_done_s': done_at - started,
           'prime_mib': int(primes[0][0]) if primes else 0, 'prime_ms': float(primes[0][1]) if primes else 0,
           'loan_ms': float(loans[0]), 'reclaim_ms': float(reclaims[0]),
           'text': text, 'token_ids': output_ids, 'done': done, 'budget': budget}
    rows.append(row)
    with (a.out / 'runs.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)
    if len(rows) > 1:
        assert rows[-1]['token_ids'] == rows[0]['token_ids'], 'output tokens changed'
        assert rows[-1]['budget'] == rows[0]['budget'], 'memory budgets changed'
    time.sleep(3)
summary = {'pairs': {}, 'limits': 'Direct engine-protocol timing, tokenizer prepared before launch; excludes API server startup. Report moved startup work separately from total first-answer time.'}
for metric in ('startup_s', 'ready_to_first_token_s', 'ready_to_done_s', 'start_to_first_token_s', 'start_to_done_s'):
    summary['pairs'][metric] = [{'base': rows[b][metric], 'candidate': rows[c][metric],
                                'delta': rows[c][metric] - rows[b][metric],
                                'percent': 100 * (rows[c][metric] / rows[b][metric] - 1)}
                               for b, c in ((0, 1), (3, 2))]
(a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
