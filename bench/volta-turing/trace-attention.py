#!/usr/bin/env python3
"""Prove attention dispatch on a completed model request using a launch interposer.

This is a Linux diagnostic, not a GPU profile or a serving-performance test.
Unlike a capture-only log, success requires direct launches plus protocol DONE.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('config', type=Path)
p.add_argument('request', type=Path)
p.add_argument('interposer', type=Path)
p.add_argument('out', type=Path)
a = p.parse_args()
a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=False)
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise SystemExit('GPU busy')
repo = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(repo), str(repo / 'tools')]
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate, openai_to_messages
cfg = json.loads(a.config.read_text()); body = json.loads(a.request.read_text())
directory = Path(cfg['tokenizer'])
vocab = json.loads((directory / 'vocab.json').read_text()); tokens = [None] * len(vocab)
for token, number in vocab.items(): tokens[number] = token
tokenizer = Tokenizer(tokens, (directory / 'merges.txt').read_text().splitlines(),
                      json.loads((directory / 'token_type.json').read_text()))
messages, tools, kwargs = openai_to_messages(body)
ids = tokenizer.encode(ChatTemplate(directory / 'chat_template.jinja').render(messages, tools, **kwargs), parse_special=True)
max_new = int(body.get('max_tokens') or 1)
if max_new <= 0: raise ValueError('trace request needs a positive output allowance')
for name, value in [('config', cfg), ('request', body)]:
    (a.out / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
(a.out / 'binary.sha256').write_text(hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest() + '\n')
(a.out / 'interposer.sha256').write_text(hashlib.sha256(a.interposer.read_bytes()).hexdigest() + '\n')
env = dict(os.environ, **cfg.get('env', {}))
env['LD_PRELOAD'] = str(a.interposer.resolve()) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
env['STRATA_ATTENTION_LAUNCH_LOG'] = str(a.out / 'launches.tsv')
proc = subprocess.Popen([cfg['exe'], '--serve', *cfg['args']], env=env, cwd=cfg.get('cwd'),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        start_new_session=True)
sent = done = False
pending = b''
try:
    with selectors.DefaultSelector() as selector, (a.out / 'engine.log').open('wb') as log:
        selector.register(proc.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 900
        while selector.get_map():
            if time.monotonic() > deadline: raise TimeoutError('engine protocol timeout')
            for key, _ in selector.select(timeout=1):
                data = os.read(key.fd, 65536)
                if not data:
                    selector.unregister(key.fileobj); continue
                log.write(data); log.flush(); pending += data
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if line.startswith(b'READY ') and not sent:
                        proc.stdin.write((f'GEN {max_new} ' + ','.join(map(str, ids)) + '\n').encode())
                        proc.stdin.flush(); sent = True
                    if line.startswith(b'DONE ') and sent:
                        done = True; proc.stdin.close()
        status = proc.wait(timeout=30)
    if status or not done: raise RuntimeError(f'engine exit {status}; request completed={done}')
except BaseException:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    raise
rows = [line.split('\t') for line in (a.out / 'launches.tsv').read_text().splitlines()]
direct = [r for r in rows if 'prompt_attn_k8v4_stage_kernel' in r[0] and r[1:4] == ['0', '0', '0']]
if not direct: raise SystemExit('no successful non-captured K8V4 stage launches')
summary = {'prompt_tokens': len(ids), 'max_new': max_new, 'completed': done, 'direct_stage_launches': len(direct),
           'grid_shapes': sorted({tuple(map(int, r[4:7])) for r in direct}),
           'dynamic_shared_bytes': sorted({int(r[10]) for r in direct}),
           'limits': 'Runtime launch records plus completed request; no achieved-occupancy or GPU-time measurement.'}
(a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
