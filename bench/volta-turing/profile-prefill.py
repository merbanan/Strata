#!/usr/bin/env python3
"""Trace one real prefill to establish which CUDA attention kernel executes.

Uses the engine's normal GEN protocol and exits at EOF. Profiled timings are
not serving-performance evidence. Requests come from the saved benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('config', type=Path)
p.add_argument('request', type=Path)
p.add_argument('out', type=Path)
a = p.parse_args()
a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=False)
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise SystemExit('GPU is busy; no trace started')
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
(a.out / 'request.json').write_text(json.dumps(body, indent=2) + '\n')
(a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
(a.out / 'binary.sha256').write_text(hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest() + '\n')
command = ['nsys', 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none', '--wait=all',
           '--output=' + str(a.out / 'prefill'), cfg['exe'], '--serve', *cfg['args']]
env = dict(os.environ, **cfg.get('env', {}))
with (a.out / 'trace.log').open('wb') as log:
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                            env=env, cwd=cfg.get('cwd'), start_new_session=True)
    try:
        proc.communicate(('GEN 1 ' + ','.join(map(str, ids)) + '\n').encode(), timeout=900)
    except BaseException:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        raise
    if proc.returncode: raise SystemExit(f'profiler/engine exited {proc.returncode}; see trace.log')
log = (a.out / 'trace.log').read_text(errors='replace')
if not any(line.startswith('DONE ') for line in log.splitlines()):
    raise SystemExit('engine did not complete the profiled request')
with (a.out / 'kernels.csv').open('w') as output:
    subprocess.run(['nsys', 'stats', '--report=cuda_gpu_kern_sum', '--format=csv',
                    str(a.out / 'prefill.nsys-rep')], stdout=output, stderr=subprocess.STDOUT, check=True)
rows = [line for line in (a.out / 'kernels.csv').read_text().splitlines()
        if 'prompt_attn_k8v4_stage_kernel' in line]
if not rows: raise SystemExit('K8V4 stage kernel absent from the actual GPU trace')
print('Profiled prompt tokens:', len(ids))
print('\n'.join(rows))
