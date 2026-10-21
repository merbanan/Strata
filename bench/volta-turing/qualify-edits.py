#!/usr/bin/env python3
"""Diagnose the original edit workload without hiding failed transformations.

Run on an already-started isolated server. Keep source/task unchanged; the
optional instruction-first variant tests prompt placement, not kernel quality.
This client never executes generated Python.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('url')
p.add_argument('out', type=Path)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--instruction-first', action='store_true')
p.add_argument('--max-tokens', type=int, default=4096)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
source = a.source.read_text()
spec = importlib.util.spec_from_file_location('lookup_checks', Path(__file__).with_name('analyze-lookup.py'))
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
commands = {
    'edit-rename': 'Rename the class GGUFFile to GgufReader everywhere (definition and every use). '
                   'Output the complete modified file in one code block, nothing else.',
    'edit-log': 'Add `import logging` at the top and a `log = logging.getLogger(__name__)` line after the imports; '
                'change nothing else. Output the complete modified file in one code block, nothing else.',
}
rows = []
for name, command in commands.items():
    file_text = 'Here is a Python file:\n\n```python\n' + source + '\n```'
    prompt = command + '\n\n' + file_text if a.instruction_first else file_text + '\n\n' + command
    body = {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
            'max_tokens': a.max_tokens, 'chat_template_kwargs': {'enable_thinking': False}}
    data = json.dumps(body).encode()
    (a.out / f'{name}-request.json').write_bytes(data + b'\n')
    request = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions', data,
                                     {'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=1200) as response:
        raw = json.load(response)
    (a.out / f'{name}-response.json').write_text(json.dumps(raw, indent=2) + '\n')
    choice = raw['choices'][0]
    text = choice['message'].get('content') or ''
    (a.out / f'{name}.txt').write_text(text)
    result = checks.check_output(name, text, source)
    row = {'probe': name, 'request_sha256': hashlib.sha256(data).hexdigest(),
           'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
           'instruction_first': a.instruction_first, 'max_tokens': a.max_tokens,
           'finish_reason': choice['finish_reason'], 'usage': raw.get('usage'),
           'wall_s': time.monotonic() - start, 'checks': result,
           'passed': bool(result.get('requested_edit_only_ast')) and choice['finish_reason'] == 'stop'}
    rows.append(row)
    (a.out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(row), flush=True)
# A failing task is evidence; it must also fail a qualification pipeline.
raise SystemExit(0 if all(row['passed'] for row in rows) else 1)
