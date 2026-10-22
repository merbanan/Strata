#!/usr/bin/env python3
"""Direct-engine ABBA: retain full continuation tokens and task failures.

Original lookup tasks are retained. Instruction-first edit variants are named
separately. Generated code is parsed, never executed by this GPU runner.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Engine:
    def __init__(self, config, directory):
        env = dict(os.environ, **config.get('env', {}))
        if config.get('lib_dirs'):
            env['LD_LIBRARY_PATH'] = ':'.join(config['lib_dirs']) + ':' + env.get('LD_LIBRARY_PATH', '')
        self.process = subprocess.Popen([config['exe'], '--serve', *config['args']], cwd=config.get('cwd'),
                                        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.log = (directory / 'engine.log').open('wb')
        self.pending = b''

    def line(self, deadline):
        while b'\n' not in self.pending:
            if time.monotonic() > deadline:
                raise TimeoutError('engine protocol deadline')
            for key, _ in self.selector.select(timeout=1):
                data = os.read(key.fd, 65536)
                if not data:
                    raise RuntimeError(f'engine stdout closed, exit={self.process.poll()}')
                self.pending += data
                self.log.write(data)
                self.log.flush()
        line, self.pending = self.pending.split(b'\n', 1)
        if line.startswith((b'ERR ', b'FATAL ')):
            raise RuntimeError(line.decode(errors='replace'))
        return line

    def ready(self):
        deadline = time.monotonic() + 900
        while not self.line(deadline).startswith(b'READY '):
            pass

    def generate(self, ids, maximum):
        start = time.perf_counter()
        self.process.stdin.write((f'GEN {maximum} ' + ','.join(map(str, ids)) + '\n').encode())
        self.process.stdin.flush()
        first = None
        output = []
        deadline = time.monotonic() + 1200
        while True:
            line = self.line(deadline)
            now = time.perf_counter()
            if line.startswith(b'T '):
                if first is None:
                    first = now
                output.append(int(line.split()[1]))
            elif line.startswith(b'DONE '):
                done = line.decode().split()
                assert first is not None and int(done[1]) == len(output) and int(done[2]) == len(ids), done
                assert done[5] in ('stop', 'length'), done
                return output, done, {'ttft_s': first - start, 'wall_s': now - start,
                                      'decode_ms': float(done[4]),
                                      'decode_tok_s': int(done[1]) * 1000 / float(done[4])}

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write(b'QUIT\n')
                self.process.stdin.flush()
                self.process.stdin.close()
                # Drain logs so a full stdout pipe cannot prevent clean exit.
                deadline = time.monotonic() + 30
                while self.selector.get_map() and time.monotonic() < deadline:
                    for key, _ in self.selector.select(timeout=1):
                        data = os.read(key.fd, 65536)
                        if data:
                            self.log.write(data)
                        else:
                            self.selector.unregister(key.fileobj)
                self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        finally:
            self.log.close()
            self.selector.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--source', type=Path, required=True, help='original lookup src_edit.py')
    p.add_argument('--warmup', type=Path, required=True, help='qualified retrieval requests.json')
    p.add_argument('--arms', default='base,candidate,candidate,base')
    p.add_argument('--feature-env', default='STRATA_ROUTE_Q8_FUSE',
                   help='the only environment setting allowed to differ between arms')
    p.add_argument('--base-value', default='0')
    p.add_argument('--candidate-value', default='1')
    p.add_argument('--validate-configs-only', action='store_true',
                   help='check the arm configurations and exit without GPU work or output creation')
    a = p.parse_args()
    arms = a.arms.split(',')
    if not arms or any(arm not in ('base', 'candidate') for arm in arms):
        p.error('invalid arms')
    if a.base_value == a.candidate_value:
        p.error('feature values must differ')
    repo = Path(__file__).resolve().parents[2]
    a.out = a.out.resolve()
    configs = {name: json.loads(path.read_text()) for name, path in [('base', a.base), ('candidate', a.candidate)]}
    normalized = []
    for arm, value in [('base', a.base_value), ('candidate', a.candidate_value)]:
        cfg = json.loads(json.dumps(configs[arm]))
        assert cfg['env'].pop(a.feature_env) == value, f'{arm}: unexpected {a.feature_env}'
        assert not any(name in cfg['env'] for name in ('LD_PRELOAD', 'STRATA_DUMP_FIRST_LOGITS',
                                                      'STRATA_ROUTE_Q8_LAUNCH_LOG', 'STRATA_GRAPH_MEMORY_LOG'))
        cfg.pop('log', None)
        normalized.append(cfg)
    assert normalized[0] == normalized[1], 'configuration mismatch beyond selected feature switch'
    if a.validate_configs_only:
        print(f'PASS configuration comparison: {a.feature_env}={a.base_value}/{a.candidate_value}')
        return 0
    a.out.mkdir(parents=True, exist_ok=False)
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
    eos = {vocab['<|endoftext|>'], vocab['<|im_end|>']}
    assert eos == {248044, 248046} and '--eos-ids' not in configs['base']['args']
    source = a.source.read_text()
    file_text = 'Here is a Python file:\n\n```python\n' + source + '\n```'
    edits = {
        'edit-rename': 'Rename the class GGUFFile to GgufReader everywhere (definition and every use). '
                       'Output the complete modified file in one code block, nothing else.',
        'edit-log': 'Add `import logging` at the top and a `log = logging.getLogger(__name__)` line after the imports; '
                    'change nothing else. Output the complete modified file in one code block, nothing else.',
    }
    prompts = [(name, name, file_text + '\n\n' + instruction, 2600) for name, instruction in edits.items()]
    prompts += [
        ('fresh-code', 'fresh-code', 'Write a Python module implementing a thread-safe LRU cache class with a TTL per entry, '
         'get/put/delete, and a pytest test file for it. Code only.', 2600),
        ('prose', 'prose', 'Explain in about 400 words how TCP congestion control works (slow start, congestion avoidance, fast '
         'retransmit, fast recovery), for a student.', 2600),
    ]
    prompts += [(name + '-instruction-first', name, instruction + '\n\n' + file_text, 4096)
                for name, instruction in edits.items()]
    warmup = next(c for c in json.loads(a.warmup.read_text()) if c['target'] == 1024)
    cases = [{'name': 'warmup', 'checker': 'marker', 'body': warmup['body'], 'expected': warmup['expected']}]
    cases += [{'name': name, 'checker': checker, 'body': {'messages': [{'role': 'user', 'content': prompt}],
               'temperature': 0, 'max_tokens': limit, 'chat_template_kwargs': {'enable_thinking': False}}}
              for name, checker, prompt, limit in prompts]
    for case in cases:
        messages, tools, kwargs = openai_to_messages(case['body'])
        case['ids'] = tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True)
        case['request_sha256'] = hashlib.sha256(json.dumps(case['body'], sort_keys=True).encode()).hexdigest()
    (a.out / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (a.out / 'src_edit.py').write_text(source)
    spec = importlib.util.spec_from_file_location('lookup_checks', Path(__file__).with_name('analyze-lookup.py'))
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    manifest = {'binary': configs['base']['exe'], 'binary_sha256': digest(configs['base']['exe']),
                'source_sha256': digest(a.source), 'harness_sha256': digest(__file__),
                'feature_env': a.feature_env, 'base_value': a.base_value, 'candidate_value': a.candidate_value,
                'scope': 'Direct engine; original lookup tasks plus separately named instruction-first edits. No generated code execution.'}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (a.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows = []
    budgets = []
    for leg, arm in enumerate(arms, 1):
        assert digest(configs[arm]['exe']) == manifest['binary_sha256'], 'binary changed'
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU busy'
        out = a.out / f'{leg}-{arm}'
        out.mkdir()
        (out / 'config.json').write_text(json.dumps(configs[arm], indent=2) + '\n')
        engine = Engine(configs[arm], out)
        try:
            engine.ready()
            for case in cases:
                output, done, timing = engine.generate(case['ids'], case['body']['max_tokens'])
                text = tokenizer.decode(output[:-1] if output and output[-1] in eos else output)
                if case['checker'] == 'marker':
                    result = {'exact_marker': text.strip() == case['expected']}
                    qualified = result['exact_marker'] and done[5] == 'stop'
                    assert qualified, 'warmup task failed'
                else:
                    result = checks.check_output(case['checker'], text, source)
                    qualified = bool(result.get('requested_edit_only_ast')) and done[5] == 'stop'
                    if qualified:
                        # Parsing alone accepts misplaced future imports. Compile
                        # without executing to check that the edited module is valid.
                        block = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
                        code = block[0] if block else text
                        try:
                            compile(code, '<generated-edit>', 'exec')
                            result['compiles_without_execution'] = True
                            if case['checker'] == 'edit-log':
                                nodes = ast.parse(code).body
                                last_import = max(i for i, node in enumerate(nodes)
                                                  if isinstance(node, (ast.Import, ast.ImportFrom)))
                                wanted = ast.dump(ast.parse('log = logging.getLogger(__name__)').body[0], include_attributes=False)
                                result['logger_after_imports'] = (last_import + 1 < len(nodes) and
                                    ast.dump(nodes[last_import + 1], include_attributes=False) == wanted)
                                qualified = result['logger_after_imports']
                        except SyntaxError as error:
                            result.update(compiles_without_execution=False, compile_error=str(error))
                            qualified = False
                    if case['checker'] in ('fresh-code', 'prose'):
                        result['requires_independent_review'] = True
                row = {'leg': leg, 'arm': arm, 'name': case['name'], 'request_sha256': case['request_sha256'],
                       'done': done, 'tokens': output, 'timing': timing, 'checks': result, 'qualified_task': qualified}
                (out / (case['name'] + '.txt')).write_text(text)
                (out / (case['name'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
                rows.append(row)
                with (a.out / 'runs.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(json.dumps({k: v for k, v in row.items() if k != 'tokens'}), flush=True)
        finally:
            engine.close()
        assert engine.process.returncode == 0, 'engine did not exit cleanly'
        log = (out / 'engine.log').read_text()
        budget = {'slots': sorted(set(re.findall(r'expert cache (\d+) slots', log))),
                  'chunks': sorted(set(re.findall(r'prompt chunk auto: (\d+) tokens', log))),
                  'pinned_expert_gib': sorted(set(re.findall(r'allocating ([\d.]+) GiB page-locked cache complement', log)))}
        assert all(budget.values()), 'missing memory budget evidence'
        budgets.append(budget)
        assert budget == budgets[0], 'memory budgets changed'
        (out / 'budget.json').write_text(json.dumps(budget, indent=2) + '\n')
        time.sleep(3)
    result = {'tasks': {}, 'budgets': budgets,
              'limits': 'Timing includes baseline task failures; only qualified tasks support successful-task speed claims. '
                        'Fresh code and prose require independent review. Exact continuation equality is a separate gate.'}
    for case in cases[1:]:
        selected = [r for r in rows if r['name'] == case['name']]
        entry = {'all_tokens_identical': all(r['tokens'] == selected[0]['tokens'] for r in selected),
                 'matched_prefix_reuse': all(r['done'][8] == selected[0]['done'][8] for r in selected),
                 'all_tasks_qualified': all(r['qualified_task'] for r in selected)}
        if arms == ['base', 'candidate', 'candidate', 'base']:
            entry['pairs'] = {}
            for metric in ('ttft_s', 'wall_s', 'decode_ms', 'decode_tok_s'):
                entry['pairs'][metric] = [{'base': selected[b]['timing'][metric], 'candidate': selected[c]['timing'][metric],
                                           'percent': 100 * (selected[c]['timing'][metric] / selected[b]['timing'][metric] - 1)}
                                          for b, c in ((0, 1), (3, 2))]
        result['tasks'][case['name']] = entry
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    # Task failures are kept for review; never turn them into a quality pass.
    return 0 if all(x['all_tokens_identical'] and x['matched_prefix_reuse'] for x in result['tasks'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
