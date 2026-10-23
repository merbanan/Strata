#!/usr/bin/env python3
"""Audit saved task records and traces without executing generated code.

Partial reports describe completed requests only. Manual fresh-code/prose
reviews apply only to the identical output bytes and task type. Supports the
fused-prefill diagnostic, IQ2_XS experiment and QSA FP16 comparisons.
Task qualification and performance qualification remain separate.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def edit_check(kind, text, source):
    blocks = list(re.finditer(r'```(?:python|py)?\s*\n(.*?)```', text, re.S))
    if len(blocks) != 1 or len(re.findall(r'^\s*```', text, re.M)) != 2:
        return {'qualified': False, 'reason': 'requires one complete code block'}
    block = blocks[0]
    if text[:block.start()].strip() or text[block.end():].strip():
        return {'qualified': False, 'reason': 'extra text outside the requested code block'}
    code = block[1]
    try:
        compile(code, '<saved edit; never executed>', 'exec')
        tree = ast.parse(code)
    except SyntaxError as error:
        return {'qualified': False, 'reason': 'invalid Python module', 'error': str(error)}
    if kind == 'edit-rename':
        expected = ast.parse(source.replace('GGUFFile', 'GgufReader'))
        equal = ast.dump(tree, include_attributes=False) == ast.dump(expected, include_attributes=False)
        return {'qualified': equal, 'reason': 'exact requested AST' if equal else 'missing rename or unrelated AST changes'}
    wanted = ast.dump(ast.parse('log = logging.getLogger(__name__)').body[0], include_attributes=False)
    imports = [i for i, node in enumerate(tree.body) if isinstance(node, ast.Import) and
               len(node.names) == 1 and node.names[0].name == 'logging' and node.names[0].asname is None]
    assignments = [i for i, node in enumerate(tree.body) if ast.dump(node, include_attributes=False) == wanted]
    if len(imports) != 1 or len(assignments) != 1:
        return {'qualified': False, 'reason': 'missing or duplicate logging import/logger assignment'}
    last_import = max(i for i, node in enumerate(tree.body) if isinstance(node, (ast.Import, ast.ImportFrom)))
    if assignments[0] != last_import + 1:
        return {'qualified': False, 'reason': 'logger is not immediately after imports'}
    tree.body = [node for i, node in enumerate(tree.body) if i not in (*imports, *assignments)]
    equal = ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
    return {'qualified': equal, 'reason': 'exact requested AST' if equal else 'unrelated AST changes'}


def completed_traces(raw):
    requests = []
    active = None
    for line in raw.decode().splitlines():
        match = re.fullmatch(r'strata trace: request (\d+) (\d+)', line)
        if match:
            assert active is None, 'request without prior DONE'
            active = {'prompt_tokens': int(match[1]), 'mode': int(match[2]), 'chunks': [], 'caps': [], 'tokens': []}
        elif active is not None:
            if line.startswith('T '):
                active['tokens'].append(int(line.split()[1]))
            elif line.startswith('DONE '):
                active['done'] = line.split()
                requests.append(active); active = None
            else:
                match = re.fullmatch(r'strata trace: prompt chunk (\d+) of (\d+)', line)
                if match: active['chunks'].append([int(match[1]), int(match[2])])
                match = re.fullmatch(r'strata trace: fused group tail T (\d+) cap (\d+)', line)
                if match: active['caps'].append([int(match[1]), int(match[2])])
    return requests


def validate_qsa_control(cfg, raw_log, policy):
    assert cfg['env']['STRATA_IDX_F16'] == '1', 'QSA comparison requires FP16 keys'
    if policy == 'fixed':
        assert cfg['env']['STRATA_VERIFY_DEFER_CPU'] == '0', 'deferred CPU must be disabled'
        options = (('--adapt-async', '0'), ('--adapt-swaps', '0'), ('--pcie-frac', '0.11'))
        assert b'strata diagnostic: fixed draft-policy costs; correctness control only' in raw_log
        assert b'adaptive tier asynchronous' not in raw_log
    else:
        assert policy == 'production'
        assert cfg['env']['STRATA_VERIFY_DEFER_CPU'] == '1', 'production deferred CPU must be enabled'
        assert cfg['env']['STRATA_PREFILL_CPU_SHARE'] == 'auto'
        assert cfg['env']['STRATA_PCIE_FRAC_DEFAULT'] == 'old'
        assert '--pcie-frac' not in cfg['args'], 'production PCIe probe must remain active'
        options = (('--adapt-async', '1'), ('--adapt-swaps', '192'))
        assert b'adaptive tier asynchronous' in raw_log
        assert b'PCIe probe:' in raw_log
        assert b'fixed draft-policy costs' not in raw_log
    for option, value in options:
        assert cfg['args'].count(option) == 1
        assert cfg['args'][cfg['args'].index(option) + 1] == value, option


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workload', type=Path)
    parser.add_argument('--prepared-cases', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--qsa-policy', choices=('fixed', 'production'), default='fixed')
    args = parser.parse_args()
    inputs = {}
    def read(path):
        raw = path.read_bytes()
        inputs[str(path.resolve())] = {
            'sha256': digest(raw), 'bytes': len(raw),
            'prefix_snapshot': args.partial and path.name in ('engine.log', 'runs.jsonl'),
        }
        return raw
    assert not args.output.exists(), 'preserve earlier review reports'
    raw_cases = read(args.workload/'cases.json')
    assert raw_cases == read(args.prepared_cases)
    manifest = json.loads(read(args.workload/'manifest.json'))
    feature = manifest['feature_env']
    assert feature in ('STRATA_PF_FUSED', 'STRATA_IQ2XS_GU_FUSED', 'STRATA_SCORES_HALVE'), 'unsupported experiment'
    assert args.qsa_policy == 'fixed' or feature == 'STRATA_SCORES_HALVE'
    assert manifest['base_value'] == '0' and manifest['candidate_value'] == '1'
    diagnostic = manifest['diagnostic_budgets']
    cases = json.loads(raw_cases)
    assert len(cases) == 19 and len({c['name'] for c in cases}) == 19
    arms = ('base', 'candidate', 'candidate', 'base')
    expected = [(leg, arm, case) for leg, arm in enumerate(arms, 1) for case in cases]
    raw_rows = read(args.workload/'runs.jsonl')
    lines = raw_rows.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        assert args.partial, 'incomplete final record'
        lines.pop()
    rows = [json.loads(line) for line in lines]
    assert 0 < len(rows) <= len(expected)
    if not args.partial:
        assert len(rows) == len(expected)
        summary = json.loads(read(args.workload/'summary.json'))
        assert summary['diagnostic_budgets'] == diagnostic
        assert summary['timing_comparisons_enabled'] == (not diagnostic)
        if not diagnostic:
            assert summary['budgets_equal'], 'strict run has unequal memory budgets'
    source = read(args.workload/'src_edit.py').decode()
    tokenizer_path = Path(__file__).resolve().parents[2]/'tools/strata_tokenizer.py'
    read(tokenizer_path)
    spec = importlib.util.spec_from_file_location('workload_tokenizer', tokenizer_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    first_config = json.loads(read(args.workload/'1-base/config.json'))
    tokenizer_dir = Path(first_config['tokenizer'])
    vocab = json.loads(read(tokenizer_dir/'vocab.json'))
    tokens = [None] * len(vocab)
    for token, index in vocab.items(): tokens[index] = token
    tokenizer = module.Tokenizer(tokens, read(tokenizer_dir/'merges.txt').decode().splitlines(),
                                 json.loads(read(tokenizer_dir/'token_type.json')))
    eos = {vocab['<|endoftext|>'], vocab['<|im_end|>']}
    reviews = {}
    for kind in ('fresh-code', 'prose'):
        for path in sorted(args.reviews.rglob(kind+'-independent-review.json')):
            review = json.loads(read(path))
            sha = review['source_sha256']
            assert digest(read(Path(review['source']))) == sha, path
            key = kind, sha
            if key in reviews: assert reviews[key]['qualified'] == review['qualified'], 'conflicting reviews'
            reviews[key] = {'qualified': review['qualified'], 'review': str(path.resolve()), 'review_sha256': digest(path.read_bytes())}
    traces = {}
    normalized_configs = {}
    results = []
    for row, (leg, arm, case) in zip(rows, expected):
        assert (row['leg'], row['arm'], row['name']) == (leg, arm, case['name'])
        assert row['request_sha256'] == case['request_sha256'] == digest(json.dumps(case['body'], sort_keys=True).encode())
        directory = args.workload/f'{leg}-{arm}'
        assert json.loads(read(directory/(case['name']+'.json'))) == row
        text_bytes = read(directory/(case['name']+'.txt'))
        text = text_bytes.decode()
        ids = row['tokens']
        assert tokenizer.decode(ids[:-1] if ids and ids[-1] in eos else ids) == text, 'saved text differs from emitted tokens'
        if leg not in traces:
            cfg = json.loads(read(directory/'config.json'))
            assert cfg['tokenizer'] == str(tokenizer_dir)
            assert cfg['env'].pop(feature) == ('0' if arm == 'base' else '1')
            cfg.pop('log', None)
            normalized_configs[leg] = cfg
            assert all(c == normalized_configs[leg] for c in normalized_configs.values()), 'configuration drift beyond feature switch'
            raw_log = read(directory/'engine.log')
            if feature == 'STRATA_SCORES_HALVE':
                validate_qsa_control(cfg, raw_log, args.qsa_policy)
            traces[leg] = completed_traces(raw_log)
        index = cases.index(case)
        trace = traces[leg][index]
        assert trace['done'] == row['done'] and trace['tokens'] == row['tokens']
        assert trace['mode'] == 0 and trace['prompt_tokens'] == len(case['ids']) == int(row['done'][2])
        assert len(row['tokens']) == int(row['done'][1])
        reused, read_tokens = int(row['done'][8]), int(row['done'][14])
        assert 0 <= reused <= len(case['ids']) and reused + read_tokens == len(case['ids'])
        finished = row['done'][5] == 'stop'
        kind = case['checker']
        if kind == 'marker':
            review = {'qualified': text.strip() == case['expected'], 'reason': 'exact retrieval marker'}
        elif kind.startswith('edit-'):
            review = edit_check(kind, text, source)
        else:
            review = reviews.get((kind, digest(text_bytes)), {'qualified': False, 'reason': 'independent review pending'})
        # Kernel coverage is separate from task correctness. Original short
        # cases cannot qualify a fused-prefill numerical change.
        exercised_fused = any(t >= 3072 and cap == 0 for t, cap in trace['caps'])
        if feature == 'STRATA_PF_FUSED' and 'context_target' in case and arm == 'candidate':
            assert exercised_fused, 'long-context candidate did not exercise fused prefill'
        results.append({'leg': leg, 'arm': arm, 'name': case['name'], 'output_sha256': digest(text_bytes),
                        'finish': row['done'][5], 'qualified': finished and review['qualified'],
                        'review': review, 'prompt_tokens': len(case['ids']),
                        'prompt_tokens_reused': reused, 'prompt_tokens_read': read_tokens,
                        'chunks': trace['chunks'], 'group_caps': trace['caps'],
                        'fused_prefill_exercised': exercised_fused})
    pending = sum(r['review'].get('reason') == 'independent review pending' for r in results)
    records_complete = len(results) == len(expected)
    read(Path(__file__))
    report = {'status': 'partial_review' if args.partial else
              ('independent_reviews_pending' if pending else 'completed_workload_review'),
              'requests_reviewed': len(results), 'planned_requests': len(expected),
              'feature_env': feature, 'qsa_policy': args.qsa_policy if feature == 'STRATA_SCORES_HALVE' else None,
              'diagnostic_budgets': diagnostic,
              'records_complete': records_complete, 'pending_independent_reviews': pending,
              'saved_text_matches_emitted_tokens': True,
              'complete': records_complete and not pending, 'adopted': False,
              'counts': dict(Counter(f"{r['arm']}:{'qualified' if r['qualified'] else 'unqualified'}" for r in results)),
              'results': results, 'inputs': inputs,
              'limits': 'Incomplete or unreviewed tasks remain unqualified. Raw response/protocol and applicable fused-prefill coverage are checked. Diagnostic budgets forbid timing comparisons; equal budgets alone do not qualify performance. Kernel launch coverage requires separate evidence. No model-wide quality or adoption claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','inputs')}, indent=2))


if __name__ == '__main__': main()
