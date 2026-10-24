#!/usr/bin/env python3
"""Score fixed continuations after real batched prefill, using existing engine diagnostics.

pin=P splits the raw prompt at P. --short-read scores only the following tail
through verify windows. Each arm uses the same corpus tokens and split. Scores
are full-vocabulary target log probabilities and argmax, not full-distribution
KL. The 64-token lists are retained without imputing missing probabilities.
No timing from this instrumented runner is a performance result.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def set_arg(args, name, value):
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args.extend([name, str(value)])


def parse_rows(path, first, count, ids):
    lines = path.read_text().splitlines()[first:]
    assert len(lines) == count, f'expected {count} new scores, got {len(lines)}'
    rows = []
    for line in lines:
        fields = line.split('\t')
        assert len(fields) == 72, 'expected eight fields and 64 top probabilities'
        pos, target, top = int(fields[0]), int(fields[1]), int(fields[3])
        assert 0 <= pos < len(ids) - 1 and target == ids[pos + 1]
        lp = float(fields[2])
        assert math.isfinite(lp) and lp <= 0
        rows.append({'pos': pos, 'target': target, 'argmax': top, 'logp': lp})
    return rows


def compare(left, right):
    assert [(r['pos'], r['target']) for r in left] == [(r['pos'], r['target']) for r in right]
    n = len(left)
    a = -sum(r['logp'] for r in left) / n
    b = -sum(r['logp'] for r in right) / n
    return {'positions': n, 'top1_agreement': sum(x['argmax'] == y['argmax'] for x,y in zip(left,right)) / n,
            'left_nll': a, 'right_nll': b, 'delta_nll': b-a,
            'perplexity_ratio': math.exp(b-a),
            'max_target_logp_difference': max(abs(x['logp']-y['logp']) for x,y in zip(left,right))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--targets', default='1024,4096,16384,32768')
    parser.add_argument('--segments', default='code,docs,english,multilingual,agent')
    parser.add_argument('--tail', type=int, default=128)
    parser.add_argument('--arms', default='base,candidate,candidate,base')
    parser.add_argument('--switch-env', default='STRATA_PF_FUSED',
                        help='single 0/1 environment switch that distinguishes the arms')
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    assert 1 <= a.tail <= 512
    targets = [int(n) for n in a.targets.split(',')]
    assert all(n > a.tail for n in targets)
    arms = a.arms.split(',')
    assert arms and all(arm in ('base', 'candidate') for arm in arms)
    repo = Path(__file__).resolve().parents[2]
    runner = module('decode_runner', Path(__file__).with_name('decode-compare.py'))
    configs = {arm: json.loads(path.read_text()) for arm,path in [('base',a.base),('candidate',a.candidate)]}
    normalized = []
    for arm,value in [('base','0'),('candidate','1')]:
        cfg = json.loads(json.dumps(configs[arm]))
        assert cfg['env'].pop(a.switch_env) == value
        cfg.pop('log',None)
        normalized.append(cfg)
    assert normalized[0] == normalized[1], 'only fused switch may differ'
    binary_hash = digest(configs['base']['exe'])
    sys.path.insert(0,str(repo / 'tools'))
    from strata_tokenizer import Tokenizer
    td = Path(configs['base']['tokenizer'])
    vocab = json.loads((td/'vocab.json').read_text())
    tokens = [None]*len(vocab)
    for token,i in vocab.items(): tokens[i]=token
    tok = Tokenizer(tokens,(td/'merges.txt').read_text().splitlines(),json.loads((td/'token_type.json').read_text()))
    cases = []
    sources = {}
    for segment in a.segments.split(','):
        assert segment in ('code','docs','english','multilingual','agent')
        source = repo/'data/quant_quality/v1'/f'{segment}.txt'
        sources[str(source)] = digest(source)
        text = source.read_text()
        for target in targets:
            # A different initial label prevents cross-case prefix reuse. Repeated
            # corpus is explicit; these are numerical probes, not task quality.
            label = f'Teacher-forced numerical case {segment}, prefix {target}.\n'
            repeats = 1
            while True:
                ids = tok.encode(label + (text+'\n')*repeats, parse_special=False)
                if len(ids) >= target+a.tail+1: break
                repeats *= 2
            cases.append({'name':f'{segment}-{target}','prefix':target,'tail':a.tail,
                          'corpus_repetitions':repeats,'ids':ids[:target+a.tail+1]})
    max_context = None
    for i, arg in enumerate(configs['base']['args'][:-1]):
        if arg == '--max-context': max_context = int(configs['base']['args'][i + 1])
    if max_context is not None:
        for case in cases:
            if len(case['ids']) + 1 > max_context:
                raise ValueError(f"{case['name']}: {len(case['ids'])} prompt tokens plus one generated token "
                                 f"exceed --max-context {max_context}")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=False)
    manifest = {'binary_sha256':binary_hash,'harness_sha256':digest(__file__),
                'engine_runner_sha256':digest(Path(__file__).with_name('decode-compare.py')),
                'sources':sources,'arms':arms,'switch_env':a.switch_env,'scope':__doc__,
                'controls':'No adaptation; prefix snapshots/turn/root cuts disabled; pin creates the explicit prefix cut. Original auto prefill and cache budgets retained.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    if a.prepare_only: return 0
    all_rows = []
    for leg,arm in enumerate(arms,1):
        assert digest(configs[arm]['exe']) == binary_hash
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU busy'
        directory = a.out/f'{leg}-{arm}'
        directory.mkdir()
        cfg = json.loads(json.dumps(configs[arm]))
        for key,value in [('--adapt-swaps',0),('--adapt-async',0),('--prompt-cache',3),
                          ('--prompt-cache-root',0),('--turn-token',-1),('--short-read',a.tail)]:
            set_arg(cfg['args'],key,value)
        cfg['env'].update(STRATA_PREFIX_SNAPSHOTS='0',STRATA_CACHE_MESSAGE_BOUNDARY='0',
                          STRATA_TRACE='1',STRATA_LOGPOS=str(directory/'positions.tsv'),STRATA_LOGPOS_TOPK='64')
        for key in ('STRATA_DUMP_FIRST_LOGITS','STRATA_CKPT_REREAD','STRATA_PIPELINE_SWITCH'):
            cfg['env'].pop(key,None)
        (directory/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        engine = runner.Engine(cfg,directory)
        scored = 0
        try:
            engine.ready()
            for case in cases:
                engine.process.stdin.write((f"GEN 1 pin={case['prefix']} " + ','.join(map(str,case['ids']))+'\n').encode())
                engine.process.stdin.flush()
                deadline = time.monotonic()+1200
                while True:
                    line = engine.line(deadline)
                    if line.startswith(b'DONE '):
                        done = line.decode().split()
                        break
                assert int(done[2]) == len(case['ids']) and done[5] in ('stop','length'),done
                assert int(done[8]) == 0, f'unexpected prefix reuse: {done}'
                rows = parse_rows(directory/'positions.tsv',scored,case['tail'],case['ids'])
                assert [r['pos'] for r in rows] == list(range(case['prefix'],case['prefix']+case['tail']))
                scored += len(rows)
                log = (directory/'engine.log').read_text()
                assert 'pin=' not in '\n'.join(l for l in log.splitlines() if 'ignored:' in l)
                entry = {'leg':leg,'arm':arm,'case':case['name'],'done':done,'rows':rows}
                all_rows.append(entry)
                (directory/(case['name']+'.json')).write_text(json.dumps(entry,indent=2)+'\n')
                print(json.dumps({'leg':leg,'arm':arm,'case':case['name'],'positions':len(rows),'nll':-sum(r['logp'] for r in rows)/len(rows)}),flush=True)
        finally:
            engine.close()
        assert engine.process.returncode == 0
        log = (directory/'engine.log').read_text()
        batched = [int(n) for n in re.findall(r'strata trace: read (\d+) tokens \(batched\)',log)]
        windows = [int(n) for n in re.findall(r'strata trace: read (\d+) tokens \(windows\)',log)]
        assert batched == [c['prefix'] for c in cases],batched
        assert windows == [c['tail'] for c in cases],windows
        if a.switch_env == 'STRATA_PF_FUSED' and arm == 'candidate' and any(n >= 4096 for n in targets):
            assert 'prompt experts on the fused int8 kernels' in log, 'no fused route evidence'
        (directory/'coverage.json').write_text(json.dumps({'batched':batched,'windows':windows,'scored_positions':scored},indent=2)+'\n')
        time.sleep(3)
    comparisons = {}
    if arms == ['base','candidate','candidate','base']:
        for left,right in ((1,2),(4,3),(1,4),(2,3)):
            comparisons[f'{left}-vs-{right}'] = {}
            for case in cases:
                l = next(x['rows'] for x in all_rows if x['leg']==left and x['case']==case['name'])
                r = next(x['rows'] for x in all_rows if x['leg']==right and x['case']==case['name'])
                comparisons[f'{left}-vs-{right}'][case['name']] = compare(l,r)
    (a.out/'summary.json').write_text(json.dumps({'comparisons':comparisons,'requests':len(all_rows),
         'positions':sum(len(r['rows']) for r in all_rows),'limits':__doc__},indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
