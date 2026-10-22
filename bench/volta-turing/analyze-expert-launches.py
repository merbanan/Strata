#!/usr/bin/env python3
"""Count direct expert launches and replayed nodes separately from capture.

These are submitted kernel counts and launch capacities, not active routed
groups, transferred bytes, GPU time or completed device work. Require a clean
model/test process and matching output separately. Graph updates, disabled
nodes and conditional graphs are outside the current source-audited scope.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def shape(fields):
    assert len(fields) == 8 and fields[0].startswith('_Z'), 'invalid kernel shape'
    dimensions = tuple(map(int, fields[1:]))
    assert all(n > 0 for n in dimensions[:6]) and dimensions[6] >= 0
    return (fields[0], *dimensions)


def analyze(text):
    graphs, pending = {}, None
    direct, captured, replayed, events = Counter(), Counter(), Counter(), Counter()
    for line in text.splitlines():
        f = line.split('\t')
        event = f[0]
        events[event] += 1
        if pending is not None:
            assert event == 'NODE' and f[1] == pending, 'incomplete graph metadata'
        if event == 'LAUNCH':
            assert len(f) == 12 and f[1:3] == ['0','0'], 'failed launch or capture query'
            assert f[3] in ('0','1'), 'invalidated capture'
            (direct if f[3] == '0' else captured)[shape(f[4:])] += 1
        elif event == 'GRAPH':
            assert len(f) == 5 and f[2:4] == ['0','1'], 'uninspected or failed graph'
            count = int(f[4])
            assert count >= 0 and f[1] not in graphs, 'duplicate live graph handle'
            graphs[f[1]] = {'expected':count, 'shapes':Counter()}
            if count: pending = f[1]
        elif event == 'NODE':
            assert len(f) == 11 and pending == f[1], 'unexpected graph node'
            item, count = shape(f[3:]), int(f[2])
            record = graphs[f[1]]
            assert count > 0 and item not in record['shapes']
            record['shapes'][item] = count
            assert len(record['shapes']) <= record['expected']
            if len(record['shapes']) == record['expected']: pending = None
        elif event == 'REPLAY':
            assert len(f) == 4 and f[2:] == ['0','1'], 'unknown or failed graph replay'
            assert f[1] in graphs, 'replay before graph registration or after destruction'
            replayed.update(graphs[f[1]]['shapes'])
        elif event == 'DESTROY':
            assert len(f) == 3 and f[2] == '0' and f[1] in graphs, 'invalid destruction'
            del graphs[f[1]]
        else:
            raise AssertionError(f'unknown trace record: {event}')
    assert pending is None, 'truncated graph metadata'
    assert events and direct + replayed, 'no executed expert kernels'
    counts = []
    for item in sorted(set(direct) | set(captured) | set(replayed)):
        counts.append(dict(name=item[0], grid=list(item[1:4]), block=list(item[4:7]), shared_bytes=item[7],
                           direct=direct[item], captured=captured[item], replayed=replayed[item],
                           submitted=direct[item]+replayed[item]))
    return dict(events=dict(events), live_graph_handles_at_end=len(graphs), counts=counts,
                direct_kernels=sum(direct.values()), capture_records=sum(captured.values()),
                replayed_kernels=sum(replayed.values()),
                submitted_kernels=sum(direct.values())+sum(replayed.values()), limits=__doc__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve earlier analysis'
    raw = args.trace.read_bytes()
    result = analyze(raw.decode())
    result.update(trace=str(args.trace.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'counts'},indent=2))


if __name__ == '__main__': main()
