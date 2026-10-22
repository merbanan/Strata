#!/usr/bin/env python3
"""Summarize trace-graph-memory.cpp records, retaining measurement limitations."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    header = None
    pending_window = None
    records = []
    for line in raw.decode(errors='replace').splitlines():
        label = re.search(r'strata verify: the (\d+)-token window graph has (\d+) nodes:', line)
        if label:
            pending_window = tuple(map(int, label.groups()))
        if 'GRAPH_MEM_HEADER\t' in line:
            header = line[line.index('GRAPH_MEM_HEADER\t'):].split('\t')[1:]
        elif 'GRAPH_MEM\t' in line:
            assert header, 'record before header'
            values = line[line.index('GRAPH_MEM\t'):].split('\t')[1:]
            assert len(values) == len(header), 'truncated/interleaved memory record'
            row = {key: value if key in ('event', 'exec') else int(value) for key, value in zip(header, values)}
            row['window_tokens'] = None
            if row['event'] == 'instantiate' and row['status'] == 0 and pending_window:
                assert row['top_nodes'] == pending_window[1], 'window label/node count mismatch'
                row['window_tokens'] = pending_window[0]
                pending_window = None
            records.append(row)
    assert records, 'no graph memory records'
    graphs = {}
    errors = []
    for row in records:
        if row['status'] != 0 and row['event'] != 'exit_snapshot':
            errors.append(row)
        if row['id'] == 0:
            errors.append({'reason': 'untracked graph or failed creation', 'record': row})
            continue
        if row['event'] == 'instantiate':
            assert row['id'] not in graphs, 'duplicate graph lifetime id'
            graphs[row['id']] = {'id': row['id'], 'exec': row['exec'], 'window_tokens': row['window_tokens'],
                                 'top_nodes': row['top_nodes'], 'top_kernel_nodes': row['top_kernel_nodes'],
                                 'inspected': row['inspected'], 'events': [], 'destroyed': False}
        assert row['id'] in graphs, 'graph event without creation record'
        graph = graphs[row['id']]
        assert graph['exec'] == row['exec']
        assert row['launch_attempts'] >= row['launches_submitted']
        if graph['events']:
            assert row['launch_attempts'] >= graph['events'][-1]['launch_attempts']
            assert row['launches_submitted'] >= graph['events'][-1]['launches_submitted']
        graph['events'].append(row)
        if row['event'] == 'destroy' and row['status'] == 0:
            graph['destroyed'] = True
        graph['launch_attempts'] = row['launch_attempts']
        graph['launches_submitted'] = row['launches_submitted']
        if (row['before_status'] == row['after_status'] == 0
                and row['before_device'] == row['after_device']):
            graph[row['event'] + '_observed_free_change_bytes'] = row['after_free'] - row['before_free']
    result = {
        'log': str(args.log), 'log_sha256': hashlib.sha256(raw).hexdigest(),
        'graph_lifetimes': len(graphs), 'peak_live_executables': max(row['live_execs'] for row in records),
        'launches_submitted': sum(graph['launches_submitted'] for graph in graphs.values()),
        'all_destroyed': all(graph['destroyed'] for graph in graphs.values()), 'errors': errors,
        'graphs': list(graphs.values()),
        'limits': ['Device free-memory changes are observations, not exact bytes owned by each graph.',
                   'No added GPU synchronization; upload/first-launch observations can precede device completion.',
                   'Counters count successful host submissions, not independent proof of GPU completion.',
                   'Node counts are top-level only; child graph nodes are not flattened.',
                   'Exit snapshots can precede global destructors and do not establish leaks.',
                   'Arena allocations and batch policies require separate coverage.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('graphs', 'limits')}, indent=2))


if __name__ == '__main__':
    main()
