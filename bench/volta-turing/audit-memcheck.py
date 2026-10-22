#!/usr/bin/env python3
"""Reject incomplete Compute Sanitizer memcheck reports, even with zero errors.

This checks the report and process exit. The caller must separately establish
the intended request, kernel routes and output correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def audit(log: str, returncode: int) -> dict:
    counts = [int(n.replace(',', '')) for n in re.findall(
        r'^=+\s+ERROR SUMMARY:\s+([\d,]+) errors?\b', log, re.MULTILINE)]
    diagnostics = [line for line in log.splitlines()
                   if 'ERROR SUMMARY:' not in line and re.search(
                       r'error|warning|fatal|didn.t track|not track|undetected',
                       line, re.IGNORECASE)]
    reasons = []
    if returncode != 0:
        reasons.append('nonzero process exit')
    if not counts:
        reasons.append('missing final error summary')
    if any(count != 0 for count in counts):
        reasons.append('reported memory or API errors')
    if diagnostics:
        reasons.append('diagnostics or incomplete instrumentation')
    return {
        'qualified': not reasons,
        'returncode': returncode,
        'error_counts': counts,
        'diagnostic_lines': len(diagnostics),
        'diagnostic_examples': diagnostics[:8],
        'reasons': reasons,
        'scope': 'Report/exit check only; request and kernel coverage need separate evidence.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--returncode', type=int, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    result = audit(raw.decode(errors='replace'), args.returncode)
    result.update(log=str(args.log.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')
    return 0 if result['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
