#!/usr/bin/env python3
"""Audit opt-in STRATA_EXP_ROUTE_TRACE records against serial window traces."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def fields(line: str) -> dict[str, int]:
    return {key: int(value) for key, value in re.findall(r"\b([A-Za-z_]+)=(-?\d+)", line)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("engine_log", type=Path)
    ap.add_argument("pack", type=Path)
    ap.add_argument("--spec-split", action="store_true")
    args = ap.parse_args()
    formats = {}
    for line in args.pack.read_text().splitlines():
        if line and not line.startswith("#"):
            col = line.split()
            formats[int(col[0])] = (int(col[1]), int(col[2]))
    if len(formats) != 48:
        raise SystemExit(f"expected 48 layer formats; got {len(formats)}")

    pending = []
    windows = defaultdict(Counter)
    cells = defaultdict(lambda: defaultdict(lambda: Counter()))
    discarded_capture_rows = Counter()
    for line in args.engine_log.read_text(errors="replace").splitlines():
        if "strata expert route:" in line:
            pending.append(fields(line))
        elif "strata window trace:" in line:
            w = fields(line)
            if w.get("pipeline") != 0:
                raise SystemExit("pipeline windows need a separate audit")
            t = w["T"]
            split = args.spec_split and t >= 2
            expected_group_widths = sorted([(t + 1) // 2, t // 2] if split else [t])
            expected_rows = 48 * len(expected_group_widths)
            if len(pending) < expected_rows:
                raise SystemExit(f"req {w['req']} round {w['round']}: only {len(pending)} route rows, expected {expected_rows}")
            # The first use of a window width may capture graphs and run
            # separate planner calls. The last complete layer/group set before
            # the committed-window line is the execution being reported.
            discarded_capture_rows[str(w["req"])] += len(pending) - expected_rows
            pending = pending[-expected_rows:]
            got_group_widths = sorted({row["tokens"] for row in pending})
            if got_group_widths != sorted(set(expected_group_widths)):
                raise SystemExit(f"req {w['req']} round {w['round']}: group widths {got_group_widths} != {expected_group_widths}")
            if len(pending) != expected_rows:
                raise SystemExit(f"req {w['req']} round {w['round']}: {len(pending)} route rows, expected {expected_rows}")
            per_layer = Counter(row["layer"] for row in pending)
            if set(per_layer) != set(formats) or set(per_layer.values()) != {len(expected_group_widths)}:
                raise SystemExit(f"req {w['req']} round {w['round']}: layer coverage differs")
            req = str(w["req"])
            windows[req][t] += 1
            for row in pending:
                layer = row["layer"]
                if (row["gu"], row["down"]) != formats[layer] or row["k"] != 10:
                    raise SystemExit(f"req {req} round {w['round']}: format/K mismatch at layer {layer}")
                pair = f"{row['gu']}/{row['down']}"
                width = str(row["tokens"])
                cell = cells[req][f"{pair}@{width}"]
                cell["calls"] += 1
                for tier in ("resident", "pcie"):
                    groups = row[f"{tier}_groups"]
                    entries = row[f"{tier}_entries"]
                    if groups < 0 or entries < groups or entries > row["tokens"] * row["k"]:
                        raise SystemExit(f"req {req} round {w['round']}: invalid {tier} count")
                    cell[f"{tier}_groups"] += groups
                    cell[f"{tier}_entries"] += entries
                    cell[f"{tier}_zero_calls"] += groups == 0
            pending.clear()
    if pending:
        raise SystemExit(f"{len(pending)} route rows have no terminating window trace")
    print(json.dumps({"discarded_capture_or_prefill_rows": dict(discarded_capture_rows),
                      "windows": {req: {str(t): n for t, n in sorted(widths.items())}
                                   for req, widths in sorted(windows.items())},
                      "cells": {req: {key: dict(value) for key, value in sorted(tab.items())}
                                for req, tab in sorted(cells.items())}}, indent=2))


if __name__ == "__main__":
    main()
