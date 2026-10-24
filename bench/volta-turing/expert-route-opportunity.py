#!/usr/bin/env python3
"""Count possible native expert calls from an actual serial-window trace.

This is a host dispatch inventory, not a GPU active-group or time profile.
The source's resident and PCIe calls are captured once per layer/verify group;
inactive groups may do no work, and graph replay does not log host calls.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def pack_pairs(path: Path) -> Counter:
    pairs = Counter()
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        pairs[(int(fields[1]), int(fields[2]))] += 1
    return pairs


def windows(path: Path) -> dict[int, Counter]:
    by_request = defaultdict(Counter)
    for line in path.read_text(errors="replace").splitlines():
        if "strata window trace:" not in line:
            continue
        fields = dict(re.findall(r"\b([A-Za-z_]+)=(\d+)", line))
        if fields.get("pipeline") != "0":
            continue
        by_request[int(fields["req"])][int(fields["T"])] += 1
    return by_request


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", type=Path)
    ap.add_argument("engine_log", type=Path)
    ap.add_argument("--requests", type=int, nargs="+", required=True)
    ap.add_argument("--spec-split", action="store_true", help="the trace's server used --spec-split")
    args = ap.parse_args()
    pairs = pack_pairs(args.pack)
    traces = windows(args.engine_log)
    if sum(pairs.values()) != 48:
        raise SystemExit(f"expected 48 native layers, got {sum(pairs.values())}")
    out = {"pack_format_pairs": {f"{gu}/{down}": count for (gu, down), count in sorted(pairs.items())},
           "requests": {}}
    for req in args.requests:
        widths = traces.get(req)
        if not widths:
            raise SystemExit(f"no serial windows for request {req}")
        nwin = sum(widths.values())
        group_widths = Counter()
        for t, count in widths.items():
            if args.spec_split and t >= 2:
                group_widths[(t + 1) // 2] += count
                group_widths[t // 2] += count
            else:
                group_widths[t] += count
        ngroups = sum(group_widths.values())
        resident_calls = ngroups * 48
        # verify.cpp: K=10; the resident call has grid_groups=0, the PCIe call
        # has kPcieGroupRows=4 (< cap_groups), so PCIe keeps mode 0 on CUDA.
        # iq_kernels.cu: sm75_down_em turns the resident down call to mode 0
        # only when cap_groups > 30, i.e. group width >= 4 at K=10.
        mode0_resident_down = 48 * sum(n for t, n in group_widths.items() if t > 3)
        out["requests"][str(req)] = {
            "serial_windows": nwin,
            "widths": {str(t): n for t, n in sorted(widths.items())},
            "spec_split": args.spec_split,
            "verify_groups": ngroups,
            "group_widths": {str(t): n for t, n in sorted(group_widths.items())},
            "resident_calls_per_projection": resident_calls,
            "pcie_calls_per_projection": resident_calls,
            "resident_gu_mode4_calls": resident_calls,
            "resident_down_mode4_calls": resident_calls - mode0_resident_down,
            "resident_down_mode0_calls": mode0_resident_down,
            "pcie_mode0_calls_per_projection": resident_calls,
            "format_pair_calls_per_projection": {f"{gu}/{down}": ngroups * count
                                                 for (gu, down), count in sorted(pairs.items())},
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
