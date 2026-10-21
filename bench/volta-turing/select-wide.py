#!/usr/bin/env python3
"""Select opt-in wide rows only when every measured shape/run beats group16 by 3%."""
import argparse
import collections
import json
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base", required=True, type=Path)
p.add_argument("--out", required=True, type=Path)
p.add_argument("logs", nargs="+", type=Path)
a = p.parse_args()
if len(a.logs) < 2:
    p.error("at least two independent timing logs are required")
types = {"IQ4_XS": 23, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "Q3_K": 11}
pattern = re.compile(r"^BENCH (.+?)\s+T=(\d+)\s+multi ([\d.]+) us \| il rows1 ([\d.]+) rows2 ([\d.]+) rows4 ([\d.]+) us$", re.M)
cells = collections.defaultdict(list)
keys = None
for path in a.logs:
    text = path.read_text()
    if "mmvq_il_parity: OK (33 cases, T 1-8" not in text or "Timing reference: group=16" not in text or "FAIL" in text:
        p.error(f"incomplete, failed, or wrong-reference log: {path}")
    found = set()
    for m in pattern.finditer(text):
        name, n = m[1].strip(), int(m[2])
        if n < 5:
            continue
        n_out = int(name.rsplit("x", 1)[1])
        cls = sum(n_out >= limit for limit in (2048, 4096, 8192, 12288))
        key = types[name.split()[0]], n, cls
        ref, r1, r2 = map(float, (m[3], m[4], m[5]))
        if min(ref, r1, r2) <= 0:
            p.error(f"nonpositive timing in {path}: {name}")
        cells[key].append({"shape": name, "log": str(path), "reference_us": ref,
                           "rows1_us": r1, "rows2_us": r2})
        found.add((name, n))
    if keys is not None and keys != found:
        p.error("timing logs must cover identical shapes")
    if len(found) != 120:  # 30 non-odd-row shapes, four wide column counts
        p.error(f"expected 120 wide timing cases in {path}, found {len(found)}")
    keys = found

entries, choices = [], []
for kind in types.values():
    for n in range(5, 9):
        rows = []
        for cls in range(5):
            samples = cells[kind, n, cls]
            chosen, worst = 0, None
            if samples:
                ratios = {r: max(s[f"rows{r}_us"] / s["reference_us"] for s in samples) for r in (1, 2)}
                best = min(ratios, key=ratios.get)
                worst = ratios[best]
                if worst < 0.97:
                    chosen = best
            rows.append(chosen)
            choices.append({"type": kind, "columns": n, "class": cls, "rows": chosen,
                            "worst_ratio": worst, "samples": samples})
        entries.append(f"{kind}:{n}:" + ",".join(map(str, rows)))
cfg = json.loads(a.base.read_text())
cfg.setdefault("env", {})["STRATA_MMVQ_IL_ROWS"] = ";".join(entries)
cfg["env"]["STRATA_QIL_WIDE"] = "1"
cfg["env"]["STRATA_MMVQ_IL_WIDE"] = "1"
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(cfg, indent=2) + "\n")
a.out.with_suffix(".selection.json").write_text(json.dumps(choices, indent=2) + "\n")
print(f"Enabled {sum(c['rows'] > 0 for c in choices)}/100 cells; unmeasured cells stay off")
print(cfg["env"]["STRATA_MMVQ_IL_ROWS"])
