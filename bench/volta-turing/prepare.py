#!/usr/bin/env python3
"""Prepare reproducible, isolated configs from the recorded v19 baseline."""
import argparse
import copy
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
workspace = repo.parent
p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
base = json.loads((workspace / "Strata-data/bench/2026-10-21-video/v19.json").read_text())
base["port"] = 18080
for name in ("base", "mode0", "mode4", "mode8", "v20-default", "wide1", "wide2"):
    cfg = copy.deepcopy(base)
    cfg["log"] = str((a.out / f"{name}-engine.log").resolve())
    if name.startswith("mode"):
        cfg["env"]["STRATA_EXP_MODE"] = name[-1]
    if name.startswith("wide") or name == "v20-default":
        # Keep /build/ in this path: the existing harness derives serve/server.py from it.
        cfg["exe"] = str(repo / "build/strata")
    if name.startswith("wide"):
        row = name[-1]
        cfg["env"]["STRATA_MMVQ_IL_ROWS"] = ";".join(
            f"{kind}:{n}:" + ",".join([row] * 5)
            for kind in (23, 12, 13, 14, 11) for n in range(5, 9)
        )
        cfg["env"]["STRATA_QIL_WIDE"] = "1"
    (a.out / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(a.out.resolve())
