#!/usr/bin/env python3
"""Copy a GGUF shard with named tensors taken from another GGUF of the same model.

The engine reads a native pack's token embedding straight from the GGUF and has GPU dequantizers only for the
i-quants, Q3_K and Q2_0. orcarouter's Uncensored IQ2_XXS stores token_embd as Q2_K, so it cannot start; its
IQ2_M sibling (same weights, other quantization) stores the same tensor as IQ3_S. This writes a copy of the
IQ2_XXS shard with that tensor swapped in (same name and shape; type and bytes from the donor). Everything else,
including every metadata key, is copied unchanged.

  replace_gguf_tensor.py --input SHARD --donor OTHER_GGUF [--donor ...] --output NEW_SHARD --tensor token_embd.weight

--donor may be given once per shard of a split donor; each tensor is taken from whichever donor holds it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

GGUF_PY = Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from move_gguf_tensors import write  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--donor", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tensor", action="append", required=True)
    args = ap.parse_args()
    src = gguf.GGUFReader(args.input, "r")
    donor = {t.name: (t, path) for path in args.donor for t in gguf.GGUFReader(path, "r").tensors}
    out = []
    for t in src.tensors:
        if t.name in args.tensor:
            if t.name not in donor:
                print(f"{t.name}: not in any donor", file=sys.stderr)
                return 1
            d, path = donor[t.name]
            if list(d.shape) != list(t.shape):
                print(f"{t.name}: shape {list(t.shape)} vs donor {list(d.shape)}", file=sys.stderr)
                return 1
            print(f"* {t.name}: {t.tensor_type.name} -> {d.tensor_type.name} (from {path.name})")
            out.append(d)
        else:
            out.append(t)
    write(args.output, src, out)
    print("* done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
