#!/usr/bin/env python3
"""Move named tensors from one shard of a split GGUF to another, unchanged.

Size-capped splits can cut a layer's expert tensors across shards (orcarouter's Uncensored IQ2_M:
blk.14.ffn_down_exps in shard 1, its gate/up in shard 2), which tools/iq_pack.py rejects ("its gate/up/down
tensors are in different shards"). Moving the stray tensor next to its siblings fixes that. Tensor bytes,
types and shapes are copied as-is; every metadata key of both shards is kept. split.tensors.count is the
TOTAL over all shards, so moving a tensor leaves it (and the engine's shard-pairing check) valid.

  move_gguf_tensors.py --src SHARD_A --dst SHARD_B --out-src NEW_A --out-dst NEW_B --tensor NAME [--tensor ...]

The moved tensors are written into the destination before the first tensor of the same layer (or at the end).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

GGUF_PY = Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402


def copy_kv(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, has_arch: bool) -> None:
    if not has_arch:
        writer.kv_data[0].pop(gguf.Keys.General.ARCHITECTURE, None)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def write(path: Path, reader: gguf.GGUFReader, tensors: list) -> None:
    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field is not None else None
    writer = gguf.GGUFWriter(path, arch=arch or "", endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    copy_kv(reader, writer, arch is not None)
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    total = sum(t.data.nbytes for t in tensors)
    done = 0
    for t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
        done += t.data.nbytes
        print(f"\r  {path.name}: {100 * done // max(total, 1)}%", end="", flush=True)
    print()
    writer.close()


def layer_of(name: str) -> int | None:
    m = re.match(r"^blk\.(\d+)\.", name)
    return int(m[1]) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--out-src", type=Path, required=True)
    ap.add_argument("--out-dst", type=Path, required=True)
    ap.add_argument("--tensor", action="append", required=True)
    args = ap.parse_args()

    src = gguf.GGUFReader(args.src, "r")
    dst = gguf.GGUFReader(args.dst, "r")
    names = set(args.tensor)
    moved = [t for t in src.tensors if t.name in names]
    missing = names - {t.name for t in moved}
    if missing:
        print(f"not in {args.src.name}: {sorted(missing)}", file=sys.stderr)
        return 1
    if names & {t.name for t in dst.tensors}:
        print("a tensor to move already exists in the destination", file=sys.stderr)
        return 1

    src_out = [t for t in src.tensors if t.name not in names]
    dst_out = list(dst.tensors)
    for t in moved:
        layer = layer_of(t.name)
        at = next((i for i, d in enumerate(dst_out) if layer is not None and layer_of(d.name) == layer), len(dst_out))
        dst_out.insert(at, t)
    print(f"* moving {len(moved)} tensor(s): {args.src.name} ({len(src.tensors)} -> {len(src_out)}) -> "
          f"{args.dst.name} ({len(dst.tensors)} -> {len(dst_out)})")
    write(args.out_src, src, src_out)
    write(args.out_dst, dst, dst_out)
    print("* done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
