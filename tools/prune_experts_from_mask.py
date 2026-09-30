#!/usr/bin/env python3
"""Prune a GGUF's MoE experts down to a fixed retained-index set per layer, taken from an
ISTA-DASLab `*.rco-allocation.txt` file (Section 2: `<tensor>[<original index>]: <type or empty>`,
empty = pruned). Unlike the published Coder release, this keeps the SOURCE quantization (e.g. Q2_0)
for every retained expert - it only removes experts, it does not requantize anything.

Only four tensor kinds per layer carry the expert axis, and in this GGUF it is axis 0 of the raw
byte array (contiguous per-expert blocks), confirmed against blk.0 of the Q2_0 shard 1:
  blk.<L>.ffn_down_exps.weight, blk.<L>.ffn_gate_exps.weight, blk.<L>.ffn_up_exps.weight  (quantized)
  blk.<L>.ffn_gate_inp.weight                                                             (router, BF16)
Everything else (attention, hyper-connection mixer, SSM/GDN state, shared-expert path, output head,
token embedding) is copied through byte-for-byte, unchanged.

The second shard (the PLE n-gram table) is untouched by expert pruning and is reused as-is.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

GGUF_PY = Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))
import gguf  # noqa: E402

EXPERT_TENSOR_RE = re.compile(r'^blk\.(\d+)\.(ffn_down_exps|ffn_gate_exps|ffn_up_exps|ffn_gate_inp)\.weight$')
MASK_LINE_RE = re.compile(r'^blk\.(\d+)\.(ffn_down_exps|ffn_gate_exps|ffn_up_exps)\.weight\[(\d+)\]: (.*)$')


def parse_kept_experts(mask_path: Path, expert_count: int, kept_count: int) -> dict[int, list[int]]:
    """Read an rco-allocation.txt and return {layer: [kept original indices, ascending]}."""
    per_layer: dict[int, dict[int, dict[int, str]]] = {}
    with open(mask_path) as f:
        for line in f:
            m = MASK_LINE_RE.match(line.rstrip('\n'))
            if not m:
                continue
            layer, tname, idx, typ = int(m[1]), m[2], int(m[3]), m[4]
            per_layer.setdefault(layer, {}).setdefault(tname, {})[idx] = typ

    kept: dict[int, list[int]] = {}
    for layer, tensors in per_layer.items():
        sets = {}
        for tname in ('ffn_down_exps', 'ffn_gate_exps', 'ffn_up_exps'):
            d = tensors.get(tname)
            if d is None or len(d) != expert_count:
                raise ValueError(f'layer {layer} {tname}: expected {expert_count} indexed entries, '
                                  f'got {len(d) if d else 0}')
            sets[tname] = sorted(i for i, t in d.items() if t != '')
        if not (sets['ffn_down_exps'] == sets['ffn_gate_exps'] == sets['ffn_up_exps']):
            raise ValueError(f'layer {layer}: down/gate/up kept-expert sets disagree')
        k = sets['ffn_down_exps']
        if len(k) != kept_count:
            raise ValueError(f'layer {layer}: {len(k)} kept, expected {kept_count}')
        kept[layer] = k
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, required=True, help='source GGUF (shard 1)')
    ap.add_argument('--output', type=Path, required=True, help='pruned GGUF to write')
    ap.add_argument('--mask', type=Path, required=True, help='rco-allocation.txt with the retained-index mask')
    ap.add_argument('--expert-count-key', default='qwen4exp.expert_count')
    ap.add_argument('--expert-count', type=int, default=512)
    ap.add_argument('--kept-count', type=int, default=256)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if args.output.exists() and not args.force:
        print(f'{args.output} already exists; pass --force to overwrite', file=sys.stderr)
        return 1

    print(f'* parsing mask: {args.mask}')
    kept_per_layer = parse_kept_experts(args.mask, args.expert_count, args.kept_count)
    print(f'  {len(kept_per_layer)} layers, {args.kept_count} of {args.expert_count} experts kept each')

    print(f'* opening: {args.input}')
    reader = gguf.GGUFReader(args.input, 'r')
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()

    def kept_for(name: str) -> list[int] | None:
        m = EXPERT_TENSOR_RE.match(name)
        if not m:
            return None
        layer = int(m[1])
        if layer not in kept_per_layer:
            raise ValueError(f'{name}: no mask entry for layer {layer}')
        return kept_per_layer[layer]

    print(f'* writing: {args.output}')
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    # split.count / split.no / split.tensors.count are kept as-is: pruning changes tensor SHAPES, not
    # the tensor COUNT (192 tensors are resized, none added or removed), so shard1/shard2 still agree
    # on split.tensors.count and the engine's shard-pairing check still passes.
    n_pruned_tensors = 0
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == args.expert_count_key:
            print(f'  {field.name}: {value} -> {args.kept_count}')
            value = args.kept_count
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    # Pass 1: tensor info (shapes/sizes) for every tensor - required before any data is written.
    for tensor in reader.tensors:
        kept = kept_for(tensor.name)
        if kept is None:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                    tensor.data.nbytes, tensor.tensor_type)
        else:
            n_pruned_tensors += 1
            pruned_shape = (len(kept),) + tuple(tensor.data.shape[1:])
            pruned_nbytes = tensor.data.nbytes * len(kept) // tensor.data.shape[0]
            writer.add_tensor_info(tensor.name, pruned_shape, tensor.data.dtype,
                                    pruned_nbytes, tensor.tensor_type)

    print(f'  {n_pruned_tensors} tensors pruned, {len(reader.tensors) - n_pruned_tensors} copied unchanged')

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    total = sum(t.data.nbytes for t in reader.tensors)
    # after pruning, the actual bytes written for pruned tensors are less than the source; track written bytes
    written = 0
    last_pct = -1
    for tensor in reader.tensors:
        kept = kept_for(tensor.name)
        data = tensor.data if kept is None else tensor.data[kept]
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        written += tensor.data.nbytes  # progress against ORIGINAL bytes read, for a stable ETA
        pct = int(100 * written / total)
        if pct != last_pct:
            print(f'\r  writing: {pct}% ({written / 1e9:.1f} / {total / 1e9:.1f} GB read)', end='', flush=True)
            last_pct = pct
    print()

    writer.close()
    print(f'* done: {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
