#!/usr/bin/env python3
"""Expert-pruned Q2_0 models: keep N of each layer's 512 experts (default 256 and 384), source quantization kept.

    python tools/prune_q2_0.py --gguf <models>/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \\
        --mask <Coder release>/tensor-allocation/...rco-allocation.txt          (optional, see below)

For each N this writes, under --out (default: <the model's folder>/pruned):
  Qwen3.8-...-Q2_0-prunedN-00001-of-00002.gguf   shard 1 with N experts per layer (tools/prune_experts_from_mask.py)
  Qwen3.8-...-Q2_0-prunedN-00002-of-00002.gguf   the PLE shard, hard-linked (it has no experts; symlinked across
                                                 file systems)
  maskN.rco-allocation.txt                       the kept experts, in the release's section-2 format
  expert-profile-prunedN.bin                     data/expert-profile.bin mapped onto the kept experts
and builds the pack (tools/iq_pack.py + the tokenizer) under --packs (default: <the model's folder>/../../packs)
as q2_0-prunedN, then prints the engine flags.

Which experts are kept:
  --mask given (ISTA-DASLab's Coder release, `tensor-allocation/*.rco-allocation.txt`, 256 per layer, chosen with
  RCO on code, agentic and vision data): N >= 256 keeps the mask's 256 plus each layer's next N - 256 by the
  shipped profile's rank, so every larger model holds the smaller ones' experts; N < 256 keeps the mask's top N
  by that rank.
  no --mask: each layer's top N by the shipped profile's rank (data/expert-profile.bin).

Pass the pruned shard 2 as --ple-gguf (the link beside shard 1): the engine also opens the shard beside --native,
and the original shard 2 under another name counts twice ("duplicate split shard number").

Measured on an RTX 2060 SUPER 8 GB (bench, 2026-10-07; teacher-forced KL vs the full model over ~3,300 positions):
512 (37.6 GB): decode 45.4 tok/s; 384 (29.1 GB): 43.1, mean KL 0.094, argmax 86-99% the same;
256 (20.6 GB): 47.8, mean KL 0.35, argmax 79-89% the same. Prefill is 16-21% faster pruned.
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ENGINE = TOOLS.parent
sys.path.insert(0, str(TOOLS))
from prune_experts_from_mask import prune  # noqa: E402
from make_profile import MAGIC, VERSION  # noqa: E402

N_EXPERT = 512
MASK_LINE_RE = re.compile(r'^blk\.(\d+)\.(ffn_down_exps|ffn_gate_exps|ffn_up_exps)\.weight\[(\d+)\]: (.*)$')


def read_profile(path: Path) -> tuple[int, int, list[tuple[int, int]]]:
    blob = path.read_bytes()
    if blob[:4] != MAGIC:
        sys.exit(f'{path}: not an expert profile (no {MAGIC!r} header)')
    _ver, n_layer, n_expert, _slots, n = struct.unpack_from('<5I', blob, 4)
    return n_layer, n_expert, [struct.unpack_from('<HH', blob, 24 + 4 * i) for i in range(n)]


def read_mask(path: Path) -> dict[int, set[int]]:
    """{layer: kept original indices} from an rco-allocation.txt (an empty type = pruned)."""
    kept: dict[int, set[int]] = {}
    for line in open(path):
        m = MASK_LINE_RE.match(line.rstrip('\n'))
        if m and m[2] == 'ffn_down_exps' and m[4] != '':
            kept.setdefault(int(m[1]), set()).add(int(m[3]))
    return kept


def choose(n: int, n_layer: int, n_expert: int, ranked: list[tuple[int, int]],
           mask: dict[int, set[int]] | None) -> dict[int, list[int]]:
    rank = {}
    for i, pair in enumerate(ranked):
        rank.setdefault(pair, i)
    kept = {}
    for layer in range(n_layer):
        order = sorted(range(n_expert), key=lambda e: rank.get((layer, e), len(ranked) + e))
        if mask is None:
            pick = order[:n]
        else:
            base = mask.get(layer)
            if not base:
                sys.exit(f'the mask has no experts for layer {layer}')
            if n <= len(base):
                pick = [e for e in order if e in base][:n]
            else:
                pick = list(base) + [e for e in order if e not in base][:n - len(base)]
        kept[layer] = sorted(pick)
    return kept


def write_mask(path: Path, kept: dict[int, list[int]], n_expert: int) -> None:
    """Section 2 of the release's format, for prune_experts_from_mask.py --mask (type 'Q2_0' = kept)."""
    with open(path, 'w') as f:
        for layer in sorted(kept):
            k = set(kept[layer])
            for t in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
                for e in range(n_expert):
                    f.write(f'blk.{layer}.{t}.weight[{e}]: {"Q2_0" if e in k else ""}\n')


def write_pruned_profile(path: Path, ranked: list[tuple[int, int]], kept: dict[int, list[int]], n_layer: int,
                         n: int) -> None:
    """The shipped ranking with each kept pair renamed to its new index, pruned pairs dropped (as
    data/expert-profile-coder.bin was made); the per-layer table is each expert's new rank."""
    new_index = {layer: {e: i for i, e in enumerate(ks)} for layer, ks in kept.items()}
    seen, out = set(), []
    for layer, e in ranked:
        if layer in new_index and e in new_index[layer] and (layer, e) not in seen:
            seen.add((layer, e))
            out.append((layer, new_index[layer][e]))
    for layer in range(n_layer):          # experts the profile never ranked go last, in index order
        for i in range(n):
            if (layer, kept[layer][i]) not in seen:
                out.append((layer, i))
    table = [[-1] * n for _ in range(n_layer)]
    for slot, (layer, e) in enumerate(out):
        table[layer][e] = slot
    with open(path, 'wb') as f:
        f.write(MAGIC + struct.pack('<5I', VERSION, n_layer, n, len(out), len(out)))
        for layer, e in out:
            f.write(struct.pack('<HH', layer, e))
        for layer in range(n_layer):
            f.write(struct.pack('<%di' % n, *table[layer]))


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src.resolve())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gguf', type=Path, required=True, help="the Q2_0 model's shard 1 (-00001-of-00002.gguf)")
    ap.add_argument('--keep', type=int, nargs='+', default=[256, 384], help='experts kept per layer (default 256 384)')
    ap.add_argument('--mask', type=Path, help="the Coder release's rco-allocation.txt (256 per layer)")
    ap.add_argument('--profile', type=Path, default=ENGINE / 'data' / 'expert-profile.bin')
    ap.add_argument('--out', type=Path, help='output folder (default: <the model folder>/pruned)')
    ap.add_argument('--packs', type=Path, help='pack folder (default: <the model folder>/../../packs)')
    ap.add_argument('--no-pack', action='store_true', help='only write the GGUFs, masks and profiles')
    ap.add_argument('--force', action='store_true', help='rewrite pruned shards that already exist')
    a = ap.parse_args()

    shard1 = a.gguf.resolve()
    if '-00001-of-' not in shard1.name:
        sys.exit(f'{shard1.name}: pass the model\'s shard 1 (...-00001-of-00002.gguf)')
    shard2 = shard1.with_name(shard1.name.replace('-00001-of-', '-00002-of-'))
    if not shard2.exists():
        sys.exit(f'{shard2} is missing (the PLE shard, needed beside every pruned shard 1)')
    out = (a.out or shard1.parent / 'pruned').resolve()
    packs = (a.packs or shard1.parent.parent.parent / 'packs').resolve()
    out.mkdir(parents=True, exist_ok=True)

    n_layer, n_expert, ranked = read_profile(a.profile)
    if n_expert != N_EXPERT:
        sys.exit(f'{a.profile} is a profile for {n_expert} experts per layer; this tool prunes the 512-expert model')
    mask = read_mask(a.mask) if a.mask else None
    for n in a.keep:
        if not 0 < n < n_expert:
            sys.exit(f'--keep {n}: must be 1..{n_expert - 1}')

    env = dict(os.environ)
    try:
        from _paths import gguf_py
        env.setdefault('STRATA_GGUF_PY', gguf_py())
    except SystemExit:
        pass
    flags = []
    for n in sorted(a.keep):
        tag = f'pruned{n}'
        print(f'\n=== {n} of {n_expert} experts per layer ===', flush=True)
        kept = choose(n, n_layer, n_expert, ranked, mask)
        write_mask(out / f'mask{n}.rco-allocation.txt', kept, n_expert)
        p1 = out / shard1.name.replace('-00001-of-', f'-{tag}-00001-of-')
        p2 = out / shard2.name.replace('-00002-of-', f'-{tag}-00002-of-')
        if p1.exists() and not a.force:
            print(f'* {p1.name} exists, kept (--force rewrites it)')
        else:
            prune(shard1, p1, kept, n)
        link(shard2, p2)
        prof = out / f'expert-profile-{tag}.bin'
        write_pruned_profile(prof, ranked, kept, n_layer, n)
        print(f'* profile: {prof}')
        pack = packs / f'q2_0-{tag}'
        if not a.no_pack:
            subprocess.run([sys.executable, str(TOOLS / 'iq_pack.py'), '--gguf', str(p1), '--out', str(pack)],
                           env=env, check=True)
            if not (pack / 'tokenizer' / 'vocab.json').exists():
                subprocess.run([sys.executable, str(TOOLS / 'strata_tokenizer.py'), '--gguf', str(p1),
                                '--out', str(pack)], env=env, check=True)
        flags.append((n, f'--pack {pack} --native {p1} --ple-gguf {p2} --expert-profile {prof}'))

    print('\nEngine flags (in place of the full model\'s; everything else unchanged):')
    for n, f in flags:
        print(f'  {n}: {f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
