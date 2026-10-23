#!/usr/bin/env python3
"""Losslessly screen IQ2_XS signed-nibble storage against saved model experts."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


BLOCK_BYTES = 74
NIBBLE_BYTES = 144  # d16, eight original scale bytes, six padding bytes, 128 code bytes
ALPHABET = np.array([-43, -25, -8, 8, 25, 43], dtype=np.int16)
MAGNITUDES = np.array([8, 25, 43], dtype=np.int16)


def table(source: str, name: str, count: int) -> np.ndarray:
    body = re.search(r'GGML_TABLE_BEGIN\([^,]+, ' + name + r', ' + str(count)
                     + r'\)(.*?)GGML_TABLE_END\(\)', source, re.S)
    if body is None:
        raise ValueError(f'missing {name}')
    values = [int(x, 0) for x in re.findall(r'0x[0-9a-fA-F]+|\b\d+\b', body[1])]
    if len(values) != count:
        raise ValueError(f'{name}: {len(values)} entries, expected {count}')
    return np.asarray(values, dtype=np.uint64 if name == 'iq2xs_grid' else np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('weights_json', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--blocks-per-chunk', type=int, default=2048)
    args = p.parse_args()
    if args.blocks_per_chunk < 1:
        p.error('--blocks-per-chunk must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    source_path = repo / 'third_party/ggml/ggml-common.h'
    grid_words = table(source_path.read_text(), 'iq2xs_grid', 512)
    grid = grid_words.astype('<u8', copy=False).view(np.uint8).reshape(512, 8).astype(np.int16)
    signs = table(source_path.read_text(), 'ksigns_iq2xs', 128)
    if not np.isin(grid, MAGNITUDES).all() or np.unique(signs).size != 128:
        raise ValueError('unexpected IQ2_XS grid/sign table')
    digits = np.searchsorted(MAGNITUDES, grid)
    powers = 3 ** np.arange(8, dtype=np.int32)
    keys = (digits * powers).sum(axis=1)
    if np.unique(keys).size != 512:
        raise ValueError('magnitude grid is not invertible')
    reverse = np.full(3 ** 8, 65535, dtype=np.uint16)
    reverse[keys] = np.arange(512, dtype=np.uint16)
    reverse_sign = np.full(256, 255, dtype=np.uint8)
    reverse_sign[signs] = np.arange(128, dtype=np.uint8)
    encode = np.full(256, 255, dtype=np.uint8)
    encode[(ALPHABET + 128).astype(np.uint8)] = np.arange(6, dtype=np.uint8)
    bit = np.arange(8, dtype=np.uint8)

    def convert(blocks: np.ndarray) -> np.ndarray:
        q = np.frombuffer(blocks[:, 2:66].copy().tobytes(), dtype='<u2').reshape(-1, 32)
        mag = grid[q & 511]
        sign = ((signs[q >> 9][..., None] >> bit) & 1).astype(np.int16)
        values = mag * (1 - 2 * sign)
        codes = encode[(values + 128).astype(np.uint8)]
        if (codes == 255).any():
            raise ValueError('signed IQ2_XS value is outside the six-value alphabet')
        packed = codes.reshape(-1, 256)
        nib = packed[:, ::2] | (packed[:, 1::2] << 4)
        converted = np.zeros((blocks.shape[0], NIBBLE_BYTES), dtype=np.uint8)
        converted[:, :2] = blocks[:, :2]
        converted[:, 2:10] = blocks[:, 66:74]
        converted[:, 16:] = nib

        restored_codes = np.empty_like(packed)
        restored_codes[:, ::2] = converted[:, 16:] & 15
        restored_codes[:, 1::2] = converted[:, 16:] >> 4
        restored = ALPHABET[restored_codes].reshape(-1, 32, 8)
        if not np.array_equal(restored, values):
            raise ValueError('signed nibble roundtrip changed decoded weights')
        restored_digits = np.searchsorted(MAGNITUDES, np.abs(restored))
        restored_keys = (restored_digits * powers).sum(axis=-1)
        restored_grid = reverse[restored_keys]
        restored_masks = (((restored < 0).astype(np.uint8)) << bit).sum(axis=-1)
        restored_sign = reverse_sign[restored_masks]
        if (restored_grid == 65535).any() or (restored_sign == 255).any():
            raise ValueError('cannot restore canonical IQ2_XS packets')
        restored_q = restored_grid | (restored_sign.astype(np.uint16) << 9)
        if not np.array_equal(restored_q, q):
            raise ValueError('canonical GGUF packets changed')
        if not np.array_equal(converted[:, :2], blocks[:, :2]) or not np.array_equal(converted[:, 2:10], blocks[:, 66:74]):
            raise ValueError('original scale metadata changed')
        return converted

    # Exhaust all grid/sign combinations before inspecting model samples.
    exhaustive_q = (np.tile(np.arange(512, dtype=np.uint16), 128)
                    | (np.repeat(np.arange(128, dtype=np.uint16), 512) << 9))
    exhaustive = np.zeros((exhaustive_q.size, BLOCK_BYTES), dtype=np.uint8)
    exhaustive[:, 2:66] = np.repeat(exhaustive_q[:, None], 32, axis=1).view(np.uint8).reshape(-1, 64)
    exhaustive[:, 66:74] = np.arange(8, dtype=np.uint8)
    for at in range(0, len(exhaustive), args.blocks_per_chunk):
        convert(exhaustive[at:at + args.blocks_per_chunk])

    inventory = json.loads(args.weights_json.read_text())
    totals = {'experts': 0, 'iq2_blocks': 0, 'original_bytes': 0, 'converted_bytes': 0}
    layers = []
    sample_path = args.out / 'converted-gu-all.bin'
    original_path = args.out / 'original-gu-all.bin'
    with sample_path.open('wb') as sample, original_path.open('wb') as originals:
        for layer in inventory['layers']:
            if layer['gu'] != 17:
                raise ValueError(f'unexpected non-IQ2_XS gate/up at layer {layer["layer"]}')
            path = Path(layer['path'])
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != layer['sha256']:
                raise ValueError(f'changed model sample: {path}')
            if len(data) != layer['blob_bytes'] * len(layer['experts']):
                raise ValueError('blob size mismatch')
            gu_bytes = 2 * 640 * (2560 // 256) * BLOCK_BYTES
            digest = hashlib.sha256()
            for i in range(len(layer['experts'])):
                gu = data[i * layer['blob_bytes']:i * layer['blob_bytes'] + gu_bytes]
                originals.write(gu)
                blocks = np.frombuffer(gu, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
                for at in range(0, len(blocks), args.blocks_per_chunk):
                    converted = convert(blocks[at:at + args.blocks_per_chunk])
                    raw = converted.tobytes()
                    digest.update(raw)
                    sample.write(raw)
                    totals['iq2_blocks'] += len(converted)
                    totals['original_bytes'] += len(converted) * BLOCK_BYTES
                    totals['converted_bytes'] += len(raw)
                totals['experts'] += 1
            layers.append({'layer': layer['layer'], 'experts': layer['experts'],
                           'converted_gu_sha256': digest.hexdigest()})
    report = {'status': 'lossless_cpu_screen_complete', 'adopted': False,
              'exhaustive_grid_sign_pairs': int(exhaustive_q.size),
              'original_block_bytes': BLOCK_BYTES, 'converted_block_bytes': NIBBLE_BYTES,
              'totals': totals, 'layers': layers,
              'sample_path': str(sample_path),
              'sample_sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(),
              'original_path': str(original_path),
              'original_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(),
              'inputs': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (Path(__file__), source_path, args.weights_json)},
              'limits': 'CPU codec/metadata equality only; no GPU kernel, cache-residency or inference result yet.'}
    args.out.joinpath('report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'exhaustive_grid_sign_pairs', 'totals', 'sample_sha256')}, indent=2))


if __name__ == '__main__':
    main()
