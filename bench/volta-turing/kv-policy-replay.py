#!/usr/bin/env python3
"""Replay page selections through Strata-like CLOCK and LRU screens.

Input is JSONL with one {"blocks": [unique page ids]} record per resolve call.
The parity harness writes this format with STRATA_KV_SELECTION_TRACE=PATH.
No model or GPU memory is needed for replay. The probe policy is a CPU screen,
not an implementation of the upstream Triton kernel or a performance claim.
"""

import argparse
from array import array
import json
from collections import OrderedDict
import struct
import sys


class Clock:
    def __init__(self, slots):
        self.slots = slots
        self.pages = [None] * slots
        self.ref = [False] * slots
        self.lookup = {}
        self.hand = 0
        self.hits = self.misses = self.evictions = 0

    def step(self, blocks):
        selected = set(blocks)
        assert len(selected) <= self.slots, "selection exceeds slot capacity"
        missing = []
        for block in blocks:
            slot = self.lookup.get(block)
            if slot is None:
                missing.append(block)
                self.misses += 1
            else:
                self.ref[slot] = True
                self.hits += 1
        for block in missing:
            for _ in range(3 * self.slots):
                slot = self.hand
                self.hand = (self.hand + 1) % self.slots
                old = self.pages[slot]
                if old in selected:
                    continue
                if old is not None and self.ref[slot]:
                    self.ref[slot] = False
                    continue
                if old is not None:
                    del self.lookup[old]
                    self.evictions += 1
                self.pages[slot] = block
                self.lookup[block] = slot
                self.ref[slot] = True
                break
            else:
                raise RuntimeError("CLOCK found no eligible victim")


class IdealLRU:
    def __init__(self, slots):
        self.slots = slots
        self.pages = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def step(self, blocks):
        selected = set(blocks)
        assert len(selected) <= self.slots, "selection exceeds slot capacity"
        missing = []
        for block in blocks:
            if block in self.pages:
                self.pages.move_to_end(block)
                self.hits += 1
            else:
                missing.append(block)
                self.misses += 1
        for block in missing:
            if len(self.pages) == self.slots:
                victim = next(page for page in self.pages if page not in selected)
                del self.pages[victim]
                self.evictions += 1
            self.pages[block] = None


class ProbeLRU:
    """16 probes in each of up to 64 slot ranges; miss ordinal assigns a hand."""

    def __init__(self, slots):
        self.slots = slots
        self.pages = [None] * slots
        self.stamp = [-1] * slots
        self.lookup = {}
        self.epoch = 0
        self.hands = 1
        while self.hands < 64 and slots // (self.hands * 2) >= 16:
            self.hands *= 2
        self.hand = [0] * self.hands
        self.hits = self.misses = self.evictions = self.fallbacks = 0

    def step(self, blocks):
        selected = set(blocks)
        assert len(selected) <= self.slots, "selection exceeds slot capacity"
        self.epoch += 1
        missing = []
        for block in blocks:
            slot = self.lookup.get(block)
            if slot is None:
                missing.append(block)
                self.misses += 1
            else:
                self.stamp[slot] = self.epoch
                self.hits += 1
        for ordinal, block in enumerate(missing):
            hand = ordinal % self.hands
            start = hand * self.slots // self.hands
            end = (hand + 1) * self.slots // self.hands
            span = end - start
            candidates = []
            for offset in range(0, span, 16):
                base = self.hand[hand] + offset
                candidates = [start + (base + i) % span for i in range(min(16, span))]
                candidates = [slot for slot in candidates if self.pages[slot] not in selected]
                if candidates:
                    self.hand[hand] = (base + 16) % span
                    break
            if not candidates:
                self.fallbacks += 1
                candidates = [slot for slot in range(self.slots) if self.pages[slot] not in selected]
            slot = min(candidates, key=lambda s: (self.pages[s] is not None, self.stamp[s], s))
            old = self.pages[slot]
            if old is not None:
                del self.lookup[old]
                self.evictions += 1
            self.pages[slot] = block
            self.lookup[block] = slot
            self.stamp[slot] = self.epoch


def replay(records, slots, block_bytes):
    policies = {"clock": Clock(slots), "ideal_lru": IdealLRU(slots), "probe_lru_16": ProbeLRU(slots)}
    calls = max_blocks = 0
    gpu_misses = None
    for record in records:
        blocks = list(dict.fromkeys(record["blocks"]))
        if any(not isinstance(b, int) or b < 0 for b in blocks):
            raise ValueError("block ids must be nonnegative integers")
        calls += 1
        max_blocks = max(max_blocks, len(blocks))
        gpu_misses = record.get("gpu_misses_total", gpu_misses)
        for policy in policies.values():
            policy.step(blocks)
    return {
        "calls": calls,
        "slots": slots,
        "max_blocks_per_call": max_blocks,
        "gpu_misses_total": gpu_misses,
        "policies": {
            name: {
                "hits": p.hits,
                "misses": p.misses,
                "evictions": p.evictions,
                "estimated_host_bytes": p.misses * block_bytes,
                **({"probe_fallbacks": p.fallbacks} if isinstance(p, ProbeLRU) else {}),
            }
            for name, p in policies.items()
        },
    }


def qsa_dump_records(path, page_size):
    """Actual prompt selections from STRATA_QSA_DUMP_STEPS=1, one query per record.

    This is a what-if cache replay. Prompt attention does not call the streamed
    decode resolver, so these records have no GPU miss count to validate against.
    """
    if sys.byteorder != "little" or array("i").itemsize != 4:
        raise RuntimeError("the QSA dump contains little-endian 32-bit integers")
    with open(path, "rb") as source:
        while header := source.read(16):
            if len(header) != 16:
                raise ValueError("truncated QSA dump header")
            layer, pos0, n_q, cap = struct.unpack("<4i", header)
            if layer < 0 or pos0 < 0 or not 0 < n_q <= 65536 or not 0 < cap <= 65536:
                raise ValueError(f"invalid QSA dump record: {(layer, pos0, n_q, cap)}")
            ids_raw = source.read(n_q * cap * 4)
            steps_raw = source.read(n_q * 4 * 4)
            if len(ids_raw) != n_q * cap * 4 or len(steps_raw) != n_q * 4 * 4:
                raise ValueError("truncated QSA dump selection/steps")
            ids, steps = array("i"), array("i")
            ids.frombytes(ids_raw)
            steps.frombytes(steps_raw)
            for q in range(n_q):
                pos, n_kv, _, width = steps[q * 4:q * 4 + 4]
                if pos != pos0 + q or n_kv != pos + 1 or not 0 < width <= cap:
                    raise ValueError(f"invalid QSA step at layer {layer}, query {q}: {(pos, n_kv, width)}")
                cells = ids[q * cap:q * cap + width]
                if any(cell < 0 or cell > pos for cell in cells):
                    raise ValueError(f"out-of-range QSA cell at layer {layer}, query {q}")
                yield {"blocks": list(dict.fromkeys(cell // page_size for cell in cells)),
                       "layer": layer, "pos": pos}


def selftest():
    simple = [{"blocks": [0]}, {"blocks": [1]}, {"blocks": [0]}, {"blocks": [2]}]
    result = replay(simple, 2, 3264)
    for policy in result["policies"].values():
        assert policy["hits"] == 1 and policy["misses"] == 3, policy
    repeated = replay([{"blocks": [0, 0]}, {"blocks": [0]}], 2, 3264)
    for policy in repeated["policies"].values():
        assert policy["hits"] == 1 and policy["misses"] == 1, policy
    print("PASS")


def synthetic(scenario, calls):
    """Bounded policy stress cases, not recordings of model inference."""
    for call in range(calls):
        context = (call // 4) % 4
        offset = ((call // 16) * 127) % 3584
        working = [10000 + context * 8192 + (offset + i) % 4096 for i in range(1536)]
        if scenario == "shared-prefix":
            yield {"blocks": list(range(1024)) + working[:512]}
        else:
            yield {"blocks": working}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", help="JSONL page-selection trace")
    parser.add_argument("--slots", type=int, default=4828)
    parser.add_argument("--block-bytes", type=int, default=3264, help="K8V4 bytes per page")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--synthetic", choices=("shared-prefix", "disjoint-context"))
    parser.add_argument("--qsa-dump", help="prompt selection dump made with STRATA_QSA_DUMP_STEPS=1")
    parser.add_argument("--page-size", type=int, default=4, help="cells per KV page in --qsa-dump")
    parser.add_argument("--calls", type=int, default=160)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.slots < 1 or args.block_bytes < 1 or args.calls < 1 or args.page_size < 1:
        parser.error("slots, block bytes, calls, and page size must be positive")
    if args.qsa_dump:
        if args.trace or args.synthetic:
            parser.error("choose one trace input")
        report = replay(qsa_dump_records(args.qsa_dump, args.page_size), args.slots, args.block_bytes)
        report["source_kind"] = "real prompt selections; hypothetical sequential-query cache replay, not streamed decode"
        print(json.dumps(report, indent=2))
        return
    if args.synthetic:
        if args.trace:
            parser.error("choose a trace or --synthetic")
        print(json.dumps(replay(synthetic(args.synthetic, args.calls), args.slots, args.block_bytes), indent=2))
        return
    if not args.trace:
        parser.error("a trace or --synthetic is required")
    with open(args.trace, encoding="utf-8") as source:
        records = (json.loads(line) for line in source if line.strip())
        print(json.dumps(replay(records, args.slots, args.block_bytes), indent=2))


if __name__ == "__main__":
    main()
