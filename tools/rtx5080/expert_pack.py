#!/usr/bin/env python3
"""Write a GGUF shard's experts to a second drive as whole blobs, for STRATA_EXPERT_MIRROR.

    expert_pack.py <shard.gguf> <out> [--share 0.75]

The engine assembles each expert as one blob - its gate, up and down slices, in that order - out of three scattered
slices of the shard. This file stores experts n - round(n * share) .. n-1 of every layer as such blobs, each starting
on a 4 KiB boundary, so the engine reads a blob from it in one request. QLC drives serve one 2 MB request faster than
three scattered 0.7 MB ones. Give the file to the engine as STRATA_EXPERT_MIRROR (it tells this layout from a plain
copy by its header) with STRATA_EXPERT_MIRROR_SHARE / _DECODE_SHARE at or below the share it was written with.

Layout: a 4 KiB-aligned header - "STRPACK1", u32 version (1), n_layers, n_expert, first expert, u64 shard size, then
per layer u64 blob bytes, stride (blob bytes rounded up to 4 KiB) and the file offset of the layer's first blob.
"""
import argparse
import math
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_mirror import read_tensors  # noqa: E402

ROLE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
A = 4096


def up4k(x):
    return (x + A - 1) // A * A


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shard")
    ap.add_argument("out")
    ap.add_argument("--share", type=float, default=0.75)
    a = ap.parse_args()
    size = os.path.getsize(a.shard)
    data, align, tensors = read_tensors(a.shard)
    ends = sorted(off for _, _, off in tensors) + [size - data]
    layers, n_expert = {}, None
    for name, dims, off in tensors:
        m = ROLE.match(name)
        if not m:
            continue
        n = dims[-1]
        n_expert = n if n_expert is None else n_expert
        if n != n_expert:
            sys.exit(f"{name}: {n} experts, others have {n_expert}")
        span = min(e for e in ends if e > off) - off
        per = span // n
        if span - per * n >= align:
            sys.exit(f"{name}: {span} bytes do not split into {n} experts")
        layers.setdefault(int(m.group(1)), {})[m.group(2)] = (data + off, per)
    if not layers or any(len(r) != 3 for r in layers.values()):
        sys.exit(f"{a.shard}: every layer needs ffn_gate_exps, ffn_up_exps and ffn_down_exps in this shard")
    n_layers = max(layers) + 1
    if sorted(layers) != list(range(n_layers)):
        sys.exit(f"{a.shard}: layers {sorted(set(range(n_layers)) - set(layers))} have no experts here")
    first = n_expert - math.floor(n_expert * min(max(a.share, 0.0), 1.0) + 0.5)
    count = n_expert - first
    head = 8 + 4 * 4 + 8 + n_layers * 24
    at = up4k(head)
    table = []
    for L in range(n_layers):
        blob = sum(layers[L][r][1] for r in ("gate", "up", "down"))
        table.append((blob, up4k(blob), at))
        at += up4k(blob) * count
    total = at
    print(f"{n_layers} layers, experts {first}..{n_expert - 1}: {total / 1e9:.2f} GB", flush=True)
    hdr = b"STRPACK1" + struct.pack("<IIII", 1, n_layers, n_expert, first) + struct.pack("<Q", size)
    for blob, stride, off in table:
        hdr += struct.pack("<QQQ", blob, stride, off)
    t0 = time.time()
    with open(a.shard, "rb", buffering=0) as src, open(a.out, "wb", buffering=0) as dst:
        dst.write(hdr + bytes(up4k(len(hdr)) - len(hdr)))
        for L in range(n_layers):
            blob, stride, off = table[L]
            parts = []
            for r in ("gate", "up", "down"):
                o, per = layers[L][r]
                src.seek(o + first * per)
                buf = src.read(per * count)
                if len(buf) != per * count:
                    sys.exit(f"{a.shard}: short read in layer {L}")
                parts.append((buf, per))
            out = bytearray(stride * count)
            for i in range(count):
                p = i * stride
                for buf, per in parts:
                    out[p:p + per] = buf[i * per:(i + 1) * per]
                    p += per
            if dst.tell() != off:
                sys.exit(f"layer {L}: at {dst.tell()}, expected {off}")
            dst.write(out)
            print(f"\rlayer {L + 1}/{n_layers}  {dst.tell() / 1e9:.2f} GB  {dst.tell() / 1e9 / max(time.time() - t0, 1e-3):.2f} GB/s",
                  end="", flush=True)
        os.fsync(dst.fileno())
    print(f"\ndone: STRATA_EXPERT_MIRROR={os.path.abspath(a.out)} (shares up to {count / n_expert:g})")


if __name__ == "__main__":
    main()
