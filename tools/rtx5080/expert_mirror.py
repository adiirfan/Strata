#!/usr/bin/env python3
"""Make a copy of a GGUF shard's experts on a second drive, for STRATA_EXPERT_MIRROR.

    expert_mirror.py <shard.gguf> <copy> [--share 0.63]

The engine's direct reads (STRATA_PROMPT_DIRECT=1, or an unbuffered file tier) take experts n - round(n * share) and
up of every layer from the copy, so the copy only needs those.  This writes a file the size of the shard with just
those byte ranges filled in (the rest is left as holes, which take no space): about `share` of the expert bytes.
Build it with the same --share the engine runs with, or a larger one; the engine checks the copy at start and refuses
one that lacks the experts it would read from it.  --share 1 copies every expert.
"""
import argparse
import math
import os
import re
import struct
import sys
import time

EXPERT = re.compile(r"^blk\.\d+\.ffn_(gate|up|down|gate_up)_exps\.weight$")
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_tensors(path):
    """(data offset, alignment, [(name, dims, offset in data)]) of a GGUF file (v2/v3)."""
    with open(path, "rb") as f:
        def take(fmt):
            n = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, f.read(n))

        def string():
            (n,) = take("Q")
            return f.read(n).decode("utf-8", "replace")

        def skip(t):
            if t in SCALAR:
                f.seek(struct.calcsize("<" + SCALAR[t]), 1)
            elif t == 8:
                (n,) = take("Q")
                f.seek(n, 1)
            elif t == 9:
                et, n = take("IQ")
                if et in SCALAR:
                    f.seek(n * struct.calcsize("<" + SCALAR[et]), 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                sys.exit(f"{path}: unknown GGUF value type {t}")

        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = take("IQQ")
        if version < 2:
            sys.exit(f"{path}: GGUF version {version} is too old")
        align = 32
        for _ in range(n_kv):
            key = string()
            (t,) = take("I")
            if key == "general.alignment" and t == 4:
                (align,) = take("I")
            else:
                skip(t)
        tensors = []
        for _ in range(n_tensors):
            name = string()
            (nd,) = take("I")
            dims = take("Q" * nd)
            _, off = take("IQ")
            tensors.append((name, dims, off))
        data = (f.tell() + align - 1) // align * align
    return data, align, tensors


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shard")
    ap.add_argument("copy")
    ap.add_argument("--share", type=float, default=0.5, help="the STRATA_EXPERT_MIRROR_SHARE it is for (default 0.5)")
    a = ap.parse_args()
    share = min(max(a.share, 0.0), 1.0)
    size = os.path.getsize(a.shard)
    data, align, tensors = read_tensors(a.shard)
    ends = sorted(off for _, _, off in tensors) + [size - data]
    ranges = []
    for name, dims, off in tensors:
        if not EXPERT.match(name):
            continue
        n = dims[-1]
        span = min(e for e in ends if e > off) - off
        per = span // n
        if span - per * n >= align:
            sys.exit(f"{name}: {span} bytes do not split into {n} experts")
        cut = n - math.floor(n * share + 0.5)
        if cut < n:
            ranges.append((data + off + cut * per, (n - cut) * per))
    if not ranges:
        sys.exit(f"{a.shard}: no expert tensors (blk.N.ffn_*_exps.weight)")
    total = sum(b for _, b in ranges)
    print(f"{len(ranges)} expert ranges, {total / 1e9:.2f} GB of {size / 1e9:.2f} GB", flush=True)
    t0 = time.time()
    buf = bytearray(64 << 20)
    with open(a.shard, "rb", buffering=0) as src, open(a.copy, "r+b" if os.path.exists(a.copy) else "w+b", buffering=0) as dst:
        os.ftruncate(dst.fileno(), size)
        done = 0
        for at, n in sorted(ranges):
            src.seek(at)
            dst.seek(at)
            while n > 0:
                k = src.readinto(memoryview(buf)[: min(n, len(buf))])
                if not k:
                    sys.exit(f"{a.shard}: short read at {at}")
                dst.write(memoryview(buf)[:k])
                n -= k
                done += k
            print(f"\r{done / 1e9:7.2f} / {total / 1e9:.2f} GB  {done / 1e9 / max(time.time() - t0, 1e-3):.2f} GB/s", end="", flush=True)
        os.fsync(dst.fileno())
    print(f"\ndone: STRATA_EXPERT_MIRROR={os.path.abspath(a.copy)} STRATA_EXPERT_MIRROR_SHARE={share:g} (or less)")


if __name__ == "__main__":
    main()
