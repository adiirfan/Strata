#!/bin/bash
# tools/rtx5080/eo5080.sh <cap|none> <config> <tokens file> [K=V ...] [-- engine args]: the config's engine once on a
# token file (eo5080.py) under a RAM cap, the file cache dropped first where `sudo -n` may; the engine's output goes to
# /tmp/eo5080.log, ending with the run's peak memory (file cache included)
cd "$(dirname "$0")/../.."
CAP=$1; shift
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || echo "note: the file cache was not dropped (sudo -n)"
CAPARGS=(); [ "$CAP" != none ] && CAPARGS=(-p MemoryMax=$CAP -p MemorySwapMax=0)
systemd-run --user --scope -q -p MemoryAccounting=yes "${CAPARGS[@]}" bash -c \
  '.venv/bin/python tools/rtx5080/eo5080.py "$@"; echo "memory peak $(( $(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak) / 1048576 )) MiB"' \
  _ "$@" > /tmp/eo5080.log 2>&1
grep -E "^prefill |^decode |^memory peak" /tmp/eo5080.log | cut -c1-200
