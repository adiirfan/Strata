#!/bin/bash
# dab.sh <rounds> <tokens file> "<label>|<K=V K=V>|<extra args>" ... : alternate engine settings, MAX_NEW tokens each,
# print writing speed, drafts, and a hash of the output tokens (same hash = same words)
cd ~/Strata
n=$1; tf=$2; shift 2
for r in $(seq 1 $n); do
  for spec in "$@"; do
    IFS='|' read -r label envs extra <<< "$spec"
    STRATA_ENV_EXTRA="$envs" MAX_NEW=${MAX_NEW:-256} timeout 300 flock ~/gpu.lock .venv/bin/python engine_once.py $tf $extra > /tmp/dab.log 2>&1
    dec=$(grep -oE "^decode .*-> +[0-9.]+ tok/s" /tmp/dab.log | grep -oE "[0-9.]+ tok/s")
    acc=$(grep -oE "drafts accepted [0-9]+ of [0-9]+ \([0-9.]+\), [0-9.]+ tokens per round" /tmp/dab.log | head -1)
    mtp=$(grep -oE "^mtp +[0-9.]+ ms/round" /tmp/dab.log | grep -oE "[0-9.]+ ms")
    h=$(grep "^output" /tmp/dab.log | md5sum | cut -c1-8)
    printf "%-10s %-12s %s | mtp %s | out %s\n" "$label" "$dec" "$acc" "$mtp" "$h"
    [ -z "$dec" ] && tail -5 /tmp/dab.log
  done
done
