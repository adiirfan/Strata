#!/bin/bash
# tools/rtx5080/srv5080.sh start <config> [MemoryMax e.g. 8G | none] | stop | mem
# the Strata server as a transient user service (its own cgroup: RAM incl. page cache is accounted; an optional hard
# cap with swap off), on 127.0.0.1:8080; start waits until it answers
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
U=strata5080
cg() { echo /sys/fs/cgroup$(systemctl --user show $U.service -p ControlGroup --value); }
case "$1" in
start)
  systemctl --user stop $U.service 2>/dev/null; systemctl --user reset-failed $U.service 2>/dev/null
  for i in $(seq 60); do (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null || break; sleep 1; done
  CAP=(); [ -n "$3" ] && [ "$3" != none ] && CAP=(-p MemoryMax=$3 -p MemorySwapMax=0)
  systemd-run --user --unit=$U --collect -p MemoryAccounting=yes "${CAP[@]}" --working-directory="$ROOT" \
    "$ROOT/.venv/bin/python" serve/server.py --engine strata --config "$2" --port 8080 >/dev/null 2>&1
  for i in $(seq 200); do
    curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null && { echo "up ($(( $(cat $(cg)/memory.current) / 1048576 )) MiB now)"; exit 0; }
    systemctl --user is-active -q $U.service || { echo "service stopped"; journalctl --user -u $U -n 15 --no-pager | cut -c1-200; exit 1; }
    sleep 2
  done; echo "not up";;
stop) systemctl --user stop $U.service;;
mem) C=$(cg); echo "current $(( $(cat $C/memory.current) / 1048576 )) MiB, peak $(( $(cat $C/memory.peak) / 1048576 )) MiB"
     grep -E "^(anon|file|shmem|unevictable) " $C/memory.stat | awk '{printf "%s %d MiB  ", $1, $2/1048576} END {print ""}';;
esac
