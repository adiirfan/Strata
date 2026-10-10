#!/bin/bash
# kdec.sh <label> <kernel regex> [K=V ...]: a 256-token writing run under the profiler (GPU lock held): the matching
# kernels' time in the writing window, the window's GPU total, and the engine's writing rate.  ENGINE_EXE overrides.
cd ~/Strata
label=$1; pat=$2; shift 2
R=$(cat ~/rocm/rocm10-root.txt)
rm -rf /tmp/kdd
STRATA_ENV_EXTRA="$*" MAX_NEW=256 PROFILER="$R/bin/rocprofv3 --kernel-trace -d /tmp/kdd -o run --output-format csv --" \
  LD_LIBRARY_PATH=$R/lib:$R/lib/rocm_sysdeps/lib:$R/lib/llvm/lib timeout 400 flock ~/gpu.lock .venv/bin/python engine_once.py tokens-2048.txt $KDEC_ARGS > /tmp/kdd.log 2>&1
cp /tmp/kdd/run_kernel_trace.csv /tmp/kdd_$label.csv 2>/dev/null
rate=$(grep -oE "^decode .*-> +[0-9.]+ tok/s" /tmp/kdd.log | grep -oE "[0-9.]+ tok/s")
python3 - /tmp/kdd/run_kernel_trace.csv "$pat" "$label" "$rate" <<'PY'
import csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ts = lambda r: int(r["Start_Timestamp"]); te = lambda r: int(r["End_Timestamp"])
pf = [r for r in rows if "native_w11" in r["Kernel_Name"] or "prefill" in r["Kernel_Name"]]
p1 = max(map(te, pf)) + 300_000_000
win = [r for r in rows if ts(r) > p1]
tot = sum(te(r) - ts(r) for r in win) / 1e6
hit = [r for r in win if re.search(sys.argv[2], r["Kernel_Name"])]
ms = sum(te(r) - ts(r) for r in hit) / 1e6
print("%-10s '%s': %7.1f ms over %d calls | writing-window GPU %7.0f ms | %s" % (sys.argv[3], sys.argv[2], ms, len(hit), tot, sys.argv[4]))
PY
