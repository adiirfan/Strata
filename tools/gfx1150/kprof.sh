#!/bin/bash
# kprof.sh <label> <kernel regex> [K=V ...]: one 6,144-token prompt under the profiler (GPU lock held), the matching
# kernels' total time and the prefill rate.  ENGINE_EXE overrides the engine.
cd ~/Strata
label=$1; pat=$2; shift 2
R=$(cat ~/rocm/rocm10-root.txt)
rm -rf /tmp/kp
STRATA_ENV_EXTRA="$*" PROFILER="$R/bin/rocprofv3 --kernel-trace --stats -d /tmp/kp -o run --output-format csv --" \
  LD_LIBRARY_PATH=$R/lib:$R/lib/rocm_sysdeps/lib:$R/lib/llvm/lib flock ~/gpu.lock .venv/bin/python engine_once.py tokens-6144.txt > /tmp/kp.log 2>&1
rate=$(grep -oE "[0-9.]+ tok/s" /tmp/kp.log | tail -1)
python3 - /tmp/kp/run_kernel_stats.csv "$pat" "$label" "$rate" <<'PY'
import csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["TotalDurationNs"]) for r in rows) / 1e6
hit = [r for r in rows if re.search(sys.argv[2], r["Name"])]
ms = sum(float(r["TotalDurationNs"]) for r in hit) / 1e6
calls = sum(int(r["Calls"]) for r in hit)
print("%-14s %-8s kernels '%s': %7.1f ms over %d calls | all GPU %7.0f ms | prefill %s" % (sys.argv[3], "", sys.argv[2], ms, calls, tot, sys.argv[4]))
PY
