# kwin.py <kernel_trace.csv> [regex]: GPU kernel time inside the prompt window (first to last expert kernel, +-0.4 s)
import csv, sys, collections, re
rows = list(csv.DictReader(open(sys.argv[1])))
ts = lambda r: int(r["Start_Timestamp"]); te = lambda r: int(r["End_Timestamp"])
w11 = [r for r in rows if "native_w11" in r["Kernel_Name"] or "mul_mat_q" in r["Kernel_Name"]]
p0 = min(map(ts, w11)) - 400_000_000; p1 = max(map(te, w11)) + 400_000_000
win = [r for r in rows if p0 <= ts(r) <= p1]
tot = sum(te(r) - ts(r) for r in win) / 1e6
agg = collections.defaultdict(float)
for r in win:
    n = r["Kernel_Name"]
    n = re.sub(r"\(.*", "", n.replace("strata::prefill::", "").replace("strata::kernels::", "").replace("(anonymous namespace)::", "").replace("void ", ""))
    n = re.sub(r"Cijk_.*", "hipBLASLt", n)
    agg[n[:60]] += (te(r) - ts(r)) / 1e6
if len(sys.argv) > 2:
    ms = sum(v for k, v in agg.items() if re.search(sys.argv[2], k))
    print("window GPU %.0f ms | '%s' %.1f ms" % (tot, sys.argv[2], ms))
else:
    print("prompt-window GPU kernel time %.0f ms (wall %.0f ms)" % (tot, (p1 - p0) / 1e6))
    for n, ms in sorted(agg.items(), key=lambda x: -x[1])[:18]:
        print("%7.0f ms %5.1f%%  %s" % (ms, 100 * ms / tot, n))
