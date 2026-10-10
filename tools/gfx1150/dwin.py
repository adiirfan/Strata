# dwin.py <kernel_trace.csv> [top]: GPU kernel time in the writing window (after the last prompt kernel)
import csv, sys, collections, re
rows = list(csv.DictReader(open(sys.argv[1])))
ts = lambda r: int(r["Start_Timestamp"]); te = lambda r: int(r["End_Timestamp"])
rows.sort(key=ts)
pf = [r for r in rows if "native_w11" in r["Kernel_Name"] or "prefill" in r["Kernel_Name"]]
p1 = max(map(te, pf)) + 300_000_000
win = [r for r in rows if ts(r) > p1]
tot = sum(te(r) - ts(r) for r in win) / 1e6
wall = (te(win[-1]) - ts(win[0])) / 1e6
agg = collections.defaultdict(lambda: [0.0, 0])
for r in win:
    n = r["Kernel_Name"]
    n = re.sub(r"\(.*", "", n.replace("strata::kernels::", "").replace("strata::", "").replace("(anonymous namespace)::", "").replace("void ", ""))
    n = re.sub(r"Cijk_.*", "hipBLASLt", n)
    a = agg[n[:70]]; a[0] += (te(r) - ts(r)) / 1e6; a[1] += 1
# gaps
gap = 0.0; prev = None; big = 0.0
for r in win:
    if prev is not None and ts(r) > prev:
        g = (ts(r) - prev) / 1e6; gap += g
        if g > 1.0: big += g
    prev = max(prev or 0, te(r))
print("writing window: wall %.0f ms, GPU busy %.0f ms (%.0f%%), idle gaps %.0f ms (of which gaps >1 ms: %.0f ms), %d kernels" % (wall, tot, 100 * tot / wall, gap, big, len(win)))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
for n, (ms, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:top]:
    print("%7.1f ms %5.1f%% %6d calls %7.1f us/call  %s" % (ms, 100 * ms / tot, c, 1000 * ms / c, n))
