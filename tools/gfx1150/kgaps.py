# kgaps.py <kernel_trace.csv>: idle gaps between kernels inside the prompt window, by the kernel that follows them
import csv, sys, collections, re
rows = list(csv.DictReader(open(sys.argv[1])))
ts = lambda r: int(r["Start_Timestamp"]); te = lambda r: int(r["End_Timestamp"])
w11 = [r for r in rows if "native_w11" in r["Kernel_Name"]]
p0, p1 = min(map(ts, w11)), max(map(te, w11))
win = sorted([r for r in rows if p0 <= ts(r) <= p1], key=ts)
gaps = collections.defaultdict(lambda: [0, 0.0])
end = te(win[0]); total = 0.0; big = []
nm = lambda r: re.sub(r"\(.*", "", r["Kernel_Name"].replace("strata::prefill::", "").replace("strata::kernels::", "").replace("void ", "").replace("(anonymous namespace)::", ""))[:50]
for prev, r in zip(win, win[1:]):
    g = ts(r) - end
    if g > 0:
        total += g / 1e6
        k = nm(prev) + " -> " + nm(r)
        gaps[k][0] += 1; gaps[k][1] += g / 1e6
        if g > 2e6: big.append((g / 1e6, k))
    end = max(end, te(r))
print("window %.0f ms, idle %.0f ms" % ((p1 - p0) / 1e6, total))
for k, (n, ms) in sorted(gaps.items(), key=lambda x: -x[1][1])[:14]:
    print("%7.1f ms %5d gaps  %s" % (ms, n, k))
print("gaps over 2 ms:", len(big), sorted(big, reverse=True)[:5])
