# drounds.py <kernel_trace.csv>: split the writing window into rounds (at mtp_select_kernel), per round: the verify
# window size T (from gr_down_staged_kernel<T>), wall time, verify and draft parts
import csv, sys, re, collections, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
ts = lambda r: int(r["Start_Timestamp"]); te = lambda r: int(r["End_Timestamp"])
rows.sort(key=ts)
pf = [r for r in rows if "native_w11" in r["Kernel_Name"] or "prefill" in r["Kernel_Name"]]
p1 = max(map(te, pf)) + 300_000_000
win = [r for r in rows if ts(r) > p1]
# a round: from the first kernel after an mtp_select_kernel to the next mtp_select_kernel
cuts = [i for i, r in enumerate(win) if "mtp_select_kernel" in r["Kernel_Name"]]
by = collections.defaultdict(list)
for a, b in zip(cuts, cuts[1:]):
    seg = win[a + 1:b + 1]
    Ts = [int(m.group(1)) for r in seg for m in [re.search(r"gr_down_staged_kernel<(\d)", r["Kernel_Name"])] if m]
    if not Ts: continue
    T = max(set(Ts), key=Ts.count)
    # draft part: from the first MTP kernel (heuristic: the first kernel after the last gr_* of the main pass whose
    # name has mtp / draft) - use the last 'combine' of the main model instead: kernels after the head's top-k
    wall = (te(seg[-1]) - ts(seg[0])) / 1e6
    busy = sum(te(r) - ts(r) for r in seg) / 1e6
    n = len(seg)
    by[T].append((wall, busy, n))
for T in sorted(by):
    v = by[T]
    print("T=%d rounds %3d | wall %.1f ms (median %.1f) | GPU busy %.1f ms | %d kernels" % (
        T, len(v), statistics.mean(x[0] for x in v), statistics.median(x[0] for x in v),
        statistics.mean(x[1] for x in v), statistics.mean(x[2] for x in v)))
