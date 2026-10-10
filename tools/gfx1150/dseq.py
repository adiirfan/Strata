# dseq.py <kernel_trace.csv> <start index in the writing window> <count>: the kernel sequence with durations and gaps
import csv, sys, re
rows = list(csv.DictReader(open(sys.argv[1])))
ts = lambda r: int(r["Start_Timestamp"]); te = lambda r: int(r["End_Timestamp"])
rows.sort(key=ts)
pf = [r for r in rows if "native_w11" in r["Kernel_Name"] or "prefill" in r["Kernel_Name"]]
p1 = max(map(te, pf)) + 300_000_000
win = [r for r in rows if ts(r) > p1]
s, c = int(sys.argv[2]), int(sys.argv[3])
prev = None
for r in win[s:s + c]:
    n = r["Kernel_Name"]
    n = re.sub(r"\(.*", "", n.replace("strata::kernels::", "").replace("strata::", "").replace("(anonymous namespace)::", "").replace("void ", ""))
    g = (ts(r) - prev) / 1e3 if prev else 0
    print("%7.1f us  gap %5.1f  grid %-12s blk %-5s %s" % ((te(r) - ts(r)) / 1e3, g, r.get("Grid_Size", r.get("Grid_Size_X", "")), r.get("Workgroup_Size", r.get("Workgroup_Size_X", "")), n[:90]))
    prev = te(r)
