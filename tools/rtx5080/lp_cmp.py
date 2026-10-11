# tools/rtx5080/lp_cmp.py <a.tsv> <b.tsv>: per-position comparison of two teacher-forced runs on the same tokens
import sys, math
def load(p):
    rows = {}
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        pos, tgt, lp, top, toplp = int(f[0]), int(f[1]), float(f[2]), int(f[3]), float(f[4])
        tk = {}
        for c in f[8:]:
            for kv in c.split():
                if ":" in kv:
                    i, v = kv.split(":"); tk[int(i)] = float(v)
        rows[pos] = (tgt, lp, top, tk)
    return rows
A, B = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(A) & set(B))
n = len(common); agree = 0; kl = 0.0; nlla = nllb = 0.0; t5 = 0
for p in common:
    ta, la, pa, ka = A[p]; tb, lb, pb, kb = B[p]
    assert ta == tb
    agree += pa == pb; nlla -= la; nllb -= lb
    # KL(A||B) over A's top 20 + rest bucket
    restA = 1 - sum(math.exp(v) for v in ka.values()); restB = 1 - sum(math.exp(kb[i]) for i in ka if i in kb)
    s = 0.0
    for i, v in ka.items():
        q = kb.get(i, math.log(max(1e-9, restB / 1000)))
        s += math.exp(v) * (v - q)
    if restA > 1e-9 and restB > 1e-9: s += restA * math.log(restA / restB)
    kl += s
    top5a = set(sorted(ka, key=lambda i: -ka[i])[:5]); top5b = set(sorted(kb, key=lambda i: -kb[i])[:5]); t5 += len(top5a & top5b)
print("positions %d | top-1 agreement %.1f%% | top-5 overlap %.1f%% | KL %.4f | perplexity %.3f vs %.3f" %
      (n, 100 * agree / n, 100 * t5 / (5 * n), kl / n, math.exp(nlla / n), math.exp(nllb / n)))
