# tools/rtx5080/mkcfg.py <base.json> <out.json> [--flag value ...]: a copy of a server config with those engine flags set (replaced
# when present, added otherwise); "--flag -" removes the flag and its value
import json, sys
c = json.load(open(sys.argv[1])); a = c["args"]; kv = sys.argv[3:]
for i in range(0, len(kv), 2):
    f, v = kv[i], kv[i + 1]
    if f in a:
        j = a.index(f)
        if v == "-": del a[j:j + 2]
        else: a[j + 1] = v
    elif v != "-": a += [f, v]
json.dump(c, open(sys.argv[2], "w"), indent=1)
