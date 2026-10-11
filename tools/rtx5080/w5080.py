# tools/rtx5080/w5080.py <label> <config.json> <MemoryMax|none> [K=V ...]: the server (srv5080.sh) in its own cgroup,
# a 6K-token summary prompt, a 300-token story and a 300-token code answer; prompt rate, both writing rates, the
# server's RAM at the end (cgroup: anon + page cache) and an md5 of each answer.  REPEAT=N writes story + code N times.
import hashlib, json, os, random, subprocess, sys, urllib.request
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRV = "tools/rtx5080/srv5080.sh"
label, conf, cap, pairs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
cfg = json.load(open(conf))
cfg.setdefault("env", {}).update(dict(p.split("=", 1) for p in pairs))
run = "/tmp/strata-5080-run.json"
json.dump(cfg, open(run, "w"), indent=1)
up = subprocess.run([SRV, "start", run, cap], capture_output=True, text=True).stdout.strip()
if not up.startswith("up"):
    print(label, "FAILED TO START:", up[-600:]); sys.exit(1)
def ask(msg, n):
    body = {"model": "x", "messages": [{"role": "user", "content": msg}], "max_tokens": n, "reasoning_effort": "none", "temperature": 0}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))
try:
    ask("Say hello.", 8)
    text = open("docs/DETAILS.md").read()[:18000]
    p = ask("Note %d. Summarise this in three sentences:\n\n" % random.randint(0, 10**9) + text, 60)["timings"]
    for rep in range(int(os.environ.get("REPEAT", "1"))):
        s = ask("Write a long story about a lighthouse keeper who finds a message in a bottle.", 300)
        c = ask("Write a Python class for an LRU cache with get and put, then explain how it works.", 300)
        if os.environ.get("REPEAT"):
            print("   pass %d: story %5.1f code %5.1f w/s" % (rep + 1, s["timings"]["predicted_per_second"], c["timings"]["predicted_per_second"]), flush=True)
    mem = subprocess.run([SRV, "mem"], capture_output=True, text=True).stdout.strip().replace("\n", " | ")
    print("%-12s prompt %5d tok %5.0f tok/s | story %5.1f | code %5.1f w/s | %s | out %s %s" % (
        label, p["prompt_n"], p["prompt_per_second"], s["timings"]["predicted_per_second"], c["timings"]["predicted_per_second"],
        mem, hashlib.md5(s["choices"][0]["message"]["content"].encode()).hexdigest()[:8], hashlib.md5(c["choices"][0]["message"]["content"].encode()).hexdigest()[:8]), flush=True)
except Exception as e:
    print(label, "FAILED:", e)
    if cfg.get("log") and os.path.exists(cfg["log"]): print(open(cfg["log"], errors="replace").read()[-800:])
subprocess.run([SRV, "stop"])
