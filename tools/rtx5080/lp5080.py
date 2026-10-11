# tools/rtx5080/lp5080.py <label> [K=V ...]: the low-RAM server (srv5080.sh, LP_CAP, default 8G) reads a conversation
# through its writing path (--short-read, adaptive tier off) and writes every position's log-probabilities to
# /tmp/lp_<label>.tsv; compare two such files with lp_cmp.py.  LP_CFG: the config (default
# tools/rtx5080/strata-5080-lowram-iq3s.json); LP_MSGS=<json>: the conversation (default /tmp/lp_ref_story.json);
# LP_GEN=1: write the reference conversations instead (/tmp/lp_ref_story.json, /tmp/lp_ref_code.json: the model's own
# 300-token answers to a story and a code request, with these settings)
import json, os, subprocess, sys, urllib.request
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRV = "tools/rtx5080/srv5080.sh"
label, pairs = sys.argv[1], sys.argv[2:]
cfg = json.load(open(os.environ.get("LP_CFG", "tools/rtx5080/strata-5080-lowram-iq3s.json")))
cfg.setdefault("env", {}).update(dict(p.split("=", 1) for p in pairs))
out = "/tmp/lp_%s.tsv" % label
if os.path.exists(out): os.remove(out)
if not os.environ.get("LP_GEN"):
    cfg["env"]["STRATA_LOGPOS"] = out
    cfg["env"]["STRATA_LOGPOS_TOPK"] = "20"
    cfg["args"] += ["--short-read", "8192"]
cfg["args"] += ["--adapt-every", "100000"]
run = "/tmp/strata-5080-lp.json"
json.dump(cfg, open(run, "w"), indent=1)
up = subprocess.run([SRV, "start", run, os.environ.get("LP_CAP", "8G")], capture_output=True, text=True).stdout.strip()
if not up.startswith("up"): print(label, "FAILED TO START", up[-400:]); sys.exit(1)
def ask(msgs, n):
    body = {"model": "x", "messages": msgs, "max_tokens": n, "reasoning_effort": "none", "temperature": 0}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))
P = {"story": "Write a long story about a lighthouse keeper who finds a message in a bottle.",
     "code": "Write a Python function that parses a CSV file and returns the rows as dictionaries, with error handling."}
try:
    if os.environ.get("LP_GEN"):
        for k, q in P.items():
            a = ask([{"role": "user", "content": q}], 300)["choices"][0]["message"]["content"]
            json.dump([{"role": "user", "content": q}, {"role": "assistant", "content": a}], open("/tmp/lp_ref_%s.json" % k, "w"))
            print(label, k, "reference:", len(a), "chars")
    else:
        msgs = json.load(open(os.environ.get("LP_MSGS", "/tmp/lp_ref_story.json")))
        d = ask(msgs, 1)
        print(label, d["timings"]["prompt_n"], "tokens read")
finally:
    subprocess.run([SRV, "stop"], capture_output=True)
