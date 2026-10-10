# lp_serve.py <label> <engine exe> [K=V ...]: the server with these switches reads ~1,000 tokens of real text through its
# writing path (--short-read), writing every position's log-probabilities to /tmp/lp_<label>.tsv; then it stops
import json, os, signal, subprocess, sys, time, urllib.request, socket
os.chdir(os.path.expanduser("~/Strata"))
label, exe, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = json.load(open("strata-890m-q2_0.json"))
cfg["exe"] = os.path.abspath(exe)
cfg["env"].update(dict(p.split("=", 1) for p in pairs))
out = "/tmp/lp_%s.tsv" % label
if os.path.exists(out): os.remove(out)
cfg["env"]["STRATA_LOGPOS"] = out
cfg["env"]["STRATA_LOGPOS_TOPK"] = "20"
cfg["args"] += ["--short-read", "8192", "--adapt-every", "100000"]
json.dump(cfg, open("strata-890m-lp.json", "w"), indent=1)
def stop():
    for line in subprocess.run(["pgrep", "-f", "serve/serv[e]r.py"], capture_output=True, text=True).stdout.split():
        os.kill(int(line), signal.SIGTERM)
    for _ in range(60):
        if not subprocess.run(["pgrep", "-f", "engine/strat[a]"], capture_output=True).stdout: break
        time.sleep(1)
    for _ in range(60):
        with socket.socket() as so:
            if so.connect_ex(("127.0.0.1", 8080)) != 0: break
        time.sleep(1)
stop()
srv = subprocess.Popen([".venv/bin/python", "serve/server.py", "--engine", "strata", "--config", "strata-890m-lp.json", "--port", "8080"],
                       stdout=open("server-890m-lp.out", "w"), stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(240):
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2); break
    except Exception:
        if srv.poll() is not None: print(label, "SERVER DIED:", open("server-890m-lp.out").read()[-800:]); sys.exit(1)
        time.sleep(1)
text = open("docs/HOW_IT_WORKS.md").read()[:4200]
msgs = [{"role": "user", "content": "Read this:\n\n" + text}]
if os.environ.get("LP_MSGS"): msgs = json.load(open(os.environ["LP_MSGS"]))   # e.g. a question, the model's answer, "Thanks."
if os.environ.get("LP_GEN"):   # first: the answer to this question (greedy, 300 tokens) into /tmp/lp_gen.txt
    g = {"model": "x", "messages": [{"role": "user", "content": os.environ["LP_GEN"]}], "max_tokens": 300,
         "reasoning_effort": "none", "temperature": 0}
    rq = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(g).encode(), {"Content-Type": "application/json"})
    open("/tmp/lp_gen.txt", "w").write(json.load(urllib.request.urlopen(rq, timeout=900))["choices"][0]["message"]["content"])
body = {"model": "x", "messages": msgs, "max_tokens": 1, "reasoning_effort": "none", "temperature": 0}
r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=900))
print(label, d["timings"]["prompt_n"], "tokens read")
stop()
