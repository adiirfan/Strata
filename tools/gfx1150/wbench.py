# wbench.py <label> <config.json> [K=V ...]: the server on that config (+ env overrides): a 6K-token summary prompt, then
# a 300-token story and a 300-token code answer; prints the prompt rate and both writing rates
import json, os, signal, subprocess, sys, time, urllib.request, random, socket
os.chdir(os.path.expanduser("~/Strata"))
label, conf, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = json.load(open(conf))
cfg["env"].update(dict(p.split("=", 1) for p in pairs))
json.dump(cfg, open("strata-890m-wb.json", "w"), indent=1)
def stop():
    for line in subprocess.run(["pgrep", "-f", "serve/serv[e]r.py"], capture_output=True, text=True).stdout.split():
        os.kill(int(line), signal.SIGTERM)
    for _ in range(60):
        if not subprocess.run(["pgrep", "-x", "strata-r10[a-z0-9]*"], capture_output=True).stdout and \
           not subprocess.run(["pgrep", "-f", "^" + os.path.expanduser("~/Strata/engine/")], capture_output=True).stdout: break
        time.sleep(1)
    for _ in range(60):
        with socket.socket() as so:
            if so.connect_ex(("127.0.0.1", 8080)) != 0: break
        time.sleep(1)
stop()
srv = subprocess.Popen([".venv/bin/python", "serve/server.py", "--engine", "strata", "--config", "strata-890m-wb.json", "--port", "8080"],
                       stdout=open("server-890m-wb.out", "w"), stderr=subprocess.STDOUT, start_new_session=True)
def ask(msg, n):
    body = {"model": "x", "messages": [{"role": "user", "content": msg}], "max_tokens": n, "reasoning_effort": "none", "temperature": 0}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
for _ in range(240):
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2); break
    except Exception:
        if srv.poll() is not None: print(label, "SERVER DIED:", open("server-890m-wb.out").read()[-800:]); sys.exit(1)
        time.sleep(1)
try:
    ask("Say hello.", 8)
    text = open("docs/DETAILS.md").read()[:18000]
    p = ask("Note %d. Summarise this in three sentences:\n\n" % random.randint(0, 10**9) + text, 60)["timings"]
    s = ask("Write a long story about a lighthouse keeper who finds a message in a bottle.", 300)
    c = ask("Write a Python class for an LRU cache with get and put, then explain how it works.", 300)
    print("%-10s prompt %5d tok %4.0f tok/s | story %4.1f w/s | code %4.1f w/s | out %s %s" % (
        label, p["prompt_n"], p["prompt_per_second"], s["timings"]["predicted_per_second"], c["timings"]["predicted_per_second"],
        str(hash(s["choices"][0]["message"]["content"]) % 10**6), str(hash(c["choices"][0]["message"]["content"]) % 10**6)), flush=True)
except Exception as e:
    print(label, "FAILED:", e, open("server-890m-wb.out").read()[-600:])
stop()
