# lbench.py <out.json> <target tokens,...> [rounds]: against the running server - for each target, a fresh prompt of
# that many tokens (a unique first line, then /tmp/corpus.txt) ending in a writing request; records the prompt rate and
# the writing rate of a 256-token answer, and the engine's own line (drafts accepted) from its log
import json, os, random, sys, time, urllib.request, re
os.chdir(os.path.expanduser("~/Strata"))
out, targets = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
corpus = open("/tmp/corpus.txt", errors="replace").read()
TASK = "\n\n---\nIgnore the text above. Write a long story about a lighthouse keeper who finds a message in a bottle."
def ask(text, n):
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": n, "reasoning_effort": "none",
            "temperature": 0}
    r = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
def last_engine_line():
    lines = [l for l in open("strata-890m-q2_0.log", errors="replace") if l.startswith("strata serve: prompt ")]
    return lines[-1].strip() if lines else ""
ratio = 3.3   # characters per token, corrected after every request
results = json.load(open(out)) if os.path.exists(out) else []
ask("Say hello.", 8)
for rd in range(rounds):
    for tgt in targets:
        for attempt in range(3):
            head = "Note %d.\n\n" % random.randint(0, 10**12)
            nchar = int((tgt - 60) * ratio)
            start = random.randint(0, max(0, len(corpus) - nchar - 1)) if rd > 0 else 0
            text = head + corpus[start:start + nchar] + TASK
            t0 = time.time()
            d = ask(text, 256)
            tm = d["timings"]
            ratio = nchar / max(1, tm["prompt_n"] - 60)
            if abs(tm["prompt_n"] - tgt) <= max(256, tgt * 0.03) or attempt == 2: break
            print("  resize: asked %d, got %d tokens (ratio now %.3f)" % (tgt, tm["prompt_n"], ratio), flush=True)
        line = last_engine_line()
        m = re.search(r"drafts accepted (\d+) of (\d+)", line)
        r = {"target": tgt, "round": rd, "prompt_n": tm["prompt_n"], "prompt_tps": tm["prompt_per_second"],
             "prompt_s": tm["prompt_ms"] / 1000, "gen_n": tm["predicted_n"], "gen_tps": tm["predicted_per_second"],
             "drafts": [int(m.group(1)), int(m.group(2))] if m else None, "wall_s": time.time() - t0, "engine": line}
        results.append(r)
        json.dump(results, open(out, "w"), indent=1)
        print("%6dK r%d  prompt %7d tok in %6.1f s = %5.1f tok/s | writing %3d tok at %5.2f tok/s | drafts %s" % (
            tgt // 1024, rd, r["prompt_n"], r["prompt_s"], r["prompt_tps"], r["gen_n"], r["gen_tps"], r["drafts"]), flush=True)
