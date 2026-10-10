# engine_once.py <tokens file> [extra engine args...]: the server config's engine, environment and arguments, run once
# on a pretokenized prompt (no server): for profiling.  STRATA_ENV_EXTRA="K=V K=V" adds or overrides settings.
import json, os, sys
os.chdir(os.path.expanduser("~/Strata"))
cfg = json.load(open("strata-890m-q2_0.json"))
env = dict(os.environ)
env.update(cfg.get("env", {}))
for kv in os.environ.get("STRATA_ENV_EXTRA", "").split():
    k, v = kv.split("=", 1); env[k] = v
env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
args = [a for a in cfg["args"]]
i = args.index("--max-context"); args[i + 1] = "32768"
exe = os.environ.get("ENGINE_EXE", cfg["exe"])
cmd = [exe] + args + ["--tokens-file", sys.argv[1], "--max-new", os.environ.get("MAX_NEW", "1")] + sys.argv[2:]
prefix = os.environ.get("PROFILER", "").split()
os.execvpe((prefix + cmd)[0], prefix + cmd, env)
