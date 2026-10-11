# tools/rtx5080/eo5080.py <config> <tokens file> [K=V ...] [-- extra engine args]: the config's engine run once on a
# token file (MAX_NEW tokens, default 1; max context 32768), from the repository root; meant to be wrapped by eo5080.sh
import json, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
cfg = json.load(open(sys.argv[1])); tok = sys.argv[2]; rest = sys.argv[3:]
extra = rest[rest.index("--") + 1:] if "--" in rest else []
pairs = rest[:rest.index("--")] if "--" in rest else rest
env = dict(os.environ); env.update(cfg.get("env", {})); env.update(dict(p.split("=", 1) for p in pairs))
env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"])
args = list(cfg["args"])
if "--max-context" in args: args[args.index("--max-context") + 1] = "32768"
cmd = [cfg["exe"]] + args + ["--tokens-file", tok, "--max-new", os.environ.get("MAX_NEW", "1")] + extra
os.execvpe(cmd[0], cmd, env)
