#!/bin/bash
# srv.sh <config.json> <log>: stop any Strata server, start one on this config (GPU lock held), wait until it answers
cd ~/Strata
for p in $(pgrep -f "serve/serv[e]r.py"); do kill $p; done
for i in $(seq 60); do pgrep -f "^$HOME/Strata/engine/" >/dev/null || break; sleep 1; done
nohup flock ~/gpu.lock .venv/bin/python serve/server.py --engine strata --config "$1" --port 8080 > "$2" 2>&1 &
for i in $(seq 150); do curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null && { echo up; exit 0; }; sleep 2; done
echo "not up"; tail -5 "$2"
