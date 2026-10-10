#!/bin/bash
# needle_run.sh: under the GPU lock - start the server from run-890m.sh, run the needle test, stop the server
cd ~/Strata
nohup ./run-890m.sh > server-890m.out 2>&1 &
for i in $(seq 120); do sleep 2; curl -s -m 2 127.0.0.1:8080/v1/models | grep -q loaded && break; done
.venv/bin/python tools/needle_bench.py --lengths 8k,28k --depths 10,50,90 --timeout 600 2>&1 | tail -8
for p in $(pgrep -f "serve/serv[e]r.py"); do kill $p; done
for i in $(seq 40); do pgrep -f "engine/strat[a]" >/dev/null || break; sleep 1; done
