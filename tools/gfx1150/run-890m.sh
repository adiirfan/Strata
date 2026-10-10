#!/bin/sh
# Strata on the Radeon 890M (Qwen3.8-Flash-Next abliterated Q2_0): the OpenAI-compatible server on 127.0.0.1:8080.
# Needs the 48 GB graphics memory limit (ttm.pages_limit=12582912 on the kernel command line) for all 24,576 experts.
cd "$(dirname "$0")/../.."
exec .venv/bin/python serve/server.py --engine strata --config tools/gfx1150/strata-890m-q2_0.json --port 8080 "$@"
