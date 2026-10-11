#!/bin/sh
# Strata on an RTX 5080 Laptop with little RAM (Qwen3.8-Flash-Next abliterated IQ3_S): the OpenAI-compatible server on
# 127.0.0.1:8080.  The experts the GPU does not hold are read from two SSDs as needed: the model's own, and a packed
# copy of most of each layer's experts on a second drive (tools/rtx5080/expert_pack.py, STRATA_EXPERT_MIRROR in the
# config).  RAM - the engine, its expert cache and the file cache - is capped at ${STRATA_RAM:-8G}, swap off; with more
# RAM raise the config's STRATA_STAGE_KEEP too (about 490 experts per extra GB).  STRATA_EXPERT_SKIP=0.1 writes faster
# by leaving out light experts that are not at hand (slightly different answers; off unless set).  --kv-grow lends
# the graphics memory the K/V does not use yet to the expert cache, and two conversation checkpoints (--prompt-cache 2)
# keep a long prompt inside the RAM cap.  The int8 K/V fits 163,840 tokens beside the prompt path's loan on a 16 GB
# card (--max-context).  STRATA_CONFIG picks another config.
cd "$(dirname "$0")/../.."
exec systemd-run --user --scope -p MemoryMax=${STRATA_RAM:-8G} -p MemorySwapMax=0 \
  .venv/bin/python serve/server.py --engine strata --config "${STRATA_CONFIG:-tools/rtx5080/strata-5080-lowram-iq3s.json}" --port 8080 "$@"
