#!/bin/bash
# tools/rtx5080/build-rtx5080.sh <build dir> <engine name>: the CUDA engine for an RTX 50-series card (sm_120), from the
# repository root; the result is engine/<engine name>.  CUDA defaults to /usr/local/cuda; CUDAHOSTCXX / CUDAHOSTCC pick
# the host compiler nvcc runs (default g++ / gcc - set an older one when the system's is newer than the toolkit allows).
set -e
cd "$(dirname "$0")/../.."
B=$1 OUT=$2
CUDA=${CUDA:-/usr/local/cuda}
HOSTCXX=$(command -v "${CUDAHOSTCXX:-g++}") HOSTCC=$(command -v "${CUDAHOSTCC:-gcc}")
export PATH=$CUDA/bin:$PATH
.venv/bin/cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX" \
  -DCMAKE_C_COMPILER="$HOSTCC" -DCMAKE_CXX_COMPILER="$HOSTCXX" \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja"
.venv/bin/cmake --build "$B" --target strata
mkdir -p engine
cp "$B/strata" "engine/$OUT"
echo ENGINE_BUILD_OK
