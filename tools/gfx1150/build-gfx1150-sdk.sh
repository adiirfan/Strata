#!/bin/bash
# tools/gfx1150/build-gfx1150-sdk.sh <sdk dir> <build dir> <engine name>: the engine for the Radeon 890M (gfx1150)
# against an AMD TheRock toolkit (ROCm 10 measured), from the repository root; the result is engine/<engine name>
set -e
cd "$(dirname "$0")/../.."
SDK=$1 B=$2 OUT=$3
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
.venv/bin/cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1150 \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=-isystem $SDK/include --rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode" "-DCMAKE_CXX_FLAGS=-isystem $SDK/include" \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja
.venv/bin/cmake --build $B --target strata -j 12
cp $B/strata engine/$OUT
echo ENGINE_BUILD_OK
