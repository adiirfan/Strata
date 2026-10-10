// src/prefill/moe_fused_iq.cu - see include/strata/prefill/moe_fused_iq.hpp (#136: the native packs' prompt experts on
// the fused int8 kernels of moe_fused.cu).
//
// The arithmetic.  An activation block of 32 values is x = d_x * a, a = round(x / d_x) in -127..127, stored in natural
// order (per 64 values: 64 codes, then {d_0, d_1} and 8 unused bytes - 80 bytes, the Q2_0 path's size).  A weight
// block of 32 (or 16) values is w = d_w * q with q an int8: the i-quants' codebook entries with their signs applied
// (IQ2_XXS / IQ2_XS / IQ2_S: 0..43, IQ3_XXS: 0..62, IQ3_S: 0..15, IQ4_XS / IQ4_NL: -127..113), Q2_0's code - 1.  So
//     sum w x = d_w * d_x * sum q a,     |sum q a| <= 32 * 127 * 127 < 2^22,
// and the int32 dot becomes a float exactly with one float add: the mma starts from C = 0x4B400000 (the bits of
// 1.5 * 2^23), so as_float(D) - 1.5 * 2^23 is the dot.  Per output and 32 values: that add, d_w * d_x, one fma (the
// formats with a scale per 16 values: two dots, each times its d_w, then d_x).  d_w is llama.cpp's own: IQ2_XXS
// d (2s + 1) / 8, IQ2_XS / IQ2_S d (2s + 1) / 8 per 16, IQ3_XXS d (2s + 1) / 4, IQ3_S d (2s + 1), IQ4_XS d (s - 32),
// IQ4_NL and Q2_0 d (dequantize_row_* in ggml-quants.c, mmq-load-tiles.cuh).
//
// The load stage.  Per 64 values of K (a stage), a thread decodes one 32-value sub-block of one of the work item's
// weight rows: its block bytes are read from the blob with 16-bit loads (the GGUF blocks are 2-byte aligned: 66 / 74 /
// 82 / 98 / 110 / 136 / 18 bytes) one stage ahead into registers (and two 256-value blocks ahead into L2), then turned
// into 32 int8 and the sub-block's scales in shared memory, double-buffered.  The codebooks live in shared memory (up
// to 8 KB, IQ2_S).  The activations arrive by cp.async, four stages deep, as in moe_fused.cu; the fragments come by
// ldmatrix.  The products are mma.sync m16n8k32 (m16n8k16 for the formats with a scale per 16 values); the epilogues
// (SwiGLU, H to int8 per 32 features; down into the per-slot rows) are moe_fused.cu's.
#include "strata/prefill/moe_fused_iq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::prefill::fused {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fused experts (native): %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// 16 warps, each 64 weight rows x 16 routed rows; WW of them along the weight rows and 16 / WW along the routed rows: a
// work item is (WR = 64 WW weight rows) x (TR = 256 / WW routed rows of one expert) - 256 x 64 or 128 x 128, chosen
// per launch by the rows per expert (pick_ww)
constexpr int THREADS = 512;
constexpr int WLD = 80;                 // bytes per decoded weight row of a stage: 64 int8 + 16 (no bank conflicts)
__host__ __device__ constexpr int tile_rows(int ww) { return 256 / ww; }
__host__ __device__ constexpr int weight_rows(int ww) { return 64 * ww; }
constexpr int ASTAGES = 4;              // cp.async depth of the activations (64 values of K a stage)
constexpr int GU_ROWS_K = 2560, D_ROWS_K = 640;   // K of gate/up (n_embd) and of down (n_ff)

// the formats (ggml type ids)
constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0,
              // Aurora (opt-in, HIP gfx11 only): Unsloth UD-Q4_K_XL's experts - gate/up Q4_K (one layer Q5_K), down Q5_1
              // (five layers Q8_0).  Their weights are w = d_w q + m_w (a minimum), so a sub-block's dot is
              //   sum w x = d_x (d_w sum q a + m_w sum a):
              // the activation blocks carry sum a per 32 values in their eight spare bytes (quant_act_nat_kernel and
              // the H epilogue write it for every path; no other format reads it)
              T_Q4_K = GGML_TYPE_Q4_K, T_Q5_K = GGML_TYPE_Q5_K, T_Q5_1 = GGML_TYPE_Q5_1, T_Q8_0 = GGML_TYPE_Q8_0;
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_iq4_nl) == 18 && sizeof(block_q2_0) == 18 && sizeof(block_q4_K) == 144 &&
              sizeof(block_q5_K) == 176 && sizeof(block_q5_1) == 24 && sizeof(block_q8_0) == 34,
              "the block layouts this file decodes");

// block bytes, scale per 16 values, codebook bytes in shared memory
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : t == T_Q4_K ? 144 : t == T_Q5_K ? 176 : t == T_Q5_1 ? 24 : t == T_Q8_0 ? 34 : 18;
}
// the formats with a minimum (w = d q + m), and the raw words a sub-block's load stage holds
__host__ __device__ constexpr bool has_min(int t) { return t == T_Q4_K || t == T_Q5_K || t == T_Q5_1; }
__host__ __device__ constexpr int raw_words(int t) {
    return t == T_Q4_K ? 10 : t == T_Q5_K ? 18 : t == T_Q5_1 ? 6 : t == T_Q8_0 ? 9 : 5;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : 0;
}
__host__ __device__ constexpr size_t smem_bytes(int t, int ww) {
    return (size_t) 2 * weight_rows(ww) * (WLD + 16) + (size_t) ASTAGES * tile_rows(ww) * AB +
           (size_t) tile_rows(ww) * 4 + grid_bytes(t);
}

// moe_fused.cu's grouping tables (the same layout: fused::group writes them)
struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    int2* tiles;
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

// ---- activations in natural order: one warp per 64 values
__global__ void quant_act_nat_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = xa + w * AB;
    const int c0 = a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0, c1 = a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0;
    out[lane] = (uint8_t) (int8_t) c0;
    out[32 + lane] = (uint8_t) (int8_t) c1;
    int s0 = c0, s1 = c1;   // the codes' sums per 32 values (the formats with a minimum read them)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, (float) s0, (float) s1);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = MAGIC + A (16 x 32 s8, row) * B (32 x 8 s8, col): the int32 dot with the magic bias already added (the C
// operand), so as_float(d) - 1.5 * 2^23 is the dot as a float
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
// the same at m16n8k16 (A 16 x 16, B 16 x 8)
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// four 8 x 16-byte matrices, row addresses from lanes 0-7, 8-15, 16-23, 24-31: lane (g, t) gets bytes 4t..4t+3 of row g
// of each - an mma fragment
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}

#endif
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800) || defined(__HIPCC__)
#if defined(__HIPCC__)
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
#endif
// ---- the load stage: a 32-value sub-block's bytes into registers, then int8 and scales
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
#if defined(__HIPCC__)
// gfx11 (Aurora): the signs from a shared-memory table instead of the compare / subtract emulation.  An entry (one per
// sign byte b; for the 7-bit sign formats per 7 bits, the parity bit folded in) is {m0, c0, m1, c1}: m = 0xFF in the
// bytes whose sign bit is set (bits 0-3 for .x, 4-7 for .y), c = m & 0x01010101.  q = (g ^ m) + c is -g where the sign
// is set - the two's complement, byte-wise - and g elsewhere; no byte carries into the next, because no codebook entry
// has a zero byte (256 - g with g in 1..127; the grids' bytes are 1..62), so it is the bytes signed8 gives.
__device__ __forceinline__ void signed8t(uint32_t gx, uint32_t gy, const uint4 sg, uint32_t& qx, uint32_t& qy) {
    qx = (gx ^ sg.x) + sg.y;
    qy = (gy ^ sg.z) + sg.w;
}
__host__ __device__ constexpr int sign_entries(int t) { return t == T_IQ2_S || t == T_IQ3_S ? 256 : 128; }
__device__ __forceinline__ uint4 sign_entry(int i, bool parity) {
    uint32_t b = (uint32_t) i;
    if (parity) b ^= (__popc(b) & 1) << 7;
    const uint32_t m0 = __vcmpne4(((b & 15) * 0x01010101u) & 0x08040201u, 0),
                   m1 = __vcmpne4(((b >> 4) * 0x01010101u) & 0x08040201u, 0);
    return make_uint4(m0, m0 & 0x01010101u, m1, m1 & 0x01010101u);
}
#endif
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in .x, the high ones' in .y
// W (X5): direct-selector form on gfx: p = low 3 bits of each nibble picks within the table half, bit 3 picks the half;
// exhaustively bit-exact over all 2^32 inputs (X5). Disable with -DSTRATA_W_NO_T16.
#if defined(__HIPCC__) && !defined(STRATA_W_NO_T16)
__device__ __forceinline__ uint32_t table16_one(uint32_t x, const uint32_t (&t)[4]) {
    const uint32_t p = x & 0x07070707u;
    const uint32_t a = __builtin_amdgcn_perm(t[1], t[0], p), b = __builtin_amdgcn_perm(t[3], t[2], p);
    return __builtin_amdgcn_perm(b, a, ((x >> 1) & 0x04040404u) | 0x03020100u);
}
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    lo = table16_one(q4, t);
    hi = table16_one(q4 >> 4, t);
}
#else
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}
#endif

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; Q2_0: the 64-value block,
// ib 0..1; IQ4_NL: the 32-value block).
// 32 bits at a 2-byte aligned address (one load when it is 4-byte aligned)
__device__ __forceinline__ uint32_t ld32a(const uint8_t* p) {
    return ((uintptr_t) p & 3) == 0 ? *(const uint32_t*) p : ld32(p);
}
template <int T, int NW> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[NW]) {
    static_assert(NW >= raw_words(T), "raw words");
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else if constexpr (T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else if constexpr (T == T_Q4_K || T == T_Q5_K) {
        // qs: 32 bytes of the sub-block pair (ib / 2), the nibble chosen in convert; w[8] = d | dmin << 16; w[9]: the
        // scale/min bytes (get_scale_min_k4's: ib < 4: sc[ib], sc[ib + 4]; else sc[ib + 4], sc[ib - 4], sc[ib]), ib << 24
        constexpr int QS = T == T_Q4_K ? 16 : 48;
        const uint8_t* q = bp + QS + 32 * (ib >> 1);
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32a(q + 4 * k);
        w[8] = ld32a(bp);
        const uint8_t* sc = bp + 4;
        w[9] = ib < 4 ? (uint32_t) sc[ib] | ((uint32_t) sc[ib + 4] << 8)
                      : (uint32_t) sc[ib + 4] | ((uint32_t) sc[ib - 4] << 8) | ((uint32_t) sc[ib] << 16);
        w[9] |= (uint32_t) ib << 24;
        if constexpr (T == T_Q5_K) {
#pragma unroll
            for (int k = 0; k < 8; ++k) w[10 + k] = ld32a(bp + 16 + 4 * k);   // qh
        }
    } else if constexpr (T == T_Q5_1) {   // d, m, qh, qs[16]
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32a(bp + 8 + 4 * k);
        w[4] = ld32a(bp + 4);
        w[5] = ld32a(bp);
    } else if constexpr (T == T_Q8_0) {   // d, qs[32]
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[8] = ld16(bp);
    } else {   // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T, int NW>
__device__ __forceinline__ void convert(const uint32_t (&w)[NW], const uint8_t* grid, const uint32_t (&kv)[4],
                                        uint32_t (&q)[8], float& s0, float& s1, const uint4* sgn = nullptr) {
    if constexpr (T == T_Q4_K || T == T_Q5_K) {
        const int ib = (int) (w[9] >> 24), sh = 4 * (ib & 1);
        uint32_t sc, mn;
        if (ib < 4) { sc = w[9] & 63; mn = (w[9] >> 8) & 63; }
        else {
            sc = ((w[9] & 0xF) | (((w[9] >> 8) & 0xC0) >> 2));   // sc[ib + 4] & 15 | (sc[ib - 4] >> 6) << 4
            mn = (((w[9]) >> 4) & 0xF) | (((w[9] >> 16) & 0xC0) >> 2);   // sc[ib + 4] >> 4 | (sc[ib] >> 6) << 4
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q[k] = (w[k] >> sh) & 0x0F0F0F0Fu;
            if constexpr (T == T_Q5_K) q[k] |= ((w[10 + k] >> ib) & 0x01010101u) << 4;
        }
        s0 = half_at(w[8]) * (float) sc;
        s1 = -(half_at(w[8] >> 16) * (float) mn);
    } else if constexpr (T == T_Q5_1) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t lo = ((w[4] >> (4 * k)) & 0xF) * 0x00204081u, hi = ((w[4] >> (16 + 4 * k)) & 0xF) * 0x00204081u;
            q[k] = (w[k] & 0x0F0F0F0Fu) | ((lo & 0x01010101u) << 4);
            q[4 + k] = ((w[k] >> 4) & 0x0F0F0F0Fu) | ((hi & 0x01010101u) << 4);
        }
        s0 = half_at(w[5]);
        s1 = half_at(w[5] >> 16);
    } else if constexpr (T == T_Q8_0) {
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = w[k];
        s0 = s1 = half_at(w[8]);
    } else if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(c >> 9) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
#if defined(__HIPCC__)
            signed8t(e.x, e.y, sgn[(w[1] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0], g[i1], sgn[(w[2] >> (7 * l)) & 127], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
#if defined(__HIPCC__)
            signed8t(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                     sgn[(w[2] >> (8 * l)) & 255], q[2 * l], q[2 * l + 1]);
#else
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
#endif
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    } else {   // Q2_0: code - 1 via a byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        s0 = s1 = half_at(w[2]);
    }
}

template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}
#endif

// One work item = (TR routed rows of expert e - one or two of group()'s 64-row tiles -, a block of WR weight rows);
// a persistent grid walks the batch's items.  GU: gate/up rows of WR / 2 features (format WT) against the rows'
// tokens' activations, SwiGLU, H to int8 per 32 features into `out`.  !GU: down rows (format WT) against H, FP32 into
// `dm`.  Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, gate rows as mma rows 0-7 and
// up rows as 8-15, so a lane holds gate and up of one feature), routed rows 16 wt.. as two n8.
template <int WT, bool GU, int WW>
__global__ void __launch_bounds__(THREADS, 1)
native_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
              const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int TR = tile_rows(WW), WR = weight_rows(WW);
    constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;   // 64-value stages along K
    constexpr int NFB = (GU ? 1280 : 2560) / WR;          // weight-row blocks per tile
    constexpr int ACT_LD = NS * AB;                       // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;                                               // [2][WR][WLD] decoded int8 weights
    float* ws = (float*) (smem + 2 * WT_BYTES);                       // [2][WR][4] their scales per 16 values
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);                // [ASTAGES][TR][AB] activations
    int* srow = (int*) (stages + ASTAGES * ACT_STAGE);                // [TR] the activation row of each tile row
    uint8_t* sgrid = (uint8_t*) (srow + TR);                          // the codebook

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    // Local weight rows: m tile i of warp wf is rows 64 wf + 16 i.. (mma rows g and g + 8: rows +g and +8+g).  For
    // gate/up a tile is 8 features, its gate rows then its up rows, so a lane holds gate and up of one feature.  The
    // fragments come by ldmatrix: lane l gives the address of row (l & 7) of matrix l >> 3 - A: {rows 0-7, rows 8-15}
    // x {bytes 0-15, 16-31} of a 32-value chunk; B: {tokens nb0 + 0-7, + 8-15} x {bytes 0-15, 16-31}.
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    // this thread's decode unit (the first 2 WR threads): local weight row ur, sub-block uj of each stage
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        if ((row0 - tb.off[e]) % TR != 0) continue;              // a 64-row tile inside an earlier item's TR rows
        const int nrows = min(TR, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WR;
        // local row ur: gate/up - feature (WR / 2) fb + 8 (ur >> 4) + (ur & 7), its up row when bit 3 is set
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* {                  // the block of stage s's sub-block
            if (GU) return wrow + (s >> 2) * BS;
            return WT == T_IQ4_NL ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return GU ? 2 * (s & 3) + uj : uj; };
        // the weight bytes into L2 ahead of the register loads (one stage ahead only - less than a DRAM round trip):
        // gate/up PF super-blocks ahead (the row's two threads take a block's first and last line), down the whole row
        // slice at once (180 or 360 bytes a row)
        constexpr int PF = 2;
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < GU_ROWS_K / 256) pf_l2(wrow + sb * BS + (uj ? BS - 1 : 0));
        };
        if (GU) {
#pragma unroll
            for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        } else if (dec) {
            for (size_t o = 128 * (size_t) uj; o < geo.d_row + 127; o += 256) pf_l2(wrow + min(o, geo.d_row - 1));
        }
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();                                          // srow
        // the rows past the item's end are not loaded: a warp entirely past it skips the products, and the columns
        // of a partial one are never written (no reduction mixes columns)
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % ASTAGES) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += THREADS) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[5], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1);
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
        };
#pragma unroll
        for (int s = 0; s < ASTAGES - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[5] = {0, 0, 0, 0, 0};
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<ASTAGES - 2>();
            __syncthreads();
            if (s + ASTAGES - 1 < NS) load_act(s + ASTAGES - 1);
            cp_commit();
            if (GU && (s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % ASTAGES) * ACT_STAGE;
                // the stage's scales: per weight row 4 (per 16 values), per routed row 2 (per 32)
                float2 dx[2][2];                                  // [n8 tile][column 2 tig + cc]
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
#pragma unroll
                for (int h = 0; h < 2; ++h) {                     // the stage's two 32-value halves
                    uint32_t bq[4];                               // {n0 bytes 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];                            // {rows g, g + 8} x {bytes 0-15, 16-31}
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                // a scale per 16 weights: the two halves' dots, each times its scale, then d_x
                                int d0[4], d1[4];
                                mma16(d0, a[0], a[1], bq[2 * n]);
                                mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, a, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? wa0 : wb0) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[q]), acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            // the next stage's weights (into the other buffer, read after the next barrier).  (Decoding them before
            // this stage's products instead was slower on the RTX 5070: a layer of 2048 / 3584 / 8192 tokens, IQ2_S
            // 5.2 / 9.5 / 17.5 ms -> 6.0 / 10.2 / 18.3.)
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature (WR / 2) fb + 32 wf + 8 i + g (lane g of m tile i: rows 64 wf + 16 i + g and + 8), tile
            // rows nb0 + 8 n + 2 tig + c; H to int8 per row over the warp's 32 features (one 32-value half-block of the
            // down product's K)
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * 2560 + rbase + 64 * wf + 16 * i + 8 * (q >> 1) + g] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

#if defined(__HIPCC__)
// ---- Aurora (S23): the native packs' fused experts on gfx11 (RDNA3 / RDNA3.5) matrix cores, v_wmma_i32_16x16x16_iu8
// (wave32).  The CUDA kernel's arithmetic: a 32-value sub-block decoded to int8 (load_unit / convert above) and its
// scales; per 32 values the int32 dot starts from MAGIC (the C operand), so as_float(d) - 1.5 * 2^23 is the dot; the
// formats with a scale per 16 values take each 16-value k-step's dot alone.  Fragments (gfx11): A lane l holds the 16
// k of row l % 16 (lanes 16..31 repeat lanes 0..15), B lane l the 16 k of column l % 16, C/D lane l holds
// D[2i + l / 16][l % 16].  Both operands are in natural order: the decoded weights (LDS, double-buffered) and the
// activations (quant_act_nat_kernel; read by each lane from global/L2, a stage ahead).
// A work item = NW_ROWS weight rows x a 64-row tile; 8 waves: 4 along the weight rows (32 each) x 2 along the tile
// (32 each), each 2 x 2 WMMA tiles.  Gate/up: local row r is feature r / 2, its gate (r even) or up (r odd) row, so a
// lane pair l, l + 16 holds gate and up of one feature.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_NAT_W11 1
#else
#define STRATA_NAT_W11 0
#endif
// The VGPR cap of the occupancy variant (amdgpu_waves_per_eu).  8 was measured against ROCm 7's clang; clang 22 (ROCm 7.10)
// computes wrong results with it (#1180: gfx1100, every run), so there the cap is off.  -DSTRATA_W_LB=N sets it.
#if defined(STRATA_W_LB)
constexpr int NW_LB = STRATA_W_LB;
#elif defined(__clang_major__) && __clang_major__ >= 22
constexpr int NW_LB = 1;
#else
constexpr int NW_LB = 8;
#endif
typedef int nw_i4 __attribute__((ext_vector_type(4)));
typedef int nw_i8 __attribute__((ext_vector_type(8)));
constexpr int NW_ROWS = 128;
constexpr int NW_THREADS = 256;

__device__ __forceinline__ nw_i8 nw_wmma(nw_i4 a, nw_i4 b, nw_i8 c) {
#if STRATA_NAT_W11
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
    __builtin_trap();
    return c;
#endif
}
__device__ __forceinline__ nw_i4 nw_u4(uint4 v) { return nw_i4{(int) v.x, (int) v.y, (int) v.z, (int) v.w}; }
// A workgroup barrier for LDS traffic only: s_waitcnt lgkmcnt(0) + s_barrier.  __syncthreads() is a workgroup-scope
// fence on every address space, which on gfx11 also waits for the outstanding global loads (vmcnt(0)) - the activation
// prefetch would have to land within one stage - and invalidates the L0 cache (buffer_gl0_inv) at every barrier.  The
// kernel shares nothing through global memory inside a launch, so the LDS-only fence is all it needs.
// the value of the lane 16 away (the other half of the wave): v_permlanex16 with the identity selectors, one VALU op
// (__shfl_xor by 16 is a ds_bpermute, an LDS op)
__device__ __forceinline__ float nw_swap16(float v) {
    return __uint_as_float((unsigned) __builtin_amdgcn_permlanex16((int) __float_as_uint(v), (int) __float_as_uint(v),
                                                                   0x76543210, 0xFEDCBA98, false, false));
}
__device__ __forceinline__ void nw_bar() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// W (X1): ALIAS = the H tile lives on the weight buffers (one extra barrier); LB > 0 = amdgpu_waves_per_eu(LB) (VGPR cap).
// Disable both with -DSTRATA_W_NO_OCC (the launch sites then use <WT, GU, false, 0> and grid factor wgp_blocks).
template <int WT, bool GU, bool ALIAS = false, int LB = 0>
__global__ void __launch_bounds__(NW_THREADS) __attribute__((amdgpu_waves_per_eu(LB > 0 ? LB : 1)))
native_w11_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* act,   // act, src: no __restrict__ (see fetch_b)
                  const int32_t* src, uint8_t* __restrict__ out, float* __restrict__ dm, int dm16 = 0) {
#if STRATA_NAT_W11
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;  // 64-value stages along K
    constexpr int NFB = (GU ? 1280 : 2560) / NW_ROWS;     // work items per token tile
    constexpr int ACT_LD = NS * AB;
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    constexpr int GB = grid_bytes(WT) > 0 ? grid_bytes(WT) : 16;
    constexpr bool Q2 = WT == T_Q2_0;
    // W (X3).  The token tile: TT routed rows of one expert per work item (two of the grouping's 64-row tiles for
    // Q2_0: the weights of the item are decoded, and read from memory, once per 128 tokens instead of per 64; the
    // formats with a codebook keep 64 so that three blocks still fit a WGP's LDS).  Items are enumerated per expert,
    // ceil(rows / TT) of them, from a prefix sum over the launch's experts that each block builds in LDS.  Of the
    // eight waves, four run along the weight rows (32 each) and two along the tokens (TT / 2 each, NTT 16-row tiles).
    constexpr int TT = Q2 ? 128 : kTileRows;
    constexpr int NTT = TT / 32;
    constexpr int NP = (TT * (AB / 16) + NW_THREADS - 1) / NW_THREADS;   // activation pieces (16 bytes) a thread stages
    constexpr int PIECES = TT * (AB / 16);
    static_assert((PIECES - (NP - 1) * NW_THREADS & (PIECES - (NP - 1) * NW_THREADS - 1)) == 0, "the last pieces' count is a power of two (their duplicates' mask)");
    // W (X2).  Q2_0's weight scale is one per 64 values - one per row and stage - and lives where its D lanes read it:
    // ws2[buf][row / 16][hi][i] is the scale of row 16 (row / 16) + 2 i + hi, so lane (hi, l16) of a tile's WMMA reads
    // its eight scales with two 16-byte loads (the other formats: ws[buf][row][4], per 16 values).
    constexpr int WS_BYTES = Q2 ? NW_ROWS * 4 : NW_ROWS * 16;
    constexpr int WT_BYTES = 2 * NW_ROWS * WLD, BT_BYTES = 2 * TT * AB;
    // One arena: the decoded weights (double-buffered), the staged activations (double-buffered: a stage's TT rows x
    // 80 bytes, in the global layout), the weight scales.  The activations come through LDS so that the four waves
    // along the weight rows share one fetch, the fetch is two stages ahead with NP x 4 registers per thread in flight
    // (not 36 per lane), and the fragments arrive by 16-byte LDS loads like the weights'.
    __shared__ __align__(16) uint8_t arena[WT_BYTES + BT_BYTES + 2 * WS_BYTES];
    uint8_t(*wt)[NW_ROWS][WLD] = reinterpret_cast<uint8_t(*)[NW_ROWS][WLD]>(arena);
    uint8_t(*bt)[TT][AB] = reinterpret_cast<uint8_t(*)[TT][AB]>(arena + WT_BYTES);
    float(*ws)[NW_ROWS][4] = reinterpret_cast<float(*)[NW_ROWS][4]>(arena + WT_BYTES + BT_BYTES);
    float(*ws2)[NW_ROWS / 16][2][8] = reinterpret_cast<float(*)[NW_ROWS / 16][2][8]>(arena + WT_BYTES + BT_BYTES);
    __shared__ __align__(16) uint8_t sgrid[GB];
    constexpr int SGN = (WT == T_IQ2_XXS || WT == T_IQ2_XS || WT == T_IQ2_S || WT == T_IQ3_XXS || WT == T_IQ3_S)
                            ? sign_entries(WT) : 1;
    __shared__ __align__(16) uint4 ssign[SGN];                   // the signs of a sign byte, as byte masks
    __shared__ int spre[kMaxBatch + 1];                          // items before expert e0 + i (prefix sum), spre[ne] = all
    // the H tile of a gate/up item, on the arena once the stages are done with it: hs[token][HS_LD] (a lane's 16
    // features are four 16-byte stores; 68 floats a row: 16-byte aligned, and 16 rows at a column spread over the
    // banks), hm[wm][token] = max |h| over wave wm's 16 features
    constexpr int HS_LD = 68;
    static_assert((size_t) TT * HS_LD * 4 + 4 * TT * 4 <= (size_t) WT_BYTES + BT_BYTES, "H tile must fit the arena");
    float(*hs)[HS_LD] = reinterpret_cast<float(*)[HS_LD]>(arena);
    float(*hm)[TT] = reinterpret_cast<float(*)[TT]>(arena + TT * HS_LD * 4);
    // the wave index is uniform (readfirstlane: the compiler does not know the wave size here), so the per-wave
    // tile checks are scalar branches and the per-wave parts of the addresses live in scalar registers
    const int tid = threadIdx.x, lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += NW_THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    if constexpr (SGN > 1) {
        for (int i = tid; i < SGN; i += NW_THREADS) ssign[i] = sign_entry(i, SGN == 128);
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    // the items of the launch's experts: ceil(rows / TT) each (every thread sums the experts before its own: at most
    // 128 LDS reads, once per block)
    const int ne = b.e1 - b.e0;
    if (tid <= ne) {
        int s = 0;
        for (int i = 0; i < tid; ++i) s += (tb.off[b.e0 + i + 1] - tb.off[b.e0 + i] + TT - 1) / TT;
        spre[tid] = s;
    }
    nw_bar();
    const int nwork = spre[ne] * NFB;
    if ((int) blockIdx.x >= nwork) return;
    const int ur = tid >> 1, uj = tid & 1;                       // this thread's decode unit: row ur, sub-block uj
    // this thread's activation pieces: pieces tid, tid + 256, ... (the last one duplicated for the threads past the
    // end: the loads are unconditional, only the stores are not)
    int bp[3] = {0, 0, 0}, btk[3] = {0, 0, 0};
    bool live[3] = {false, false, false};
#pragma unroll
    for (int q = 0; q < NP; ++q) {
        const int first = q * NW_THREADS;
        live[q] = first + tid < PIECES;
        bp[q] = first + (live[q] ? tid : (tid & (PIECES - first - 1)));
        btk[q] = bp[q] / 5;
    }
    // A work item: its tile and rows, this thread's weight row and the sources of its activation pieces (rows past
    // the tile's end repeat its last one).  The pipeline runs across items: the last two stages of an item fetch the
    // next item's first two stages (see `stage`), and the item's description - the expert (a search of spre, LDS),
    // its row range (two loads), (gate/up) the routing of this thread's rows (loads) - runs one level per item ahead
    // of that, so no load of it is waited for (a wait for a late load drains every load in flight: the counter is in
    // order).
    struct Item {
        int fb, row0, nrows;
        const uint8_t* wrow;
        uint32_t boff0, boff1, boff2;   // this thread's pieces: byte offsets into act (32-bit: fewer registers)
    };
    // uniform values that arrive in vector registers (LDS / vector loads) go to scalar ones
    auto uni = [](int v) { return __builtin_amdgcn_readfirstlane(v); };
    const int grid = (int) gridDim.x;
    auto item_of = [&](int k) { return min((int) blockIdx.x + k * grid, nwork - 1); };   // the k-th item after the first
    auto expert_of = [&](int w, int& ei, int& j) {   // the item's expert (index in the launch) and its tile in it
        const int p = w / NFB;
        int lo = 0;
#pragma unroll
        for (int step = kMaxBatch; step > 0; step >>= 1)
            if (lo + step < ne && spre[lo + step] <= p) lo += step;
        ei = uni(lo);
        j = uni(p - spre[lo]);
    };
    auto range_of = [&](int ei, int& oa, int& ob) {
        oa = uni(tb.off[b.e0 + ei]);
        ob = uni(tb.off[b.e0 + ei + 1]);
    };
    auto rows_of = [&](int j, int oa, int ob, int (&sr)[NP]) {   // this thread's activation rows
        const int row0 = oa + j * TT, nrows = min(TT, ob - row0);
#pragma unroll
        for (int q = 0; q < NP; ++q) {
            const int r = row0 + min(btk[q], nrows - 1);
            sr[q] = GU ? src[r] : r;
        }
    };
    auto finish = [&](int w, int ei, int j, int oa, int ob, const int (&sr)[NP]) -> Item {
        Item it;
        it.fb = uni(w % NFB);
        it.row0 = uni(oa + j * TT);
        it.nrows = uni(min(TT, ob - it.row0));
        const uint8_t* blob = b.blob[ei];
        it.wrow = GU ? blob + ((ur & 1) ? geo.up_off : 0) + (size_t) (it.fb * (NW_ROWS / 2) + (ur >> 1)) * geo.gu_row
                     : blob + geo.down_off + (size_t) (it.fb * NW_ROWS + ur) * geo.d_row;
        it.boff0 = (uint32_t) sr[0] * ACT_LD + (bp[0] % 5) * 16;
        it.boff1 = (uint32_t) sr[1] * ACT_LD + (bp[1] % 5) * 16;
        it.boff2 = NP > 2 ? (uint32_t) sr[NP - 1] * ACT_LD + (bp[NP - 1] % 5) * 16 : 0u;
        return it;
    };
    // gate/up: a 256-value super-block covers four stages (sub-blocks 0..7), except Q2_0, whose 64-value block is
    // one stage (sub-blocks 0..1), as in its down rows
    auto unit = [&](const Item& it, int s) -> const uint8_t* {
        if (GU) return WT == T_Q2_0 ? it.wrow + s * BS : it.wrow + (s >> 2) * BS;
        return (WT == T_IQ4_NL || WT == T_Q5_1 || WT == T_Q8_0) ? it.wrow + (2 * s + uj) * BS : it.wrow + s * BS;
    };
    auto sub = [&](int s) { return GU && WT != T_Q2_0 ? 2 * (s & 3) + uj : uj; };
    auto put = [&](const uint32_t (&raw)[raw_words(WT)], int buf) {
        uint32_t q[8];
        float s0, s1;
        convert<WT>(raw, sgrid, kv, q, s0, s1, ssign);
        uint4* d = (uint4*) &wt[buf][ur][32 * uj];
        d[0] = make_uint4(q[0], q[1], q[2], q[3]);
        d[1] = make_uint4(q[4], q[5], q[6], q[7]);
        if constexpr (Q2) {
            if (uj == 0) ws2[buf][ur >> 4][ur & 1][(ur & 15) >> 1] = s0;
        } else {
            *(float2*) &ws[buf][ur][2 * uj] = make_float2(s0, s1);
        }
    };
    // The activation pieces.  The loads are unconditional, because a load inside a branch, or a register move of a
    // load's destination, makes the compiler wait for every load in flight (s_waitcnt vmcnt(0)) - the prefetch
    // distance would be gone.  For the same reason the stage loop is unrolled by two, the two register sets
    // alternating, with the last two stages peeled; and `act` / `src` are not __restrict__: an invariant load may be
    // sunk to its use (two stages later) past any compiler barrier.
    // (named registers, not arrays: an array of pieces handed around by reference ended up in scratch memory)
    auto fetch_b = [&](const Item& it, int s, uint4& p0, uint4& p1, uint4& p2) {
        p0 = *reinterpret_cast<const uint4*>(act + it.boff0 + s * AB);
        p1 = *reinterpret_cast<const uint4*>(act + it.boff1 + s * AB);
        if constexpr (NP > 2) p2 = *reinterpret_cast<const uint4*>(act + it.boff2 + s * AB);
    };
    auto store_b = [&](int buf, const uint4& p0, const uint4& p1, const uint4& p2) {
        *reinterpret_cast<uint4*>(&bt[buf][0][0] + 16 * bp[0]) = p0;
        if (live[1]) *reinterpret_cast<uint4*>(&bt[buf][0][0] + 16 * bp[1]) = p1;
        if constexpr (NP > 2) if (live[2]) *reinterpret_cast<uint4*>(&bt[buf][0][0] + 16 * bp[2]) = p2;
    };
    // the description pipeline's state at an item's top: the expert / tile of items +1, +2, +3, the row range of +1,
    // +2, the rows of +1 (the first item's own description is resolved here, once per block)
    int ei1, j1, ei2, j2, ei3, j3, oa1, ob1, oa2, ob2;
    int sr1[NP];
    expert_of(item_of(1), ei1, j1);
    expert_of(item_of(2), ei2, j2);
    expert_of(item_of(3), ei3, j3);
    range_of(ei1, oa1, ob1);
    range_of(ei2, oa2, ob2);
    rows_of(j1, oa1, ob1, sr1);
    Item cur;
    {
        int ei0, j0, oa0, ob0, sr0[NP];
        expert_of(item_of(0), ei0, j0);
        range_of(ei0, oa0, ob0);
        rows_of(j0, oa0, ob0, sr0);
        cur = finish(item_of(0), ei0, j0, oa0, ob0, sr0);
    }
    // the WMMA bias (the C operand; see MAGIC) in eight registers for the whole kernel: at the register limit the
    // compiler re-materializes it before most chains instead (eight moves each) - an opaque asm pins it
    nw_i8 m;
#pragma unroll
    for (int i = 0; i < 8; ++i) asm volatile("v_mov_b32 %0, 0x4b400000" : "=v"(m[i]));
    // in flight across the stages: raw0 = the next item's stage-0 weights, raw = the stage after the next one's;
    // ra / rc = two sets of activation pieces (stage s + 1 and s + 2)
    uint32_t raw[raw_words(WT)] = {}, raw0[raw_words(WT)] = {};
    uint4 ra0, ra1, ra2, rc0, rc1, rc2;
    load_unit<WT>(unit(cur, 0), sub(0), raw0);
    fetch_b(cur, 0, ra0, ra1, ra2);
    fetch_b(cur, 1, rc0, rc1, rc2);
    load_unit<WT>(unit(cur, 1), sub(1), raw);
    for (int k = 0, w = blockIdx.x; w < nwork; ++k, w += grid) {
        // the description pipeline: the expert of item +4, the row range of +3, the rows of +2; item +1 is complete
        int ei4, j4, oa3, ob3, sr2[NP];
        expert_of(item_of(k + 4), ei4, j4);
        range_of(ei3, oa3, ob3);
        rows_of(j2, oa2, ob2, sr2);
        asm volatile("" ::: "memory");                           // these loads stay here, not at their use
        const Item nxt = finish(item_of(k + 1), ei1, j1, oa1, ob1, sr1);
        const int fb = cur.fb, row0 = cur.row0, nrows = cur.nrows, rbase = fb * NW_ROWS;
        nw_bar();                                                 // the previous item is done with the arena
        put(raw0, 0);
        store_b(0, ra0, ra1, ra2);
        bool on_nt[NTT];                                          // this wave's 16-row token tiles with rows
#pragma unroll
        for (int nt = 0; nt < NTT; ++nt) on_nt[nt] = (TT / 2) * wn + 16 * nt < nrows;
        const bool on = on_nt[0];
        float acc[2][NTT][8];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < NTT; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] = 0.0f;
        // Stage s: its weights and activations are in buffers s & 1.  It fetches two stages ahead into nb (mode 0:
        // stage s + 2; 1: the next item's stage 0; 2: its stage 1), computes, and decodes / stores stage s + 1 (sa)
        // into the other buffers, whose readers (stage s - 1) passed this stage's barrier.
        auto stage = [&](int s, uint4& sa0, uint4& sa1, uint4& sa2, uint4& nb0, uint4& nb1, uint4& nb2, int mode) {
            nw_bar();
            if (mode == 0) fetch_b(cur, s + 2, nb0, nb1, nb2);
            else fetch_b(nxt, mode - 1, nb0, nb1, nb2);
            asm volatile("" ::: "memory");                       // keep the loads here (the compiler sinks them to their use otherwise)
            if (on) {
                const int bf = s & 1;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    const int rb = 32 * wm + 16 * mt;
                    nw_i4 A[2][2];                                // [h][k16]
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint4* ap = reinterpret_cast<const uint4*>(&wt[bf][rb + l16][32 * h]);
                        A[h][0] = nw_u4(ap[0]);
                        A[h][1] = nw_u4(ap[1]);
                    }
                    float w0[2][8], w1[2][8];                     // [h][i]: the scales of rows rb + 2 i + hi
                    if constexpr (Q2) {
                        const float4* sp = reinterpret_cast<const float4*>(&ws2[bf][rb / 16][hi][0]);
                        const float4 sa4 = sp[0], sb4 = sp[1];
                        w0[0][0] = sa4.x; w0[0][1] = sa4.y; w0[0][2] = sa4.z; w0[0][3] = sa4.w;
                        w0[0][4] = sb4.x; w0[0][5] = sb4.y; w0[0][6] = sb4.z; w0[0][7] = sb4.w;
                    } else {
#pragma unroll
                        for (int h = 0; h < 2; ++h)
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const float2 sw = *reinterpret_cast<const float2*>(&ws[bf][rb + 2 * i + hi][2 * h]);
                                w0[h][i] = sw.x; w1[h][i] = sw.y;
                            }
                    }
                    // the token tiles' operands are read one tile at a time (all of them at once would hold NTT x
                    // (4 x 4 + 2) registers across the WMMAs)
#pragma unroll
                    for (int nt = 0; nt < NTT; ++nt) {
                        if (nt > 0 && !on_nt[nt]) continue;
                        const uint8_t* br = &bt[bf][(TT / 2) * wn + 16 * nt + l16][0];
                        const float2 bx = *reinterpret_cast<const float2*>(br + 64);
                        float2 bsum = make_float2(0.0f, 0.0f);
                        if constexpr (has_min(WT)) bsum = *reinterpret_cast<const float2*>(br + 72);
                        if constexpr (Q2) {
                            // one weight scale for both halves: acc += w (dx0 dot0 + dx1 dot1)
                            const uint4* bp4 = reinterpret_cast<const uint4*>(br);
                            const nw_i8 d0 = nw_wmma(A[0][1], nw_u4(bp4[1]), nw_wmma(A[0][0], nw_u4(bp4[0]), m));
                            const nw_i8 d1 = nw_wmma(A[1][1], nw_u4(bp4[3]), nw_wmma(A[1][0], nw_u4(bp4[2]), m));
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const float v = fmaf(bx.y, dotf(d1[i]), bx.x * dotf(d0[i]));
                                acc[mt][nt][i] = fmaf(w0[0][i], v, acc[mt][nt][i]);
                            }
                        } else {
#pragma unroll
                            for (int h = 0; h < 2; ++h) {
                                const float dx = h ? bx.y : bx.x;
                                const uint4* bp4 = reinterpret_cast<const uint4*>(br + 32 * h);
                                const nw_i4 B0 = nw_u4(bp4[0]), B1 = nw_u4(bp4[1]);
                                if constexpr (K16) {
                                    const nw_i8 d0 = nw_wmma(A[h][0], B0, m), d1 = nw_wmma(A[h][1], B1, m);
#pragma unroll
                                    for (int i = 0; i < 8; ++i) {
                                        const float v = fmaf(w1[h][i], dotf(d1[i]), w0[h][i] * dotf(d0[i]));
                                        acc[mt][nt][i] = fmaf(dx, v, acc[mt][nt][i]);
                                    }
                                } else {
                                    const nw_i8 d = nw_wmma(A[h][1], B1, nw_wmma(A[h][0], B0, m));
#pragma unroll
                                    for (int i = 0; i < 8; ++i) {
                                        acc[mt][nt][i] = fmaf(w0[h][i] * dx, dotf(d[i]), acc[mt][nt][i]);
                                        if constexpr (has_min(WT)) acc[mt][nt][i] = fmaf(w1[h][i] * dx, h ? bsum.y : bsum.x, acc[mt][nt][i]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (mode != 2) {
                put(raw, (s + 1) & 1);
                store_b((s + 1) & 1, sa0, sa1, sa2);
            }
            if (mode == 0) load_unit<WT>(unit(cur, s + 2), sub(s + 2), raw);
            else if (mode == 1) load_unit<WT>(unit(nxt, 0), sub(0), raw0);
            else load_unit<WT>(unit(nxt, 1), sub(1), raw);
        };
        static_assert(NS % 2 == 0 && NS >= 4, "the stage loop is unrolled by two with the last two stages peeled");
        for (int s = 0; s + 3 < NS; s += 2) {
            stage(s, rc0, rc1, rc2, ra0, ra1, ra2, 0);
            stage(s + 1, ra0, ra1, ra2, rc0, rc1, rc2, 0);
        }
        stage(NS - 2, rc0, rc1, rc2, ra0, ra1, ra2, 1);
        stage(NS - 1, ra0, ra1, ra2, rc0, rc1, rc2, 2);
        if constexpr (GU) {
            nw_bar();                                      // every wave is done reading the arena
            if (on) {
#pragma unroll
                for (int nt = 0; nt < NTT; ++nt) {
                    if (nt > 0 && !on_nt[nt]) continue;
                    // lane (hi = 0, l16) holds the gate rows of its 16 features at token (TT / 2) wn + 16 nt + l16,
                    // its partner lane (hi = 1) the up rows: h = silu(gate) up, and the lane's max |h|
                    float h[16], am = 0.0f;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float gt = acc[mt][nt][i], up = nw_swap16(gt);
                            const float v = gt * __builtin_amdgcn_rcpf(1.0f + __expf(-gt)) * up;
                            h[8 * mt + i] = v;
                            am = fmaxf(am, fabsf(v));
                        }
                    if (hi == 0) {
                        const int r = (TT / 2) * wn + 16 * nt + l16;
                        float4* p = reinterpret_cast<float4*>(&hs[r][16 * wm]);
#pragma unroll
                        for (int c = 0; c < 4; ++c) p[c] = make_float4(h[4 * c], h[4 * c + 1], h[4 * c + 2], h[4 * c + 3]);
                        hm[wm][r] = am;
                    }
                }
            }
            nw_bar();
            // H block fb (features 64 fb ..) to int8 per 32, natural order: a thread per (token, half)
            if (tid < 2 * TT) {
                const int r = tid & (TT - 1), hh = tid / TT;
                if (r < nrows) {
                    const float am = fmaxf(hm[2 * hh][r], hm[2 * hh + 1][r]);
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (size_t) fb * AB;
                    const float4* hp = reinterpret_cast<const float4*>(&hs[r][32 * hh]);
                    uint32_t wd[8];
                    int csum = 0;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float4 v = hp[j];
                        const int c0 = __float2int_rn(v.x * inv), c1 = __float2int_rn(v.y * inv),
                                  c2 = __float2int_rn(v.z * inv), c3 = __float2int_rn(v.w * inv);
                        csum += c0 + c1 + c2 + c3;
                        wd[j] = (uint32_t) (uint8_t) (int8_t) c0 | ((uint32_t) (uint8_t) (int8_t) c1 << 8) |
                                ((uint32_t) (uint8_t) (int8_t) c2 << 16) | ((uint32_t) (uint8_t) (int8_t) c3 << 24);
                    }
                    uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                    o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                    o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                    *reinterpret_cast<float*>(o + 64 + 4 * hh) = am / 127.0f;
                    *reinterpret_cast<float*>(o + 72 + 4 * hh) = (float) csum;   // (the formats with a minimum)
                }
            }
        } else {
            if (on) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int nt = 0; nt < NTT; ++nt) {
                        const int r = (TT / 2) * wn + 16 * nt + l16;
                        if (r < nrows) {
                            const size_t at = (size_t) (row0 + r) * 2560 + rbase + 32 * wm + 16 * mt + hi;
                            if (dm16) {   // STRATA_DM_F16: FP16, clamped to its range (the combine reads half the bytes)
                                uint16_t* d = reinterpret_cast<uint16_t*>(dm) + at;
#pragma unroll
                                for (int i = 0; i < 8; ++i)
                                    d[2 * i] = __builtin_bit_cast(uint16_t, (_Float16) fminf(fmaxf(acc[mt][nt][i], -65504.0f), 65504.0f));
                            } else {
                                float* d = dm + at;
#pragma unroll
                                for (int i = 0; i < 8; ++i) d[2 * i] = acc[mt][nt][i];
                            }
                        }
                    }
            }
        }
        cur = nxt;
        ei1 = ei2; j1 = j2; oa1 = oa2; ob1 = ob2;
#pragma unroll
        for (int q = 0; q < NP; ++q) sr1[q] = sr2[q];
        ei2 = ei3; j2 = j3; oa2 = oa3; ob2 = ob3;
        ei3 = ei4; j3 = j4;
    }
#else
    __builtin_trap();
#endif
}
#endif  // __HIPCC__

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

// per device: whether every kernel here runs (sm_80+, device code in this build, fits), and their occupancy
struct DevInfo {
    bool done = false, ok = false;
    int sms = 0, occ = 1;
};
std::mutex g_mu;
DevInfo g_dev[32];

#if !defined(__HIPCC__)
template <int T, bool GU, int WW> bool setup_ww(int& occ) {
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, native_kernel<T, GU, WW>) != cudaSuccess || fa.ptxVersion < 80) return false;
    if (cudaFuncSetAttribute(native_kernel<T, GU, WW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smem_bytes(T, WW)) != cudaSuccess)
        return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, native_kernel<T, GU, WW>, THREADS, smem_bytes(T, WW)) !=
            cudaSuccess || o < 1)
        return false;
    occ = std::min(occ, o);
    return true;
}
template <int T, bool GU> bool setup_one(int& occ) { return setup_ww<T, GU, 4>(occ) && setup_ww<T, GU, 2>(occ); }
#endif

#if defined(__HIPCC__)
// Resident blocks per multiprocessor for the gfx11 WMMA prompt-expert kernels' persistent grids.  HIP counts a gfx11
// WGP (two CUs, which these kernels use as one in the default WGP mode) as one multiprocessor, and the occupancy query
// answers 1 block for these kernels where 2 run side by side - so the grid was half the GPU's room.  gfx1151 (20
// WGPs), 8192 tokens x top 10, IQ3_S / IQ4_NL, one layer: 1 block per WGP 60.7 ms, 2 44.7, 3 49.9, 4 45.2 (the same
// work items and arithmetic: the results do not change).  STRATA_PF_OCC=N sets the blocks per WGP (an experiment knob).
static int wgp_blocks(int occ) {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_OCC");
        return v != nullptr ? std::atoi(v) : 0;
    }();
    return env > 0 ? env : 2 * std::max(occ, 1);
}
#endif
const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
#if defined(STRATA_HIP_GFX906)
    // gfx906 reports compute capability 9.0 through HIP, but the mma.sync bodies above are CUDA sm_80+ only and
    // empty in a hipcc build: never available here (the prompt path keeps its own expert GEMMs)
    return d;
#elif defined(__HIPCC__)
    {   // gfx11 with this build's code: the WMMA kernels (one occupancy for all: the same shape and LDS budget)
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceProp prop;
        hipFuncAttributes fa{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || std::strncmp(prop.gcnArchName, "gfx11", 5) != 0 ||
            hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(native_w11_kernel<T_IQ3_XXS, true>)) != hipSuccess) {
            cudaGetLastError();
            return d;
        }
        int o = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&o, native_w11_kernel<T_IQ2_S, true>, NW_THREADS, 0) != hipSuccess)
            o = 0;
        d.ok = o >= 1;
        d.occ = d.ok ? o : 1;
        cudaGetLastError();
        return d;
    }
#else
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    if (major < 8) return d;
    int occ = 1 << 20;
    d.ok = setup_one<T_IQ2_XXS, true>(occ) && setup_one<T_IQ2_XS, true>(occ) && setup_one<T_IQ2_S, true>(occ) &&
           setup_one<T_IQ3_XXS, true>(occ) && setup_one<T_IQ3_S, true>(occ) && setup_one<T_IQ4_XS, true>(occ) &&
           setup_one<T_Q2_0, false>(occ) && setup_one<T_IQ4_NL, false>(occ);
    d.occ = d.ok ? occ : 1;
    cudaGetLastError();
#endif
    return d;
}

// opt-in on top of STRATA_PF_FUSED=1: STRATA_PF_FUSED_KQ=1 takes UD-Q4_K_XL's formats (the output differs from the FP16
// expert path's at rounding level; KL-gated, see aurora_s23.md)
bool kq_on() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_KQ");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}
bool gu_covered(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS
#if defined(__HIPCC__)
           || ((t == T_Q4_K || t == T_Q5_K) && kq_on())   // the gfx11 kernels only
           || t == T_Q2_0                                 // the gfx11 kernels only: a native Q2_0 pack (gate/up)
#endif
        ;
}
bool d_covered(int t) {
    return t == T_Q2_0 || t == T_IQ4_NL
#if defined(__HIPCC__)
           || ((t == T_Q5_1 || t == T_Q8_0) && kq_on())
#endif
        ;
}

template <int T, bool GU>
void launch(int ww, unsigned grid, const Batch& b, const NativeGeom& g, const Tables& tb, const void* act,
            const int32_t* src, void* out, float* dm, cudaStream_t s) {
    const uint8_t* a = (const uint8_t*) act;
    uint8_t* o = (uint8_t*) out;
    if (ww == 4) native_kernel<T, GU, 4><<<grid, THREADS, smem_bytes(T, 4), s>>>(b, g, tb, a, src, o, dm);
    else native_kernel<T, GU, 2><<<grid, THREADS, smem_bytes(T, 2), s>>>(b, g, tb, a, src, o, dm);
}

// The work item's shape for a layer of `n` routed rows over `n_expert` experts: 128 routed rows when an expert has
// about one such tile (56 to 112 rows on average), else 64.  Measured on the RTX 5070 (tests/cuda/prefill_fused_iq_test,
// a layer of 2048 / 3584 / 8192 tokens, ms, 64 -> 128): IQ2_S 5.2 -> 6.7, 9.5 -> 9.3, 17.5 -> 19.5; IQ2_XXS 4.2 ->
// 5.8, 7.4 -> 7.6, 13.6 -> 16.5; IQ3_S 5.6 -> 6.6, 10.2 -> 8.8, 18.4 -> 19.0; IQ3_XXS 5.1 -> 6.3, 9.1 -> 8.3, 16.5 ->
// 17.5 (256 rows was slower still).  STRATA_PF_FUSED_TILE=64|128 forces one.
int pick_ww(int64_t n, int n_expert) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_PF_FUSED_TILE");
        const int t = v ? std::atoi(v) : 0;
        return t == 64 ? 4 : t == 128 ? 2 : 0;
    }();
    if (forced) return forced;
    const double avg = (double) n / std::max(n_expert, 1);
    return avg > 56.0 && avg <= 112.0 ? 2 : 4;
}

}  // namespace

bool native_supported(int gu_type, int d_type) {
    static const bool off = [] {   // STRATA_PF_FUSED_NATIVE=0: the native packs keep MMQ under STRATA_PF_FUSED=1 (A/B)
        const char* v = std::getenv("STRATA_PF_FUSED_NATIVE");
        return v != nullptr && v[0] == '0';
    }();
    return !off && requested() && gu_covered(gu_type) && d_covered(d_type) && dev_info().ok;   // opt-in (=1)
}

void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_nat_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act_native");
}

void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream, bool dm16) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const int dm16i = dm16 ? 1 : 0;
    const DevInfo& d = dev_info();
    if (!d.ok || !gu_covered(g.gu_type) || !d_covered(g.d_type)) {
        std::fprintf(stderr, "prefill fused experts (native): types %d / %d are not covered here\n", g.gu_type, g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
#if defined(__HIPCC__)
    {
        const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
        // W (X1): LDS alias of the H tile + waves_per_eu(8) (192 VGPRs) + 4 blocks per WGP, only for the types whose
        // kernels do not spill under the cap (measured: gate/up IQ2_*, IQ3_*, IQ4_XS 0-5 regs; down IQ4_NL 11, Q8_0 / Q2_0 0;
        // Q4_K / Q5_K gate/up and Q5_1 down spill 100+ and keep the old launch).
#ifdef STRATA_W_NO_OCC
        const bool occ_gu = false, occ_d = false;
#else
        const bool occ_gu = g.gu_type == T_IQ2_XXS || g.gu_type == T_IQ2_XS || g.gu_type == T_IQ2_S ||
                            g.gu_type == T_IQ3_XXS || g.gu_type == T_IQ3_S || g.gu_type == T_IQ4_XS || g.gu_type == T_Q2_0;
        const bool occ_d = g.d_type == T_Q2_0 || g.d_type == T_Q8_0 || g.d_type == T_IQ4_NL;
#endif
        const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / NW_ROWS), (int64_t) d.sms * (occ_gu ? 4 : wgp_blocks(d.occ)));
        const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / NW_ROWS), (int64_t) d.sms * (occ_d ? 4 : wgp_blocks(d.occ)));
        const uint8_t* xa8 = (const uint8_t*) xa;
        uint8_t* ha8 = (uint8_t*) ha;
#define STRATA_NW_GU(T) do { if (occ_gu) native_w11_kernel<T, true, true, NW_LB><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr); else native_w11_kernel<T, true><<<g_gu, NW_THREADS, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr); } while (0)
        switch (g.gu_type) {
            case T_IQ2_XXS: STRATA_NW_GU(T_IQ2_XXS); break;
            case T_IQ2_XS: STRATA_NW_GU(T_IQ2_XS); break;
            case T_IQ2_S: STRATA_NW_GU(T_IQ2_S); break;
            case T_IQ3_XXS: STRATA_NW_GU(T_IQ3_XXS); break;
            case T_IQ3_S: STRATA_NW_GU(T_IQ3_S); break;
            case T_Q4_K: STRATA_NW_GU(T_Q4_K); break;
            case T_Q5_K: STRATA_NW_GU(T_Q5_K); break;
            case T_Q2_0: STRATA_NW_GU(T_Q2_0); break;
            default: STRATA_NW_GU(T_IQ4_XS); break;
        }
#undef STRATA_NW_GU
#define STRATA_NW_D(T) do { if (occ_d) native_w11_kernel<T, false, true, NW_LB><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm, dm16i); else native_w11_kernel<T, false><<<g_d, NW_THREADS, 0, s>>>(b, g, tb, ha8, src, nullptr, dm, dm16i); } while (0)
        if (g.d_type == T_Q2_0) STRATA_NW_D(T_Q2_0);
        else if (g.d_type == T_Q5_1) STRATA_NW_D(T_Q5_1);
        else if (g.d_type == T_Q8_0) STRATA_NW_D(T_Q8_0);
        else STRATA_NW_D(T_IQ4_NL);
#undef STRATA_NW_D
        ck(cudaGetLastError(), "experts_native");
        return;
    }
#endif
    const int ww = pick_ww(n, n_expert);
    // the most 64-row tiles the batch can have (every row in it, plus a partial tile per expert) - the items of the
    // ones inside a larger item end at once
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    switch (g.gu_type) {
        case T_IQ2_XXS: launch<T_IQ2_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_S: launch<T_IQ2_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_S: launch<T_IQ3_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        default: launch<T_IQ4_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
    }
    if (g.d_type == T_Q2_0) launch<T_Q2_0, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    else launch<T_IQ4_NL, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "experts_native");
}

}  // namespace strata::prefill::fused
