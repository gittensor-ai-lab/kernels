// Fused quantized MoE expert FFN for decode (batch small).
//
// Closes most of the gap vs llama.cpp on Qwen3-MoE decode:
//   - dequantizes ONLY the top_k routed experts, on-read inside the GEMV — no
//     bf16 materialization, no 16x wasted dequant of unused experts.
//   - one warp per output row; thousands of warps fill the GPU (vs one CTA).
//   - reads GGUF-native quantized weights directly (gate/up = Q4_K [E,F,H],
//     down = Q6_K [E,H,F]). Decode is memory-bound on the quantized weight reads
//     — the right regime for a CUDA-core GEMV.
//   - down pass accumulates the top_k experts inside each warp and writes the
//     output once (no atomics, no scratch).
//
// Q4_K/Q6_K decoders are the byte-exact ones validated in dequant_gguf.cu.
// Requires hidden and ffn to be multiples of 256 (Qwen3-30B-A3B: 2048, 768).
//
// Portable CUDA — sm_89 .. sm_120/sm_121.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

static constexpr int WPB = 8;   // warps per block

__device__ __forceinline__ float q4kf_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}
__device__ __forceinline__ float q4kf_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}
__device__ __forceinline__ void q4kf_scale_min(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4); }
}
__device__ __forceinline__ void warp_deq_q4k(const unsigned char* blk, float* s, int lane) {
    float d = q4kf_h2f(blk), dmin = q4kf_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        int s1, m1, s2, m2;
        q4kf_scale_min(2*g,   sc, &s1, &m1); q4kf_scale_min(2*g+1, sc, &s2, &m2);
        float d1 = d*s1, mm1 = dmin*m1, d2 = d*s2, mm2 = dmin*m2;
        const unsigned char* q = qs + g*32;
        s[g*64 + lane]      = d1 * (q[lane] & 0xF) - mm1;
        s[g*64 + 32 + lane] = d2 * (q[lane] >> 4)  - mm2;
    }
}
__device__ __forceinline__ void warp_deq_q6k(const unsigned char* blk, float* s, int lane) {
    const unsigned char* ql = blk; const unsigned char* qh = blk + 128;
    const signed char* sc = (const signed char*)(blk + 192); float d = q4kf_h2f(blk + 208);
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        const unsigned char* qln = ql + n*64; const unsigned char* qhn = qh + n*32; const signed char* scn = sc + n*8;
        int l = lane, is = l / 16;
        int q1 = (int)((qln[l]    & 0xF) | (((qhn[l] >> 0) & 3) << 4)) - 32;
        int q2 = (int)((qln[l+32] & 0xF) | (((qhn[l] >> 2) & 3) << 4)) - 32;
        int q3 = (int)((qln[l]    >> 4)  | (((qhn[l] >> 4) & 3) << 4)) - 32;
        int q4 = (int)((qln[l+32] >> 4)  | (((qhn[l] >> 6) & 3) << 4)) - 32;
        s[n*128 + l]      = d * scn[is + 0] * q1;
        s[n*128 + l + 32] = d * scn[is + 2] * q2;
        s[n*128 + l + 64] = d * scn[is + 4] * q3;
        s[n*128 + l + 96] = d * scn[is + 6] * q4;
    }
}
__device__ __forceinline__ float q4kf_silu(float x) { return x / (1.f + __expf(-x)); }

// ggml types: Q4_K=12 (144 B/256), Q6_K=14 (210 B/256). Q4_K_M mixes them per tensor.
__device__ __forceinline__ int q_block_bytes(int t) { return t == 14 ? 210 : 144; }
__device__ __forceinline__ void warp_deq(int t, const unsigned char* blk, float* s, int lane) {
    if (t == 14) warp_deq_q6k(blk, s, lane);
    else         warp_deq_q4k(blk, s, lane);
}

// gate_up: h[ts,f] = SiLU(<x, gate[e,f]>) * <x, up[e,f]>.  one warp per f.
// grid=(num_tokens*top_k, ffn/WPB), block=WPB*32. smem: s_x[hidden] + WPB*256.
__global__ void gate_up_q4k_kernel(
    const __nv_bfloat16* __restrict__ input, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int H, int F, int top_k, int gate_type, int up_type
) {
    extern __shared__ float smem[];          // s_x[H]
    float* s_x = smem;
    __shared__ float s_deq_all[WPB][256];
    float* s_deq = s_deq_all[threadIdx.x / 32];
    const int ts = blockIdx.x, tok = ts / top_k;
    const int e = expert_ids[ts];
    for (int i = threadIdx.x; i < H; i += blockDim.x) s_x[i] = __bfloat162float(input[(size_t)tok * H + i]);
    __syncthreads();

    const int lane = threadIdx.x % 32;
    const int f = blockIdx.y * WPB + (threadIdx.x / 32);
    if (f >= F) return;
    const int nblk = H / 256;
    const int gbb = q_block_bytes(gate_type), ubb = q_block_bytes(up_type);
    const unsigned char* gbase = gate_q + ((size_t)e * F + f) * nblk * gbb;
    const unsigned char* ubase = up_q   + ((size_t)e * F + f) * nblk * ubb;
    float g = 0.f, u = 0.f;
    for (int blk = 0; blk < nblk; blk++) {
        warp_deq(gate_type, gbase + (size_t)blk * gbb, s_deq, lane); __syncwarp();
        float p = 0.f;
        #pragma unroll
        for (int e8 = 0; e8 < 8; e8++) p += s_deq[lane + e8*32] * s_x[blk*256 + lane + e8*32];
        g += q4kf_wsum(p); __syncwarp();
        warp_deq(up_type, ubase + (size_t)blk * ubb, s_deq, lane); __syncwarp();
        p = 0.f;
        #pragma unroll
        for (int e8 = 0; e8 < 8; e8++) p += s_deq[lane + e8*32] * s_x[blk*256 + lane + e8*32];
        u += q4kf_wsum(p); __syncwarp();
    }
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(g) * u;
}

// down: out[tok,hh] = sum_j weight_j * <h[tok,j], down[e_j, hh]>.
// one warp per (token, hh); loops over top_k experts internally and writes once.
// grid=(num_tokens, hidden/WPB), block=WPB*32. smem: WPB*256 (s_deq per warp).
__global__ void down_q6k_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const float* __restrict__ h_scratch,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int down_type
) {
    __shared__ float s_deq_all[WPB][256];
    float* s_deq = s_deq_all[threadIdx.x / 32];
    const int token = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int hh = blockIdx.y * WPB + (threadIdx.x / 32);
    if (hh >= H) return;
    const int nblk = F / 256;
    const int dbb = q_block_bytes(down_type);

    float acc = 0.f;
    for (int j = 0; j < top_k; j++) {
        const int ts = token * top_k + j;
        const int e = expert_ids[ts];
        const float w = expert_weights[ts];
        const unsigned char* dbase = down_q + ((size_t)e * H + hh) * nblk * dbb;
        float y = 0.f;
        for (int blk = 0; blk < nblk; blk++) {
            warp_deq(down_type, dbase + (size_t)blk * dbb, s_deq, lane); __syncwarp();
            float p = 0.f;
            #pragma unroll
            for (int e8 = 0; e8 < 8; e8++)
                p += s_deq[lane + e8*32] * h_scratch[(size_t)ts * F + blk*256 + lane + e8*32];
            y += q4kf_wsum(p); __syncwarp();
        }
        acc += w * y;
    }
    if (lane == 0) output[(size_t)token * H + hh] = __float2bfloat16(acc);
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/moe.h"

void launch_moe_expert_ffn_q4k(
    const void* input, const void* gate_q, const void* up_q, const void* down_q,
    int gate_type, int up_type, int down_type,
    const int* expert_ids, const float* expert_weights, void* output,
    float* h_scratch, float* out_scratch,
    int num_tokens, int top_k, int hidden, int ffn, cudaStream_t stream
) {
    (void)out_scratch;
    dim3 gu(num_tokens * top_k, (ffn + WPB - 1) / WPB);
    size_t gu_smem = (size_t)hidden * sizeof(float);   // s_x only; s_deq is static
    gate_up_q4k_kernel<<<gu, WPB * 32, gu_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<const unsigned char*>(gate_q),
        reinterpret_cast<const unsigned char*>(up_q),
        expert_ids, h_scratch, hidden, ffn, top_k, gate_type, up_type);

    dim3 dn(num_tokens, (hidden + WPB - 1) / WPB);
    down_q6k_kernel<<<dn, WPB * 32, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(down_q),
        expert_ids, expert_weights, h_scratch,
        reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, down_type);
}
#endif

} // namespace kernels
} // namespace sparkinfer
