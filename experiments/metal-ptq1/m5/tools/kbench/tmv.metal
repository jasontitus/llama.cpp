#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// Tensor-accelerator PTQ1_0 matvec for a few columns.
// A tile: NR0 rows x NK weights decoded to half, w = (t-1)*d is exact in half.
// B tile: each activation column split into hi = half(y), lo = half(y - hi); products with the
// exact weights are exact and the tensor op accumulates in fp32, so hi+lo keeps ~22 bits of y.
template<int NC, int NR0, int NBK, bool HILO, int MODE = 0>
kernel void tmv(constant kargs & args, device const char * src0, device const char * src1, device float * dst,
                threadgroup char * shmem[[threadgroup(0)]],
                uint3 tgpig[[threadgroup_position_in_grid]], ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr int NK  = 128*NBK;
    constexpr int NCB = HILO ? 2*NC : NC;              // tensor columns
    constexpr int NR1 = NCB <= 8 ? 8 : (NCB <= 16 ? 16 : 32);
    constexpr int NT  = 128;
    threadgroup half  * sa = (threadgroup half *) shmem;
    threadgroup half  * sb = (threadgroup half *) (shmem + NR0*NK*2);
    threadgroup float * sc = (threadgroup float *) shmem;

    const int r0 = tgpig.x*NR0;
    const int c0 = tgpig.y*NC;
    const int nb = args.nb;

    for (int i = tiitg; i < NR1*NK; i += NT) sb[i] = 0.0h;   // zero padded columns once

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int kb = 0; kb < nb; kb += NBK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // --- decode NR0 x NBK blocks, 7 units of work per row-block
        if (MODE != 2) for (int unit = tiitg; unit < NR0*NBK*7; unit += NT) {
            const int u   = unit % 7;
            const int rb  = unit / 7;
            const int row = rb / NBK;
            const int bk  = rb % NBK;
            device const block_ptq1_0 * blk = (device const block_ptq1_0 *) (src0 + (uint64_t) min(r0 + row, args.M - 1)*args.nb01) + kb + bk;
            const half  dh = blk->d;
            threadgroup half * out = sa + row*NK + bk*128;
            if (u < 6) {
                const float4 uu = float4(*((device const uchar4 *) blk->qs + u)) * (1.0f/256.0f);
                float4 gp = 0.0f;
                const short base = u < 4 ? 4*u : 80 + 4*(u - 4);
                const short step = u < 4 ? 16 : 8;
                float p3 = 3.0f;
                FOR_UNROLL (short n = 0; n < 5; ++n) {
                    const float4 g = floor(uu*p3);
                    const float4 t = g - 3.0f*gp;
                    gp = g; p3 *= 3.0f;
                    *((threadgroup half4 *) (out + base + n*step)) = half4(t - 1.0f) * dh;
                }
            } else {
                const float2 uu = float2(blk->qh[0], blk->qh[1]) * (1.0f/256.0f);
                float2 gp = 0.0f; float p3 = 3.0f;
                FOR_UNROLL (short n = 0; n < 4; ++n) {
                    const float2 g = floor(uu*p3);
                    const float2 t = g - 3.0f*gp;
                    gp = g; p3 *= 3.0f;
                    *((threadgroup half2 *) (out + 120 + 2*n)) = half2(t - 1.0f) * dh;
                }
            }
        }
        // --- activations: hi/lo halves
        for (int i = tiitg; i < NC*NK; i += NT) {
            const int c = i / NK, k = i % NK;
            const float y = (c0 + c < args.N) ? *((device const float *) (src1 + (uint64_t) (c0 + c)*args.nb11) + kb*128 + k) : 0.0f;
            const half hi = (half) y;
            if (HILO) { sb[(2*c)*NK + k] = hi; sb[(2*c + 1)*NK + k] = (half) (y - (float) hi); }
            else      { sb[c*NK + k] = hi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (MODE != 1) mm.run(tB, tA, cT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tiitg; i < NC*NR0; i += NT) {
        const int c = i / NR0, r = i % NR0;
        if (r0 + r < args.M && c0 + c < args.N) {
            const float v = HILO ? sc[(2*c)*NR0 + r] + sc[(2*c + 1)*NR0 + r] : sc[c*NR0 + r];
            dst[(uint64_t) (c0 + c)*args.ne0 + r0 + r] = v;
        }
    }
}
#define INST(c,r,b,h,nm) INSTM(c,r,b,h,0,nm)
#define INSTM(c,r,b,h,m,nm) template [[host_name(nm)]] kernel void tmv<c,r,b,h,m>(constant kargs &, device const char *, device const char *, device float *, threadgroup char *, uint3, ushort);
INST(1,64,1,true,"tmv_c1_r64_b1") INST(2,64,1,true,"tmv_c2_r64_b1") INST(3,64,1,true,"tmv_c3_r64_b1") INST(4,64,1,true,"tmv_c4_r64_b1")
INST(8,64,1,true,"tmv_c8_r64_b1") INST(4,32,1,true,"tmv_c4_r32_b1") INST(4,32,2,true,"tmv_c4_r32_b2") INST(8,32,2,true,"tmv_c8_r32_b2")
INST(4,64,1,false,"tmv_c4_r64_b1_h") INST(8,64,1,false,"tmv_c8_r64_b1_h") INST(16,64,1,false,"tmv_c16_r64_b1_h") INST(16,64,1,true,"tmv_c16_r64_b1")
INSTM(4,32,1,true,1,"abl_nomm_c4_r32") INSTM(4,32,1,true,2,"abl_nodec_c4_r32") INSTM(4,64,1,true,1,"abl_nomm_c4_r64") INSTM(4,64,1,true,2,"abl_nodec_c4_r64")
