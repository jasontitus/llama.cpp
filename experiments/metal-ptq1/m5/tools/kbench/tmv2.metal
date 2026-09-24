#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// v2: each simdgroup owns an NR0-row slab and a private threadgroup A region; B (hi/lo half
// activations, NP tensor columns, K contiguous) is read straight from device memory.
inline void decode_unit(device const block_ptq1_0 * blk, short u, threadgroup half * out) {
    const half dh = blk->d;
    if (u < 6) {
        const float4 uu = float4(*((device const uchar4 *) blk->qs + u)) * (1.0f/256.0f);
        float4 gp = 0.0f; float p3 = 3.0f;
        const short base = u < 4 ? 4*u : 80 + 4*(u - 4);
        const short step = u < 4 ? 16 : 8;
        FOR_UNROLL (short n = 0; n < 5; ++n) {
            const float4 g = floor(uu*p3); const float4 t = g - 3.0f*gp; gp = g; p3 *= 3.0f;
            *((threadgroup half4 *) (out + base + n*step)) = fma(half4(t), half4(dh), half4(-dh));
        }
    } else {
        const float2 uu = float2(blk->qh[0], blk->qh[1]) * (1.0f/256.0f);
        float2 gp = 0.0f; float p3 = 3.0f;
        FOR_UNROLL (short n = 0; n < 4; ++n) {
            const float2 g = floor(uu*p3); const float2 t = g - 3.0f*gp; gp = g; p3 *= 3.0f;
            *((threadgroup half2 *) (out + 120 + 2*n)) = fma(half2(t), half2(dh), half2(-dh));
        }
    }
}
template<int NC, int NP, int NR0, int NSG, int NBUF, int MODE = 0, int PAD = 0>
kernel void tmv2(constant kargs & args, device const char * src0, device const char * src1, device float * dst,
                 device half * hl [[buffer(4)]], threadgroup char * shmem[[threadgroup(0)]],
                 uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NK = 128;
    constexpr int LD = NK + PAD;   // padded row stride of the A slab (bank spread)
    threadgroup half * sa0 = (threadgroup half *) shmem + sgitg*NBUF*NR0*LD;
    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
    const int nb = args.nb;

    auto tBall = tensor<device half, dextents<int32_t, 2>, tensor_inline>(hl, dextents<int32_t, 2>(args.K, NP));
    matmul2d<matmul2d_descriptor(NP, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<1>> mm;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa0, dextents<int32_t, 2>(NK, NR0), array<int32_t, 2>({1, LD}));
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tA0), decltype(tBall), float>();

    device const block_ptq1_0 * rows[(NR0*7 + 31)/32];
    short units[(NR0*7 + 31)/32];
    FOR_UNROLL (short j = 0; j < (NR0*7 + 31)/32; ++j) {
        const short unit = tiisg + 32*j;
        const short row = min(unit / 7, NR0 - 1);
        units[j] = unit < NR0*7 ? unit % 7 : -1;
        rows[j] = (device const block_ptq1_0 *) (src0 + (uint64_t) min(r0 + row, args.M - 1)*args.nb01);
    }

    for (int kb = 0; kb < nb; ++kb) {
        threadgroup half * sa = sa0 + (kb % NBUF)*NR0*LD;
        FOR_UNROLL (short j = 0; j < (NR0*7 + 31)/32; ++j) {
            if (MODE != 2 && units[j] >= 0) {
                const short unit = tiisg + 32*j;
                decode_unit(rows[j] + kb, units[j], sa + (unit/7)*LD);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0), array<int32_t, 2>({1, LD}));
        auto tB = tBall.slice(kb*NK, 0);
        if (MODE != 1) mm.run(tB, tA, cT);
        if (NBUF == 1) simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * sc = (threadgroup float *) sa0;
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NP));
    cT.store(tC);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = tiisg; i < NC*NR0; i += 32) {
        const short c = i / NR0, r = i % NR0;
        if (r0 + r < args.M) dst[(uint64_t) c*args.ne0 + r0 + r] = sc[(2*c)*NR0 + r] + sc[(2*c + 1)*NR0 + r];
    }
}
#define INST(c,np,r,s,b,nm) INSTM(c,np,r,s,b,0,nm)
#define INSTM(c,np,r,s,b,m,nm) INSTP(c,np,r,s,b,m,0,nm)
#define INSTP(c,np,r,s,b,m,pd,nm) template [[host_name(nm)]] kernel void tmv2<c,np,r,s,b,m,pd>(constant kargs &, device const char *, device const char *, device float *, device half *, threadgroup char *, uint3, ushort, ushort);
INST(2,8,16,4,1,"t2_c2_r16_s4") INST(4,8,16,4,1,"t2_c4_r16_s4") INST(8,16,16,4,1,"t2_c8_r16_s4")
INST(4,8,32,4,1,"t2_c4_r32_s4") INST(8,16,32,4,1,"t2_c8_r32_s4") INST(4,8,16,4,2,"t2_c4_r16_s4_db") INST(8,16,16,4,2,"t2_c8_r16_s4_db")
INST(4,8,16,2,1,"t2_c4_r16_s2") INST(4,8,16,8,1,"t2_c4_r16_s8") INST(3,8,16,4,1,"t2_c3_r16_s4")
INSTM(8,16,16,4,1,1,"abl2_nomm") INSTM(8,16,16,4,1,2,"abl2_nodec")
INSTP(4,8,16,4,1,0,8,"t2_c4_p8") INSTP(8,16,16,4,1,0,8,"t2_c8_p8") INSTP(8,16,16,4,1,1,8,"abl2_nomm_p8") INSTP(8,16,16,4,1,0,4,"t2_c8_p4") INSTP(8,16,16,4,1,0,16,"t2_c8_p16")
INST(8,16,16,2,1,"t2_c8_r16_s2") INST(8,16,16,1,1,"t2_c8_r16_s1") INST(8,16,8,4,1,"t2_c8_r8_s4") INST(8,16,8,8,1,"t2_c8_r8_s8") INST(4,8,16,1,1,"t2_c4_r16_s1")
