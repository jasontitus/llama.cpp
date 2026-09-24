#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// v3: A holds the exact base-3 partial quotients g_k = floor(3^k b/256) (integers <= 242, exact in
// half) in trit-major order; B holds the matching collapse coefficients (hi/lo halves). Per 128-block
// the tensor op yields sum g*c; the block scale and the -1 offset are applied per element:
// acc += d[row] * (blk - sumy[col]).
inline void decode_g(device const block_ptq1_0 * blk, short u, threadgroup half * out) {
    if (u < 6) {
        const float4 uu = float4(*((device const uchar4 *) blk->qs + u)) * (1.0f/256.0f);
        float p3 = 3.0f;
        FOR_UNROLL (short k = 0; k < 5; ++k) {
            *((threadgroup half4 *) (out + k*24 + 4*u)) = half4(floor(uu*p3)); p3 *= 3.0f;
        }
    } else {
        const float2 uu = float2(blk->qh[0], blk->qh[1]) * (1.0f/256.0f);
        float p3 = 3.0f;
        FOR_UNROLL (short k = 0; k < 4; ++k) {
            *((threadgroup half2 *) (out + 120 + 2*k)) = half2(floor(uu*p3)); p3 *= 3.0f;
        }
    }
}
template<int NC, int NP, int NR0, int NSG>
kernel void tmv3(constant kargs & args, device const char * src0, device const char * src1, device float * dst,
                 device half * hl [[buffer(4)]], device const float * sumy [[buffer(5)]], threadgroup char * shmem[[threadgroup(0)]],
                 uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NK = 128;
    threadgroup half  * sa = (threadgroup half *) shmem + sgitg*(NR0*NK + NR0);
    threadgroup half  * sd = sa + NR0*NK;                 // this block's scale per row
    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
    const int nb = args.nb;

    auto tBall = tensor<device half, dextents<int32_t, 2>, tensor_inline>(hl, dextents<int32_t, 2>(args.K, NP));
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    matmul2d<matmul2d_descriptor(NP, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply), execution_simdgroups<1>> mm;
    auto cB  = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tBall), float>();
    auto acc = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tBall), float>();

    // element -> (row, tensor column), fixed across blocks
    short erow[16], ecol[16];
    FOR_UNROLL (short i = 0; i < 16; ++i) {
        if (i < acc.get_capacity() && acc.is_valid_element(i)) {
            auto idx = acc.get_multidimensional_index(i);
            erow[i] = idx[0]; ecol[i] = idx[1];
        } else { erow[i] = -1; ecol[i] = 0; }
        if (i < acc.get_capacity()) acc[i] = 0.0f;
    }

    device const block_ptq1_0 * rows[(NR0*7 + 31)/32];
    short units[(NR0*7 + 31)/32];
    FOR_UNROLL (short j = 0; j < (NR0*7 + 31)/32; ++j) {
        const short unit = tiisg + 32*j;
        units[j] = unit < NR0*7 ? unit % 7 : -1;
        rows[j] = (device const block_ptq1_0 *) (src0 + (uint64_t) min(r0 + min(unit/7, NR0 - 1), args.M - 1)*args.nb01);
    }

    for (int kb = 0; kb < nb; ++kb) {
        FOR_UNROLL (short j = 0; j < (NR0*7 + 31)/32; ++j) {
            if (units[j] >= 0) {
                const short unit = tiisg + 32*j;
                decode_g(rows[j] + kb, units[j], sa + (unit/7)*NK);
                if (units[j] == 6) sd[unit/7] = rows[j][kb].d;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto tB = tBall.slice(kb*NK, 0);
        mm.run(tB, tA, cB);
        FOR_UNROLL (short i = 0; i < 16; ++i) {
            if (erow[i] >= 0) {
                const short c = ecol[i];
                const float sy = (c & 1) || (c >> 1) >= NC ? 0.0f : sumy[(c >> 1)*nb + kb];
                acc[i] += (float) sd[erow[i]] * (cB[i] - sy);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float * sc = (threadgroup float *) sa;
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NP));
    acc.store(tC);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = tiisg; i < NC*NR0; i += 32) {
        const short c = i / NR0, r = i % NR0;
        if (r0 + r < args.M) dst[(uint64_t) c*args.ne0 + r0 + r] = sc[(2*c)*NR0 + r] + sc[(2*c + 1)*NR0 + r];
    }
}
#define INST(c,np,r,s,nm) template [[host_name(nm)]] kernel void tmv3<c,np,r,s>(constant kargs &, device const char *, device const char *, device float *, device half *, device const float *, threadgroup char *, uint3, ushort, ushort);
INST(2,16,16,4,"t3_c2") INST(4,16,16,4,"t3_c4") INST(8,16,16,4,"t3_c8") INST(4,8,16,4,"t3_c4_np8")
