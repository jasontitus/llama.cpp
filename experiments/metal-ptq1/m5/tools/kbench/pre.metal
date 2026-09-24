#include "common.h"
// Variant P: activations pre-laid-out once (CUDA "rearrange activations" analog). Each lane reads its
// 16 collapse coefficients + sumy as five float4 loads instead of 16 loads + 28 flops per column.
template<int nr1>
inline void dot_pre(device const block_ptq1_0 * qb, thread const float4 (&yl)[nr1][5], short it, float p0, thread float (&sumf)[nr1]) {
    float acc[nr1] = {};
    FOR_UNROLL (short byte = 0; byte < 3; ++byte) {
        const float u = (float) qb->qs[byte < 2 ? 2*it + byte : 16 + it] * (1.0f/256.0f);
        const float g[5] = {floor(3.0f*u), floor(9.0f*u), floor(27.0f*u), floor(81.0f*u), floor(243.0f*u)};
        FOR_UNROLL (short col = 0; col < nr1; ++col) FOR_UNROLL (short n = 0; n < 5; ++n) {
            const short e = 5*byte + n; acc[col] += g[n] * yl[col][e/4][e%4];
        }
    }
    const float u = (float) qb->qh[it & 1] * (1.0f/256.0f);
    const float t = floor(3.0f*p0*u) - 3.0f*floor(p0*u);
    const float d = (float) qb->d;
    FOR_UNROLL (short col = 0; col < nr1; ++col) { acc[col] += t * yl[col][3][3]; sumf[col] += (acc[col] - yl[col][4][0]) * d; }
}
template<int nr0, int nr1>
kernel void pre(constant kargs & args, device const char * src0, device const char * src1, device float * dst, device const float4 * stg [[buffer(4)]],
               uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int nb = args.nb;
    const int r1 = tgpig.y * nr1;
    const int first_row = tgpig.x * nr0;
    device const block_ptq1_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) ax[row] = (device const block_ptq1_0 *) (src0 + (uint64_t) min(first_row + row, args.M - 1)*args.nb01);
    float sumf[nr0][nr1] = {};
    const short ix = tiisg/8, it = tiisg%8;
    const float p0 = (float[4]){1,3,9,27}[it >> 1];
    for (int ib = ix; ib < nb; ib += 4) {
        float4 yl[nr1][5];
        FOR_UNROLL (short col = 0; col < nr1; ++col) {
            device const float4 * s = stg + (((uint64_t) (r1 + col)*nb + ib)*8 + it)*5;
            FOR_UNROLL (short q = 0; q < 5; ++q) yl[col][q] = s[q];
        }
        FOR_UNROLL (short row = 0; row < nr0; row++) dot_pre<nr1>(ax[row] + ib, yl, it, p0, sumf[row]);
    }
    for (int row = 0; row < nr0; ++row) FOR_UNROLL (short col = 0; col < nr1; ++col) {
        const float tot = simd_sum(sumf[row][col]);
        if (tiisg == 0 && first_row + row < args.M) dst[(uint64_t) (r1 + col)*args.ne0 + first_row + row] = tot;
    }
}
#define INST(r,c,nm) template [[host_name(nm)]] kernel void pre<r,c>(constant kargs &, device const char *, device const char *, device float *, device const float4 *, uint3, ushort, ushort);
INST(4,1,"pre_r4_c1") INST(8,1,"pre_r8_c1") INST(4,2,"pre_r4_c2") INST(8,2,"pre_r8_c2") INST(4,3,"pre_r4_c3") INST(4,4,"pre_r4_c4") INST(2,4,"pre_r2_c4") INST(8,4,"pre_r8_c4")
