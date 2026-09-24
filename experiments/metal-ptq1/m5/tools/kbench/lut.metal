#include "common.h"
// Variant A: byte -> (g1..g5) from a threadgroup table instead of 5 mul+floor.
// Table entry b holds floor(3^k*b/256) for k=1..5, exactly the kernel's float decode.
template<int nr1>
inline void dot_lut(device const block_ptq1_0 * qb, threadgroup const float4 * T4, threadgroup const float * T1,
                    thread const float (&yl)[nr1][17], thread const float (&sumy)[nr1], short it, thread float (&sumf)[nr1]) {
    float acc[nr1] = {};
    FOR_UNROLL (short byte = 0; byte < 3; ++byte) {
        const uint b = qb->qs[byte < 2 ? 2*it + byte : 16 + it];
        const float4 g = T4[b]; const float g4 = T1[b];
        FOR_UNROLL (short col = 0; col < nr1; ++col) {
            acc[col] += g.x*yl[col][5*byte+0]; acc[col] += g.y*yl[col][5*byte+1]; acc[col] += g.z*yl[col][5*byte+2];
            acc[col] += g.w*yl[col][5*byte+3]; acc[col] += g4 *yl[col][5*byte+4];
        }
    }
    const float u = (float) qb->qh[it & 1] * (1.0f/256.0f);
    const float p0 = yl[0][16];
    const float t = floor(3.0f*p0*u) - 3.0f*floor(p0*u);
    const float d = (float) qb->d;
    FOR_UNROLL (short col = 0; col < nr1; ++col) { acc[col] += t * yl[col][15]; sumf[col] += (acc[col] - sumy[col]) * d; }
}
template<int nr0, int nr1, int NSG>
kernel void lut(constant kargs & args, device const char * src0, device const char * src1, device float * dst,
               threadgroup char * shmem[[threadgroup(0)]],
               uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup float4 * T4 = (threadgroup float4 *) shmem;
    threadgroup float  * T1 = (threadgroup float *) (shmem + 256*16);
    for (short b = tiisg + 32*sgitg; b < 256; b += 32*NSG) {
        const float u = (float) b * (1.0f/256.0f);
        T4[b] = float4(floor(3.0f*u), floor(9.0f*u), floor(27.0f*u), floor(81.0f*u)); T1[b] = floor(243.0f*u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int nb = args.nb;
    const int r1 = tgpig.y * nr1;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    device const float * y = (device const float *) (src1 + (uint64_t) r1*args.nb11);
    device const block_ptq1_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) ax[row] = (device const block_ptq1_0 *) (src0 + (uint64_t) min(first_row + row, args.M - 1)*args.nb01);
    float yl[nr1][17];
    float sumf[nr0][nr1] = {};
    const short ix = tiisg/8, it = tiisg%8;
    device const float * yb = y + ix*QK;
    { const float p[4] = {1,3,9,27}; FOR_UNROLL (short c = 0; c < nr1; ++c) yl[c][16] = p[it >> 1]; }
    for (int ib = ix; ib < nb; ib += 4) {
        float sumy[nr1] = {};
        FOR_UNROLL (short col = 0; col < nr1; ++col) {
            device const float * yc = (device const float *) ((device const char *) yb + col*args.nb11);
            FOR_UNROLL (short k = 0; k < 2; ++k) {
                const short m = 2*it + k; float v[5];
                FOR_UNROLL (short n = 0; n < 5; ++n) { v[n] = yc[n*16 + m]; sumy[col] += v[n]; }
                FOR_UNROLL (short n = 0; n < 4; ++n) yl[col][5*k + n] = v[n] - 3.0f*v[n+1];
                yl[col][5*k + 4] = v[4];
            }
            { float v[5];
              FOR_UNROLL (short n = 0; n < 5; ++n) { v[n] = yc[80 + n*8 + it]; sumy[col] += v[n]; }
              FOR_UNROLL (short n = 0; n < 4; ++n) yl[col][10 + n] = v[n] - 3.0f*v[n+1];
              yl[col][14] = v[4]; }
            { const float v = yc[120 + it]; yl[col][15] = v; sumy[col] += v; }
        }
        FOR_UNROLL (short row = 0; row < nr0; row++) dot_lut<nr1>(ax[row] + ib, T4, T1, yl, sumy, it, sumf[row]);
        yb += QK*4;
    }
    for (int row = 0; row < nr0; ++row) FOR_UNROLL (short col = 0; col < nr1; ++col) {
        const float tot = simd_sum(sumf[row][col]);
        if (tiisg == 0 && first_row + row < args.M) dst[(uint64_t) (r1 + col)*args.ne0 + first_row + row] = tot;
    }
}
#define INST(r,c,s,nm) template [[host_name(nm)]] kernel void lut<r,c,s>(constant kargs &, device const char *, device const char *, device float *, threadgroup char *, uint3, ushort, ushort);
INST(4,1,1,"lut_r4_c1_s1") INST(4,1,2,"lut_r4_c1_s2") INST(4,1,4,"lut_r4_c1_s4") INST(8,1,2,"lut_r8_c1_s2")
INST(4,2,1,"lut_r4_c2_s1") INST(4,2,2,"lut_r4_c2_s2") INST(4,2,4,"lut_r4_c2_s4")
INST(4,3,2,"lut_r4_c3_s2") INST(4,4,2,"lut_r4_c4_s2") INST(4,4,4,"lut_r4_c4_s4") INST(2,4,4,"lut_r2_c4_s4")
