#include "common.h"
// ---- verbatim logic of ggml's kernel_mul_mv_ptq1_0_multicol<nr0,nr1> (nr1=1 == single-vector arithmetic) ----
template<int nr1, bool DECODE>
inline void dot_mc(device const block_ptq1_0 * qb, thread const float (&yl)[nr1][17], thread const float (&sumy)[nr1], short it, thread float (&sumf)[nr1]) {
    float acc[nr1] = {};
    FOR_UNROLL (short byte = 0; byte < 3; ++byte) {
        const float u = (float) qb->qs[byte < 2 ? 2*it + byte : 16 + it] * (1.0f/256.0f);
        float g[5];
        if (DECODE) { g[0]=floor(3.0f*u); g[1]=floor(9.0f*u); g[2]=floor(27.0f*u); g[3]=floor(81.0f*u); g[4]=floor(243.0f*u); }
        else { g[0]=u; g[1]=u; g[2]=u; g[3]=u; g[4]=u; }
        FOR_UNROLL (short col = 0; col < nr1; ++col) FOR_UNROLL (short n = 0; n < 5; ++n) acc[col] += g[n] * yl[col][5*byte + n];
    }
    const float u = (float) qb->qh[it & 1] * (1.0f/256.0f);
    const float p0 = yl[0][16];
    const float t = DECODE ? floor(3.0f*p0*u) - 3.0f*floor(p0*u) : u;
    const float d = (float) qb->d;
    FOR_UNROLL (short col = 0; col < nr1; ++col) { acc[col] += t * yl[col][15]; sumf[col] += (acc[col] - sumy[col]) * d; }
}
template<int nr0, int nr1, int NSG, bool DECODE>
kernel void mc(constant kargs & args, device const char * src0, device const char * src1, device float * dst,
               uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
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
        FOR_UNROLL (short row = 0; row < nr0; row++) dot_mc<nr1, DECODE>(ax[row] + ib, yl, sumy, it, sumf[row]);
        yb += QK*4;
    }
    for (int row = 0; row < nr0; ++row) FOR_UNROLL (short col = 0; col < nr1; ++col) {
        const float tot = simd_sum(sumf[row][col]);
        if (tiisg == 0 && first_row + row < args.M) dst[(uint64_t) (r1 + col)*args.ne0 + first_row + row] = tot;
    }
}
#define INST(r,c,s,dec,nm) template [[host_name(nm)]] kernel void mc<r,c,s,dec>(constant kargs &, device const char *, device const char *, device float *, uint3, ushort, ushort);
INST(4,1,1,true,"base_r4_c1") INST(5,1,1,true,"base_r5_c1") INST(8,1,1,true,"base_r8_c1") INST(2,1,1,true,"base_r2_c1")
INST(4,1,1,false,"nodec_r4_c1")
INST(4,2,1,true,"base_r4_c2") INST(4,3,1,true,"base_r4_c3") INST(4,4,1,true,"base_r4_c4")
INST(4,4,1,false,"nodec_r4_c4")
