#include "common.h"
#define BODY(T, INIT, OP) \
  T a0 = INIT(tid), a1 = INIT(tid+1), a2 = INIT(tid+2), a3 = INIT(tid+3), a4 = INIT(tid+4), a5 = INIT(tid+5), a6 = INIT(tid+6), a7 = INIT(tid+7); \
  for (int i = 0; i < 256; ++i) { OP(a0); OP(a1); OP(a2); OP(a3); OP(a4); OP(a5); OP(a6); OP(a7); } \
  dst[tid] = (float) (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);
#define FINIT(x) (float)(x) * 1e-7f
#define HINIT(x) (half)((x) & 7) * 0.01h
#define H2INIT(x) half2((half)((x)&7)*0.01h, 0.02h)
#define IINIT(x) (uint)(x)
#define S2INIT(x) ushort2((ushort)(x), (ushort)(x+1))
#define FMA_F(a) a = fma(a, 0.999f, 0.001f)
#define FMA_H(a) a = fma(a, 0.999h, 0.001h)
#define FLOOR_F(a) a = floor(a * 1.5f) * 0.66f
#define IMUL(a) a = a * 3u + 1u
#define IMAD16(a) a = a * ushort2(3) + ushort2(1)
kernel void alu_f32(constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]]) { BODY(float, FINIT, FMA_F) }
kernel void alu_f16(constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]]) { BODY(half, HINIT, FMA_H) }
kernel void alu_h2 (constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]]) {
  half2 a0 = H2INIT(tid), a1 = H2INIT(tid+1), a2 = H2INIT(tid+2), a3 = H2INIT(tid+3), a4 = H2INIT(tid+4), a5 = H2INIT(tid+5), a6 = H2INIT(tid+6), a7 = H2INIT(tid+7);
  for (int i = 0; i < 256; ++i) { FMA_H(a0); FMA_H(a1); FMA_H(a2); FMA_H(a3); FMA_H(a4); FMA_H(a5); FMA_H(a6); FMA_H(a7); }
  half2 s = a0+a1+a2+a3+a4+a5+a6+a7; dst[tid] = (float) s.x + (float) s.y; }
kernel void alu_floor(constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]]) { BODY(float, FINIT, FLOOR_F) }
kernel void alu_imul(constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]]) { BODY(uint, IINIT, IMUL) }
kernel void alu_s2(constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]]) {
  ushort2 a0 = S2INIT(tid), a1 = S2INIT(tid+1), a2 = S2INIT(tid+2), a3 = S2INIT(tid+3), a4 = S2INIT(tid+4), a5 = S2INIT(tid+5), a6 = S2INIT(tid+6), a7 = S2INIT(tid+7);
  for (int i = 0; i < 256; ++i) { IMAD16(a0); IMAD16(a1); IMAD16(a2); IMAD16(a3); IMAD16(a4); IMAD16(a5); IMAD16(a6); IMAD16(a7); }
  ushort2 s = a0+a1+a2+a3+a4+a5+a6+a7; dst[tid] = (float) s.x + (float) s.y; }
