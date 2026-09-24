#include "common.h"
#include <metal_simdgroup_matrix>
template<typename T, typename TA>
inline void body(device float * dst, uint tid, ushort sg) {
  simdgroup_matrix<TA,8,8> a0(0), a1(0), a2(0), a3(0);
  simdgroup_matrix<T,8,8> x = make_filled_simdgroup_matrix<T,8,8>((T)0.5), y = make_filled_simdgroup_matrix<T,8,8>((T)0.25);
  for (int i = 0; i < 256; ++i) {
    simdgroup_multiply_accumulate(a0, x, y, a0); simdgroup_multiply_accumulate(a1, y, x, a1);
    simdgroup_multiply_accumulate(a2, x, x, a2); simdgroup_multiply_accumulate(a3, y, y, a3);
  }
  simdgroup_matrix<TA,8,8> s; simdgroup_multiply_accumulate(s, a0, a1, a2);
  simdgroup_multiply_accumulate(s, s, a3, s);
  simdgroup_store(s, (device TA*) dst + ((tid/32) % 1024)*64, 8);
}
kernel void mm_f32(constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]], ushort sg[[simdgroup_index_in_threadgroup]]) { body<float,float>(dst, tid, sg); }
kernel void mm_f16(constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]], ushort sg[[simdgroup_index_in_threadgroup]]) { body<half,float>(dst, tid, sg); }
kernel void mm_f16h(constant kargs & args, device const char * s0, device const char * s1, device float * dst, uint tid[[thread_position_in_grid]], ushort sg[[simdgroup_index_in_threadgroup]]) { body<half,half>(dst, tid, sg); }
