#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
template<typename TI, typename TO, int NR0, int NR1, int NK>
inline void body(threadgroup char * shmem, device float * dst, uint tid, ushort tiitg) {
  threadgroup TI * sa = (threadgroup TI *) shmem;
  threadgroup TI * sb = (threadgroup TI *) (shmem + NR0*NK*sizeof(TI));
  for (int i = tiitg; i < NR0*NK; i += 128) sa[i] = (TI) (i & 3);
  for (int i = tiitg; i < NR1*NK; i += 128) sb[i] = (TI) (i & 1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup TI, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
  auto tB = tensor<threadgroup TI, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK));
  matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
  auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), TO>();
  for (int i = 0; i < 64; ++i) { mm.run(tB, tA, cT); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tC = tensor<threadgroup TO, dextents<int32_t, 2>, tensor_inline>((threadgroup TO *) shmem, dextents<int32_t, 2>(NR0, NR1));
  cT.store(tC);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tiitg < 64) dst[(tid/128 % 4096)*64 + tiitg] = (float) ((threadgroup TO *) shmem)[tiitg];
}
#define K(nm,TI,TO,a,b,c) kernel void nm(constant kargs & args, device const char * s0, device const char * s1, device float * dst, threadgroup char * shmem[[threadgroup(0)]], uint tid[[thread_position_in_grid]], ushort tiitg[[thread_index_in_threadgroup]]) { body<TI,TO,a,b,c>(shmem, dst, tid, tiitg); }
K(t_h_64x32x32, half, float, 64, 32, 32)
K(t_h_64x8x64, half, float, 64, 8, 64)
K(t_h_128x8x64, half, float, 128, 8, 64)
K(t_f_64x32x32, float, float, 64, 32, 32)
K(t_f_64x8x64, float, float, 64, 8, 64)
K(t_i8_64x32x32, int8_t, int, 64, 32, 32)
K(t_i8_64x8x64, int8_t, int, 64, 8, 64)
