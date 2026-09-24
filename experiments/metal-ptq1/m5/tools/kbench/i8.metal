#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// prefill-shaped tile: A 64 rows x NKB (threadgroup), B 128 columns x NKB (device), K step NKB.
// EPI: per-K-block epilogue acc += scale * blk (the int8 path's block scaling), else plain accumulate.
template<typename TA, typename TB, typename TC, int NKB, bool EPI>
inline void body(threadgroup char * shmem, device const char * s1, device float * dst, ushort tiitg, uint3 tg) {
  threadgroup TA * sa = (threadgroup TA *) shmem;
  for (int i = tiitg; i < 64*NKB; i += 128) sa[i] = (TA) ((i % 3) - 1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup TA, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NKB, 64));
  device TB * pb = (device TB *) s1;
  matmul2d<matmul2d_descriptor(128, 64, NKB, false, true, true, EPI ? matmul2d_descriptor::mode::multiply : matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
  auto tB0 = tensor(pb, dextents<int32_t, 2>(NKB, 128), array<int, 2>({1, 4096}));
  auto cT  = mm.template get_destination_cooperative_tensor<decltype(tB0), decltype(tA), TC>();
  auto acc = mm.template get_destination_cooperative_tensor<decltype(tB0), decltype(tA), float>();
  for (int i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  for (int kb = 0; kb < 4096/NKB; ++kb) {
    auto tB = tensor(pb + kb*NKB, dextents<int32_t, 2>(NKB, 128), array<int, 2>({1, 4096}));
    mm.run(tB, tA, cT);
    if (EPI) {
      const float sc = 1.0f + 0.001f*kb;
      for (int i = 0; i < cT.get_capacity(); ++i) acc[i] += sc * (float) cT[i];
    }
  }
  auto tD = tensor(dst + (tg.x % 64) * 64 * 128, dextents<int32_t, 2>(64, 128), array<int, 2>({1, 64}));
  if (EPI) acc.store(tD); else { for (int i = 0; i < cT.get_capacity(); ++i) acc[i] = (float) cT[i]; acc.store(tD); }
}
#define K(nm,TA,TB,TC,NKB,EPI) kernel void nm(constant kargs & a, device const char * s0, device const char * s1, device float * dst, threadgroup char * shmem[[threadgroup(0)]], ushort tiitg[[thread_index_in_threadgroup]], uint3 tg[[threadgroup_position_in_grid]]) { body<TA,TB,TC,NKB,EPI>(shmem, s1, dst, tiitg, tg); }
K(hf32,   half,   float,  float, 32,  false)
K(i8_128, int8_t, int8_t, int,   128, false)
K(i8_128e,int8_t, int8_t, int,   128, true)
K(i8_32e, int8_t, int8_t, int,   32,  true)
K(i8_64e, int8_t, int8_t, int,   64,  true)
