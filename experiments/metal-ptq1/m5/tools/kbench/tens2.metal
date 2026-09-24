#include "common.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
// A in threadgroup (half), B in device memory (TB), 64 rows x 128 cols x 32 K per run, like mul_mm
template<typename TB>
inline void body(threadgroup char * shmem, device const char * s1, device float * dst, uint tid, ushort tiitg, uint3 tg) {
  threadgroup half * sa = (threadgroup half *) shmem;
  for (int i = tiitg; i < 64*32; i += 128) sa[i] = (half) (i & 3);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(32, 64));
  device TB * pb = (device TB *) s1;
  auto tB = tensor(pb, dextents<int32_t, 2>(32, 128), array<int, 2>({1, 4096}));
  matmul2d<matmul2d_descriptor(128, 64, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
  auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
  for (int i = 0; i < 64; ++i) { mm.run(tB, tA, cT); }
  auto tD = tensor(dst + (tg.x % 64) * 64 * 128, dextents<int32_t, 2>(64, 128), array<int, 2>({1, 64}));
  cT.store(tD);
}
kernel void mix_f(constant kargs & a, device const char * s0, device const char * s1, device float * dst, threadgroup char * shmem[[threadgroup(0)]], uint3 tid3[[thread_position_in_grid]], ushort tiitg[[thread_index_in_threadgroup]], uint3 tg[[threadgroup_position_in_grid]]) { body<float>(shmem, s1, dst, tid3.x, tiitg, tg); }
kernel void mix_h(constant kargs & a, device const char * s0, device const char * s1, device float * dst, threadgroup char * shmem[[threadgroup(0)]], uint3 tid3[[thread_position_in_grid]], ushort tiitg[[thread_index_in_threadgroup]], uint3 tg[[threadgroup_position_in_grid]]) { body<half>(shmem, s1, dst, tid3.x, tiitg, tg); }
// both operands in threadgroup memory: A 64x32 half, B 128x32 (TB)
template<typename TB>
inline void body_tg(threadgroup char * shmem, device float * dst, ushort tiitg, uint3 tg) {
  threadgroup half * sa = (threadgroup half *) shmem;
  threadgroup TB   * sb = (threadgroup TB *) (shmem + 64*32*2);
  for (int i = tiitg; i < 64*32; i += 128) sa[i] = (half) (i & 3);
  for (int i = tiitg; i < 128*32; i += 128) sb[i] = (TB) (i & 1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(32, 64));
  auto tB = tensor<threadgroup TB, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(32, 128));
  matmul2d<matmul2d_descriptor(128, 64, 32, false, true, true, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
  auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
  for (int i = 0; i < 64; ++i) { mm.run(tB, tA, cT); }
  auto tD = tensor(dst + (tg.x % 64) * 64 * 128, dextents<int32_t, 2>(64, 128), array<int, 2>({1, 64}));
  cT.store(tD);
}
kernel void tg_h(constant kargs & a, device const char * s0, device const char * s1, device float * dst, threadgroup char * shmem[[threadgroup(0)]], ushort tiitg[[thread_index_in_threadgroup]], uint3 tg[[threadgroup_position_in_grid]]) { body_tg<half>(shmem, dst, tiitg, tg); }
kernel void tg_f(constant kargs & a, device const char * s0, device const char * s1, device float * dst, threadgroup char * shmem[[threadgroup(0)]], ushort tiitg[[thread_index_in_threadgroup]], uint3 tg[[threadgroup_position_in_grid]]) { body_tg<float>(shmem, dst, tiitg, tg); }
