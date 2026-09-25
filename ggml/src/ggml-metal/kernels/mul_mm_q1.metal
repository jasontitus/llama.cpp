// Research prefill kernels in a library of their own: the device builds it at startup like the others, but a
// build failure only disables these kernels (the host then keeps the generic mul_mm), not the Metal backend.
#include "common.h"
#include "dequantize.h"

constant bool  FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool  FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12   [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13   [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2     [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3     [[function_constant(FC_MUL_MM + 5)]];

#ifdef GGML_METAL_HAS_TENSOR
// Q1_0 prefill for products made only of full tiles (research flags GGML_METAL_Q1_MM_K32_ALIGNED /
// GGML_METAL_Q1_SWIZZLE_LOG, ported from the historical Bonsai 1 work: -4.8% cold prefill on M5).
// The tensor kernel_mul_mm (mul_mm.metal) with a static K32 extent and no bounds handling: the host sends a
// whole product here only when M % 64, N % 128 and K % 32 are all zero, otherwise the whole product stays on
// the generic kernel. The same threads dequantize the same 16-weight chunks in the same K32 order with the
// same relaxed-precision matmul2d. The differences are the static K in the matmul2d descriptor and the
// operand/store extents (32 x NRB and a whole-tile store instead of the clamped views and a sliced store);
// their lowering is the device compiler's, so bitwise equality with the generic kernel is a measured property
// per device and OS (tools/check-q1-mm.cpp), not a guarantee.
// SWIZZLE_LOG > 0 remaps the grid so that 2^SWIZZLE_LOG adjacent row tiles run next to each other on the
// same activation columns.
template<int SWIZZLE_LOG>
kernel void kernel_mul_mm_q1_0_f32_k32(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
    constexpr int NK  = N_MM_NK_TOTAL; // 32
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;
    static_assert(NUM_THREADS == NRA * N_MM_NK, "one 16-weight chunk per thread");

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    const int im  = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // grouped grid: x = column tile * 2^L + row within the group, y = row group
    const int ra = ((int(tgpig.y) << SWIZZLE_LOG) + (int(tgpig.x) & ((1 << SWIZZLE_LOG) - 1))) * NRA;
    const int rb = (int(tgpig.x) >> SWIZZLE_LOG) * NRB;

    // the generic kernel's work mapping: work = tiitg, row = work / N_MM_NK, chunk = work % N_MM_NK
    const int   row     = tiitg / N_MM_NK;
    const short k_base  = (tiitg % N_MM_NK) * 16;

    threadgroup half * sa = (threadgroup half *) shmem;

    device const block_q1_0 * row_ptr = (device const block_q1_0 *)(srcA + args.nb01 * (ra + row) + offset0);
    device float * ptrB = (device float *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(float);

    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NRA));
    auto tB = tensor(ptrB + rb * strideB, dextents<int32_t, 2>(NK, NRB), array<int, 2>({1, strideB}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, NK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        const int k_pos = loop_k + k_base;

        half4x4 temp_a;
        dequantize_q1_0(row_ptr + k_pos / QK1_0, (k_pos / 16) % (QK1_0 / 16), temp_a);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            sa[row * NK + k_base + i] = temp_a[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(NK, NRB), array<int, 2>({1, strideB}));

        mm.run(tBv, tA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;

    auto tD = tensor(dstTile, dextents<int32_t, 2>(NRA, NRB), array<int, 2>({1, M}));
    cT.store(tD);
}

template [[host_name("kernel_mul_mm_q1_0_f32_k32")]]          kernel void kernel_mul_mm_q1_0_f32_k32<0>(constant ggml_metal_kargs_mul_mm &, device const char *, device const char *, device char *, threadgroup char *, uint3, ushort, ushort);
template [[host_name("kernel_mul_mm_q1_0_f32_k32_swizzle1")]] kernel void kernel_mul_mm_q1_0_f32_k32<1>(constant ggml_metal_kargs_mul_mm &, device const char *, device const char *, device char *, threadgroup char *, uint3, ushort, ushort);
template [[host_name("kernel_mul_mm_q1_0_f32_k32_swizzle2")]] kernel void kernel_mul_mm_q1_0_f32_k32<2>(constant ggml_metal_kargs_mul_mm &, device const char *, device const char *, device char *, threadgroup char *, uint3, ushort, ushort);
template [[host_name("kernel_mul_mm_q1_0_f32_k32_swizzle3")]] kernel void kernel_mul_mm_q1_0_f32_k32<3>(constant ggml_metal_kargs_mul_mm &, device const char *, device const char *, device char *, threadgroup char *, uint3, ushort, ushort);
#endif
