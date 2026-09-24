#include <metal_stdlib>
using namespace metal;
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define QK 128
struct block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; half d; };
struct kargs { int K, M, N, nb, nb01, nb11, ne0, pad; };
