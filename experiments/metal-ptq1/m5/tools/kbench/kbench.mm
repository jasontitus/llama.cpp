// Standalone PTQ1_0 matvec kernel bench: random blocks, CPU double reference, GPU-timestamp timing.
// usage: kbench file.metal kernel K M N rows_per_tg threads_per_tg cols_per_tg [iters] [tgmem_bytes]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
struct blk { uint8_t qs[24]; uint8_t qh[2]; uint16_t d; };
static_assert(sizeof(blk) == 28, "blk");
struct args { int K, M, N, nb, nb01, nb11, ne0, pad; };
static float h2f(uint16_t h) { __fp16 v; memcpy(&v, &h, 2); return (float) v; }
static int trit(int b, int n) { int g1 = (b * (int) pow(3, n + 1)) >> 8, g0 = (b * (int) pow(3, n)) >> 8; return g1 - 3 * g0; }
int main(int argc, char ** argv) {
  @autoreleasepool {
    if (argc < 9) { fprintf(stderr, "usage\n"); return 1; }
    NSString * path = [NSString stringWithUTF8String:argv[1]];
    const char * kname = argv[2];
    int K = atoi(argv[3]), M = atoi(argv[4]), N = atoi(argv[5]);
    int rows_tg = atoi(argv[6]), thr_tg = atoi(argv[7]), cols_tg = atoi(argv[8]);
    int iters = argc > 9 ? atoi(argv[9]) : 200;
    int tgmem = argc > 10 ? atoi(argv[10]) : 0;
    int copies = getenv("KB_COPIES") ? atoi(getenv("KB_COPIES")) : 16;
    int nb = K / 128;
    std::mt19937 rng(20260923);
    std::vector<blk> W((size_t) M * nb);
    std::uniform_int_distribution<int> ub(0, 255);
    std::uniform_real_distribution<float> ud(0.01f, 0.05f), uy(-1.f, 1.f);
    for (auto & b : W) {
      // valid encodings only: V in 0..242 -> b = ceil(V*256/243); qh V in 0..80 -> ceil(V*256/81)
      for (int i = 0; i < 24; i++) { int V = ub(rng) % 243; b.qs[i] = (uint8_t) ((V * 256 + 242) / 243); }
      for (int i = 0; i < 2; i++) { int V = ub(rng) % 81; b.qh[i] = (uint8_t) ((V * 256 + 80) / 81); }
      __fp16 d = (__fp16) ud(rng); memcpy(&b.d, &d, 2);
    }
    std::vector<float> Y((size_t) N * K);
    for (auto & y : Y) y = uy(rng);
    // reference
    std::vector<double> R((size_t) N * M);
    std::vector<float> wrow(K);
    for (int r = 0; r < M; r++) {
      for (int ib = 0; ib < nb; ib++) {
        const blk & b = W[(size_t) r * nb + ib];
        float d = h2f(b.d); float * w = &wrow[ib * 128];
        for (int j = 0; j < 16; j++) for (int n = 0; n < 5; n++) w[n * 16 + j] = (trit(b.qs[j], n) - 1) * d;
        for (int j = 0; j < 8; j++) for (int n = 0; n < 5; n++) w[80 + n * 8 + j] = (trit(b.qs[16 + j], n) - 1) * d;
        for (int h = 0; h < 2; h++) for (int n = 0; n < 4; n++) w[120 + n * 2 + h] = (trit(b.qh[h], n) - 1) * d;
      }
      for (int c = 0; c < N; c++) { double s = 0; const float * y = &Y[(size_t) c * K]; for (int k = 0; k < K; k++) s += (double) wrow[k] * y[k]; R[(size_t) c * M + r] = s; }
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError * err = nil;
    NSString * src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
    MTLCompileOptions * opt = [MTLCompileOptions new];
    opt.mathMode = MTLMathModeFast;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", [[err description] UTF8String]); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kname]];
    if (!fn) { fprintf(stderr, "no kernel %s\n", kname); return 1; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    NSMutableArray * bWs = [NSMutableArray new];
    for (int c = 0; c < copies; c++) [bWs addObject:[dev newBufferWithBytes:W.data() length:W.size() * sizeof(blk) options:MTLResourceStorageModeShared]];
    id<MTLBuffer> bY = [dev newBufferWithBytes:Y.data() length:Y.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bD = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModeShared];
    // KB_PRESTAGE: per (column, block, lane it) 20 floats = 16 collapse coefficients, sumy, pad x3
    std::vector<float> P((size_t) N * nb * 8 * 20, 0.f);
    for (int c = 0; c < N; c++) for (int ib = 0; ib < nb; ib++) for (int it = 0; it < 8; it++) {
      const float * yb = &Y[(size_t) c * K + ib * 128]; float * o = &P[(((size_t) c * nb + ib) * 8 + it) * 20]; float sumy = 0;
      for (int k = 0; k < 2; k++) { int m = 2 * it + k; float v[5]; for (int n = 0; n < 5; n++) { v[n] = yb[n * 16 + m]; sumy += v[n]; }
        for (int n = 0; n < 4; n++) o[5 * k + n] = fmaf(-3.0f, v[n + 1], v[n]); o[5 * k + 4] = v[4]; }
      { float v[5]; for (int n = 0; n < 5; n++) { v[n] = yb[80 + n * 8 + it]; sumy += v[n]; } for (int n = 0; n < 4; n++) o[10 + n] = fmaf(-3.0f, v[n + 1], v[n]); o[14] = v[4]; }
      o[15] = yb[120 + it]; sumy += o[15]; o[16] = sumy;
    }
    // KB_HILO=<padded tensor columns>: buffer 4 holds half hi/lo activations, column 2c = hi, 2c+1 = lo, K contiguous
    if (getenv("KB_HILO")) {
      int NP = atoi(getenv("KB_HILO"));
      std::vector<__fp16> H((size_t) NP * K, (__fp16) 0.f);
      for (int c = 0; c < N; c++) for (int k = 0; k < K; k++) {
        float y = Y[(size_t) c * K + k]; __fp16 hi = (__fp16) y; __fp16 lo = (__fp16) (y - (float) hi);
        H[(size_t) (2 * c) * K + k] = hi; H[(size_t) (2 * c + 1) * K + k] = lo;
      }
      P.assign(H.size() / 2 + 1, 0.f); memcpy(P.data(), H.data(), H.size() * 2);
    }
    // KB_HILO2=<padded tensor columns>: collapse coefficients in trit-major order k' = (k-1)*24 + j (qs bytes)
    // and 120 + 2(k-1) + h (qh bytes), split hi/lo like KB_HILO; buffer 5 = per (column, block) sum of y
    std::vector<float> S((size_t) N * nb + 1, 0.f);
    if (getenv("KB_HILO2")) {
      int NP = atoi(getenv("KB_HILO2"));
      std::vector<__fp16> H((size_t) NP * K, (__fp16) 0.f);
      for (int c = 0; c < N; c++) for (int ib = 0; ib < nb; ib++) {
        const float * yb = &Y[(size_t) c * K + ib * 128]; float cc[128]; float sy = 0;
        for (int i = 0; i < 128; i++) sy += yb[i];
        for (int j = 0; j < 24; j++) {
          int e[5]; for (int n = 0; n < 5; n++) e[n] = j < 16 ? n * 16 + j : 80 + n * 8 + (j - 16);
          for (int k = 1; k <= 4; k++) cc[(k - 1) * 24 + j] = fmaf(-3.0f, yb[e[k]], yb[e[k - 1]]);
          cc[4 * 24 + j] = yb[e[4]];
        }
        for (int h = 0; h < 2; h++) {
          int e[4]; for (int n = 0; n < 4; n++) e[n] = 120 + n * 2 + h;
          for (int k = 1; k <= 3; k++) cc[120 + 2 * (k - 1) + h] = fmaf(-3.0f, yb[e[k]], yb[e[k - 1]]);
          cc[120 + 2 * 3 + h] = yb[e[3]];
        }
        for (int i = 0; i < 128; i++) { __fp16 hi = (__fp16) cc[i]; H[(size_t) (2 * c) * K + ib * 128 + i] = hi; H[(size_t) (2 * c + 1) * K + ib * 128 + i] = (__fp16) (cc[i] - (float) hi); }
        S[(size_t) c * nb + ib] = sy;
      }
      P.assign(H.size() / 2 + 1, 0.f); memcpy(P.data(), H.data(), H.size() * 2);
    }
    id<MTLBuffer> bS = [dev newBufferWithBytes:S.data() length:S.size() * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bP = [dev newBufferWithBytes:P.data() length:P.size() * 4 options:MTLResourceStorageModeShared];
    args a = {K, M, N, nb, (int) (nb * sizeof(blk)), K * 4, M, 0};
    id<MTLCommandQueue> q = [dev newCommandQueue];
    MTLSize grid = MTLSizeMake((M + rows_tg - 1) / rows_tg, (N + cols_tg - 1) / cols_tg, 1), tg = MTLSizeMake(thr_tg, 1, 1);
    auto run = [&](int reps) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:ps];
      [e setBytes:&a length:sizeof(a) atIndex:0];
      [e setBuffer:bY offset:0 atIndex:2]; [e setBuffer:bD offset:0 atIndex:3]; [e setBuffer:bP offset:0 atIndex:4]; [e setBuffer:bS offset:0 atIndex:5];
      if (tgmem) [e setThreadgroupMemoryLength:tgmem atIndex:0];
      for (int i = 0; i < reps; i++) { [e setBuffer:bWs[i % copies] offset:0 atIndex:1]; [e dispatchThreadgroups:grid threadsPerThreadgroup:tg]; }
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { fprintf(stderr, "gpu error %s\n", [[cb.error description] UTF8String]); exit(1); }
      return (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / reps;
    };
    memset(bD.contents, 0xff, (size_t) N * M * 4);
    run(1);
    const float * D = (const float *) bD.contents;
    double se = 0, sr = 0, maxa = 0; bool finite = true;
    for (size_t i = 0; i < (size_t) N * M; i++) { if (!std::isfinite(D[i])) finite = false; double e = D[i] - R[i]; se += e * e; sr += R[i] * R[i]; maxa = std::max(maxa, fabs(e)); }
    run(20);
    std::vector<double> t; for (int i = 0; i < 5; i++) t.push_back(run(iters));
    std::sort(t.begin(), t.end());
    double us = t[2];
    double gbs = (W.size() * sizeof(blk) + Y.size() * 4.0) / (us * 1e3);
    printf("%-40s K=%5d M=%6d N=%d  %8.2f us  %6.1f GB/s  nmse=%.2e maxabs=%.2e %s regs? maxthr=%lu\n", kname, K, M, N, us, gbs, se / sr, maxa,
           (finite && se / sr < 1e-8) ? "OK" : "FAIL", (unsigned long) ps.maxTotalThreadsPerThreadgroup);
  }
  return 0;
}
