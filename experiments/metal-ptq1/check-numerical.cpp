#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Compare identical columns across batch sizes and against a double-accumulated
// dot of CPU-dequantized weights. Include row tails and padded activation strides.
static bool check(ggml_backend_t backend, int k, int m, bool padded, int pattern) {
    std::mt19937 rng(1234 + k + m + pattern);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> weights(k*m), activations(k*8), dequant(k*m);
    for (float & v : weights) { v = dist(rng); }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = pattern == 1 ? 0.f : pattern == 2 ? (i%2 ? -1.f : 1.f) : dist(rng);
    }
    std::vector<uint8_t> packed(ggml_row_size(GGML_TYPE_PTQ1_0, k)*m);
    ggml_quantize_chunk(GGML_TYPE_PTQ1_0, weights.data(), packed.data(), 0, m, k, nullptr);
    ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float(packed.data(), dequant.data(), k*m);
    std::vector<double> reference(m*8);
    for (int c = 0; c < 8; ++c) {
        for (int r = 0; r < m; ++r) {
            for (int j = 0; j < k; ++j) {
                reference[c*m+r] += double(dequant[r*k+j])*activations[c*k+j];
            }
        }
    }
    std::vector<float> single(m*8);
    double worst_ref = 0., worst_batch = 0., worst_abs = 0.;
    for (int n : {1, 2, 3, 4, 8}) {
        const int repeats = n == 1 ? 8 : 1;
        double err_ref = 0., err_batch = 0., norm = 0.;
        for (int rep = 0; rep < repeats; ++rep) {
            ggml_init_params params = {ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true};
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_PTQ1_0, k, m);
            const int stride = k + (padded ? 32 : 0);
            ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, stride, n);
            ggml_tensor * b = ggml_view_2d(ctx, storage, k, n, stride*sizeof(float), 0);
            ggml_tensor * out = ggml_mul_mat(ctx, a, b);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);
            ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) { std::fprintf(stderr, "allocation failed\n"); std::exit(1); }
            ggml_backend_tensor_set(a, packed.data(), 0, packed.size());
            for (int c = 0; c < n; ++c) {
                ggml_backend_tensor_set(storage, activations.data() + (c+rep)*k, c*stride*sizeof(float), k*sizeof(float));
            }
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { std::exit(1); }
            std::vector<float> got(m*n);
            ggml_backend_tensor_get(out, got.data(), 0, got.size()*sizeof(float));
            for (size_t i = 0; i < got.size(); ++i) {
                if (!std::isfinite(got[i])) { std::fprintf(stderr, "nonfinite output\n"); std::exit(1); }
                const double ref = reference[rep*m+i];
                err_ref += (got[i]-ref)*(got[i]-ref);
                norm += ref*ref;
                worst_abs = std::max(worst_abs, std::abs(got[i]-ref));
                if (n == 1) { single[rep*m+i] = got[i]; }
                else { err_batch += double(got[i]-single[i])*(got[i]-single[i]); }
            }
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
        worst_ref = std::max(worst_ref, err_ref/std::max(norm, 1e-30));
        worst_batch = std::max(worst_batch, err_batch/std::max(norm, 1e-30));
    }
    const bool ok = worst_ref < 1e-8 && worst_batch < 1e-8;
    std::printf("k=%d m=%d padded=%d pattern=%d ref_nmse=%.3g batch_nmse=%.3g max_abs=%.3g %s\n",
                k, m, padded, pattern, worst_ref, worst_batch, worst_abs, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) { std::fprintf(stderr, "MTL0 unavailable; refusing empty test pass\n"); return 1; }
    bool ok = true;
    for (int k : {128, 384, 512, 5120, 17408}) {
        for (int m : {1, 3, 7, 8}) {
            for (bool padded : {false, true}) { ok &= check(backend, k, m, padded, 0); }
        }
    }
    for (int pattern : {1, 2}) { ok &= check(backend, 5120, 7, true, pattern); }
    ggml_backend_free(backend);
    ggml_quantize_free();
    return ok ? 0 : 1;
}
