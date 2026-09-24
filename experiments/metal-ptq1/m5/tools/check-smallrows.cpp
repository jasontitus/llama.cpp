// Accuracy of one small-row product against a double-accumulated reference of the exact stored weights.
// usage: check-smallrows <type: bf16|pq2_0|q1_0> ; prints NMSE per column count
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    const std::string tn = argc > 1 ? argv[1] : "bf16";
    const ggml_type type = tn == "q1_0" ? GGML_TYPE_Q1_0 : tn == "pq2_0" ? GGML_TYPE_PQ2_0 : GGML_TYPE_BF16;
    const int k = 5120, m = 48;
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) { std::fprintf(stderr, "no MTL0\n"); return 1; }
    std::mt19937 rng(20260923);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> w(k*m), deq(k*m);
    for (float & v : w) v = dist(rng) * 0.05f;
    std::vector<uint8_t> packed(ggml_row_size(type, k)*m);
    ggml_quantize_chunk(type, w.data(), packed.data(), 0, m, k, nullptr);
    ggml_get_type_traits(type)->to_float(packed.data(), deq.data(), k*m);
    for (int n : {9, 17, 64, 130}) {
        std::vector<float> x(k*n);
        for (float & v : x) v = dist(rng);
        ggml_init_params params = {ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * a = ggml_new_tensor_2d(ctx, type, k, m);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(a, packed.data(), 0, packed.size());
        ggml_backend_tensor_set(b, x.data(), 0, x.size()*4);
        ggml_backend_graph_compute(backend, g);
        std::vector<float> got(m*n);
        ggml_backend_tensor_get(out, got.data(), 0, got.size()*4);
        double err = 0, nrm = 0;
        for (int c = 0; c < n; ++c) for (int r = 0; r < m; ++r) {
            double ref = 0; for (int j = 0; j < k; ++j) ref += (double) deq[r*k+j] * x[c*k+j];
            const double d = got[c*m+r] - ref; err += d*d; nrm += ref*ref;
        }
        std::printf("%s m=%d n=%d nmse_vs_double=%.3e\n", tn.c_str(), m, n, err/nrm);
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
