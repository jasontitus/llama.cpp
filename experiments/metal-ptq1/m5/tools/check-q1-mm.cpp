// Q1_0 prefill research kernels (GGML_METAL_Q1_MM_K32_ALIGNED, GGML_METAL_Q1_SWIZZLE_LOG=1..3): bitwise
// equality with the generic tensor mul_mm kernel, and proof of which kernel produced each result.
//
// Every (shape, flag setting) runs y = W x in its own child process. Metal pipelines are cached for the life of
// a process, so a fresh process creates exactly the mul_mm pipeline its product uses, and the child reports that
// kernel from the library's "loaded" log line (logged only for a pipeline that exists). Required: flags off -> the generic kernel; eligible shapes (M % 64 == 0,
// N % 128 == 0, contiguous; batched and broadcast included) -> the K32 kernel (the swizzled one when M/64 is a
// multiple of 2^log); every other shape -> the generic kernel. Every output bit must match the flags-off
// result, and a guard buffer allocated after the output must be untouched. Also checks the flags-off result
// against a double-accumulated CPU reference on sampled rows (NMSE < 1e-5; ~1e-7 is expected from the half
// operands). Bitwise equality is a property of the device compiler, so it holds for the device and OS this runs
// on, not in general. The weights are random Q1_0 blocks (random bits, scales 0.01-0.05).
//
// build (repository root, Metal build in build/):
//   clang++ -std=c++17 -O2 -Iggml/include experiments/metal-ptq1/m5/tools/check-q1-mm.cpp \
//     -Lbuild/bin -lggml -lggml-base -Wl,-rpath,build/bin -o check-q1-mm
// run: ./check-q1-mm   (the --child mode is internal)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

// src0 is [k, m, b0, c0], src1 is [k, n, b1, c1] with b1 % b0 == 0 and c1 % c0 == 0 (smaller W dims broadcast)
struct Shape { int k, m, n, b0, b1; const char * what; int c0 = 1, c1 = 1; };

static const Shape k_shapes[] = {
    {5120, 17408, 512, 1, 1, "FFN up/gate at a 512-token ubatch"},
    {17408, 5120, 512, 1, 1, "FFN down"},
    {5120, 6144, 256, 1, 1, ""},
    {5120, 1024, 128, 1, 1, "one column tile"},
    {5120,  192, 512, 1, 1, "M/64 = 3: K32, never swizzled"},
    {6144, 5120, 512, 1, 1, "attn_output / ssm_out (K = 6144)"},
    {5120, 10240, 512, 1, 1, "attn_qkv"},
    {5120, 12288, 256, 1, 1, "attn_q"},
    {5120, 248320, 128, 1, 1, "output head, all logits"},
    {5120,  512, 256, 2, 2, "batched"},
    {5120,  512, 256, 1, 3, "broadcast W over 3 batches"},
    {4096,  256, 128, 2, 4, "batched and broadcast"},
    {4096,  256, 128, 2, 2, "4D: broadcast W over dim 3", 1, 3},
    {5120,   48, 512, 1, 1, "M % 64 != 0: generic"},
    {5120, 1024, 513, 1, 1, "N % 128 != 0: generic"},
    {5120, 1024,  70, 1, 1, "N < 128: generic"},
    {5120,  512, 200, 1, 3, "broadcast, N % 128 != 0: generic"},
};

struct Config { const char * name; const char * env; const char * value; int log; };

static const Config k_configs[] = {
    {"flags off", nullptr,                        nullptr, -1},
    {"K32",       "GGML_METAL_Q1_MM_K32_ALIGNED", "1",      0},
    {"swizzle 1", "GGML_METAL_Q1_SWIZZLE_LOG",    "1",      1},
    {"swizzle 2", "GGML_METAL_Q1_SWIZZLE_LOG",    "2",      2},
    {"swizzle 3", "GGML_METAL_Q1_SWIZZLE_LOG",    "3",      3},
};

static std::set<std::string> g_compiled; // mul_mm kernels this process created pipelines for

static void log_cb(ggml_log_level level, const char * text, void *) {
    // "loaded <name>" is logged only once a pipeline exists ("compiling pipeline" also precedes failures)
    const char * p = std::strstr(text, "loaded kernel_mul_mm_");
    if (p) {
        p += std::strlen("loaded ");
        const char * e = std::strstr(p, "_bci=");
        if (e) { g_compiled.insert(std::string(p, e)); }
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) { std::fputs(text, stderr); }
}

static int n_w(const Shape & s) { return s.b0*s.c0; }   // weight matrices
static int n_x(const Shape & s) { return s.b1*s.c1; }   // activation (and output) matrices

static void make_inputs(const Shape & s, std::vector<uint8_t> & packed, std::vector<float> & x) {
    std::mt19937 rng(20260924 + s.k + s.m + s.n + 7*s.b0 + 13*s.b1 + 17*s.c1);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    x.resize((size_t) s.k*s.n*n_x(s));
    for (float & v : x) { v = dist(rng); }
    const size_t block = ggml_type_size(GGML_TYPE_Q1_0);   // fp16 scale, then 128 sign bits
    packed.resize(ggml_row_size(GGML_TYPE_Q1_0, s.k)*s.m*n_w(s));
    for (size_t off = 0; off < packed.size(); off += block) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.03f + 0.02f*dist(rng));
        std::memcpy(&packed[off], &d, sizeof(d));
        for (size_t i = sizeof(d); i < block; i += 4) {
            const uint32_t r = rng();
            std::memcpy(&packed[off + i], &r, std::min<size_t>(4, block - i));
        }
    }
}

static bool eligible(const Shape & s) { return s.m % 64 == 0 && s.n % 128 == 0 && s.k % 32 == 0; }

static std::string wanted_kernel(const Shape & s, const Config & c) {
    if (c.log < 0 || !eligible(s)) { return "kernel_mul_mm_q1_0_f32"; }
    if (c.log > 0 && (s.m/64) % (1 << c.log) == 0) { return "kernel_mul_mm_q1_0_f32_k32_swizzle" + std::to_string(c.log); }
    return "kernel_mul_mm_q1_0_f32_k32";
}

// child: run one product with one flag setting, write the output to `out`, print the compiled mul_mm kernels
static int child(int si, int ci, const char * out) {
    const Shape & s = k_shapes[si];
    const Config & c = k_configs[ci];
    for (const Config & o : k_configs) { if (o.env) { unsetenv(o.env); } }
    if (c.env) { setenv(c.env, c.value, 1); }

    ggml_log_set(log_cb, nullptr);
    ggml_backend_load_all();

    std::vector<uint8_t> packed;
    std::vector<float> x;
    make_inputs(s, packed, x);

    ggml_backend_t backend = ggml_backend_init_by_name("MTL0", nullptr);
    if (!backend) { std::fprintf(stderr, "MTL0 unavailable; refusing an empty pass\n"); return 1; }
    ggml_init_params ip = { 4*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_Q1_0, s.k, s.m, s.b0, s.c0);
    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.k, s.n, s.b1, s.c1);
    ggml_tensor * y = ggml_mul_mat(ctx, w, a);
    ggml_tensor * guard = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16384);   // allocated right after y
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(a, x.data(), 0, x.size()*sizeof(float));
    const std::vector<float> guard_values(ggml_nelements(guard), 1234.5f);
    ggml_backend_tensor_set(guard, guard_values.data(), 0, ggml_nbytes(guard));
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed\n"); return 1; }
    std::vector<float> res(ggml_nelements(y)), guard_after(ggml_nelements(guard));
    ggml_backend_tensor_get(y, res.data(), 0, res.size()*sizeof(float));
    ggml_backend_tensor_get(guard, guard_after.data(), 0, ggml_nbytes(guard));
    if (guard_after != guard_values) { std::fprintf(stderr, "the guard after the output was overwritten\n"); return 1; }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    FILE * f = std::fopen(out, "wb");
    if (!f || std::fwrite(res.data(), sizeof(float), res.size(), f) != res.size()) { std::fprintf(stderr, "cannot write %s\n", out); return 1; }
    std::fclose(f);
    for (const auto & n : g_compiled) { std::printf("KERNEL %s\n", n.c_str()); }
    return 0;
}

static bool run_child(const char * self, int si, int ci, const std::string & out, std::vector<float> & res,
                      std::set<std::string> & kernels) {
    const std::string cmd = "'" + std::string(self) + "' --child " + std::to_string(si) + " " + std::to_string(ci) +
                            " '" + out + "'";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) { return false; }
    char line[512];
    while (std::fgets(line, sizeof(line), p)) {
        if (std::strncmp(line, "KERNEL ", 7) == 0) {
            std::string n(line + 7);
            while (!n.empty() && (n.back() == '\n' || n.back() == '\r')) { n.pop_back(); }
            kernels.insert(n);
        }
    }
    if (pclose(p) != 0) { return false; }
    const Shape & s = k_shapes[si];
    res.assign((size_t) s.m*s.n*n_x(s), 0.f);
    FILE * f = std::fopen(out.c_str(), "rb");
    const bool ok = f && std::fread(res.data(), sizeof(float), res.size(), f) == res.size();
    if (f) { std::fclose(f); }
    std::remove(out.c_str());
    return ok;
}

int main(int argc, char ** argv) {
    if (argc == 5 && std::strcmp(argv[1], "--child") == 0) {
        return child(std::atoi(argv[2]), std::atoi(argv[3]), argv[4]);
    }

    const char * tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/check-q1-mm.XXXXXX";
    const char * dir = mkdtemp(tmpl.data());
    if (!dir) { std::perror("mkdtemp"); return 1; }
    const std::string out = std::string(dir) + "/y.bin";

    const int n_shapes  = sizeof(k_shapes)/sizeof(k_shapes[0]);
    const int n_configs = sizeof(k_configs)/sizeof(k_configs[0]);

    bool ok = true;
    for (int si = 0; si < n_shapes; ++si) {
        const Shape & s = k_shapes[si];
        std::vector<float> base;
        for (int ci = 0; ci < n_configs; ++ci) {
            const Config & c = k_configs[ci];
            std::vector<float> got;
            std::set<std::string> kernels;
            if (!run_child(argv[0], si, ci, out, got, kernels)) {
                std::printf("FAIL %d x %d x %d [%d/%d] %s: the child failed\n", s.k, s.m, s.n, n_w(s), n_x(s), c.name);
                ok = false;
                continue;
            }
            const std::string want = wanted_kernel(s, c);
            const bool used = kernels.size() == 1 && kernels.count(want) == 1;
            std::string seen;
            for (const auto & n : kernels) { seen += (seen.empty() ? "" : ", ") + n; }

            if (ci == 0) {
                base = got;
                // sampled double reference: rows of the first and last activation matrix
                std::vector<uint8_t> packed;
                std::vector<float> x;
                make_inputs(s, packed, x);
                const size_t row_bytes = ggml_row_size(GGML_TYPE_Q1_0, s.k);
                std::vector<float> deq((size_t) s.k);
                double err = 0, ref2 = 0;
                for (int ib : {0, n_x(s) - 1}) {
                    const int i2 = ib % s.b1, i3 = ib / s.b1;
                    const int iw = i2 / (s.b1 / s.b0) + (i3 / (s.c1 / s.c0))*s.b0;
                    for (int i = 0; i < 8; ++i) {
                        const int r = (int) ((uint64_t) (i + 8*ib) * 2654435761u % (uint64_t) s.m);
                        ggml_get_type_traits(GGML_TYPE_Q1_0)->to_float(packed.data() + ((size_t) iw*s.m + r)*row_bytes, deq.data(), s.k);
                        for (int col = 0; col < s.n; ++col) {
                            double acc = 0;
                            const float * xc = x.data() + ((size_t) ib*s.n + col)*s.k;
                            for (int j = 0; j < s.k; ++j) { acc += (double) deq[j]*xc[j]; }
                            const double d = base[((size_t) ib*s.n + col)*s.m + r] - acc;
                            err += d*d; ref2 += acc*acc;
                        }
                    }
                }
                const bool pass = used && err/ref2 < 1e-5;
                ok &= pass;
                std::printf("%5d x %6d x %3d [%d/%d] %-8s %s  NMSE(flags off vs double) %.2e  %s\n", s.k, s.m, s.n, n_w(s), n_x(s),
                            eligible(s) ? "eligible" : "tail", s.what, err/ref2, pass ? "" : "FAIL");
                std::printf("    %-9s kernel %s%s\n", c.name, seen.c_str(), used ? "" : (" FAIL (want " + want + ")").c_str());
                continue;
            }

            size_t diff = 0;
            for (size_t i = 0; i < got.size(); ++i) { diff += std::memcmp(&got[i], &base[i], sizeof(float)) != 0; }
            const bool pass = diff == 0 && used && got.size() == base.size();
            ok &= pass;
            std::printf("    %-9s %s: %zu of %zu outputs differ; kernel %s%s\n", c.name, pass ? "PASS" : "FAIL", diff, got.size(),
                        seen.c_str(), used ? "" : (" (want " + want + ")").c_str());
        }
    }
    rmdir(dir);
    std::printf("%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
