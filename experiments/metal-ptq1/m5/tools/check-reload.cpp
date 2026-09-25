// In-process flag switching check. Loads the model once, then for each arm sets the environment,
// creates a fresh context (which re-reads the research flags), evaluates a fixed token sequence in
// batches of 1 and of 2 with logits for every position, and writes all logits to <prefix>-<i>.bin.
// usage: check-reload model.gguf prefix "ENV=1 ENV2=1" ["..."] ...   (use "-" for an empty arm)
#include "llama.h"
#include "llama-context.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static const char * k_flags[] = {
    "GGML_METAL_BATCH_INVARIANT",
    "GGML_METAL_PQ2_GLU",
    "GGML_METAL_PQ2_GLU_NR0",
    "GGML_METAL_PQ2_MC_MAX",
    "GGML_METAL_PQ2_MULTICOL",
    "GGML_METAL_PQ2_NR0",
    "GGML_METAL_PQ2_NSG",
    "GGML_METAL_PTQ1_GLU",
    "GGML_METAL_PTQ1_GLU_NR0",
    "GGML_METAL_PTQ1_GLU_NSG",
    "GGML_METAL_PTQ1_MM_B128",
    "GGML_METAL_PTQ1_MULTICOL",
    "GGML_METAL_PTQ1_MULTICOL_MAX",
    "GGML_METAL_PTQ1_NR0",
    "GGML_METAL_PTQ1_NSG",
    "GGML_METAL_PTQ1_STAGE",
    "GGML_METAL_PTQ1_TENSOR",
    "GGML_METAL_PTQ1_TENSOR_MIN",
    "GGML_METAL_Q1_0_POPCNT",
    "GGML_METAL_Q1_GLU",
    "GGML_METAL_Q1_GLU_MAX",
    "GGML_METAL_Q1_GLU_NR0",
    "GGML_METAL_Q1_MM_K32_ALIGNED",
    "GGML_METAL_Q1_SWIZZLE_LOG",
    "GGML_METAL_N_CB",
    "GGML_METAL_SMALLM_MM_MAX_N",
    "GGML_GDN_ROWS_PLAIN_MAX_TOKENS",
    "GGML_METAL_SMALLM",
    "GGML_METAL_SMALLM_MM",
    "GGML_GDN_ROWS_PLAIN",
};

static void apply_env(const std::string & spec) {
    for (const char * f : k_flags) unsetenv(f);
    std::istringstream in(spec);
    std::string kv;
    while (in >> kv) {
        const size_t eq = kv.find('=');
        if (eq != std::string::npos) setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
    }
}

// Allocate before creating a backend, then retain the same graph across profile changes.
static bool check_retained_allocation(ggml_type type) {
    apply_env("");
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (!dev) { return false; }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const int k = 128, m = type == GGML_TYPE_PTQ1_0 ? 4163 : 7, n = 3;
    ggml_init_params params = {ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_tensor * guard = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const size_t reserved = ggml_backend_buft_get_alloc_size(buft, output);
    const size_t bytes = ggml_nbytes(output);
    const size_t required = type == GGML_TYPE_PTQ1_0 ?
            ((bytes + 15)/16)*16 + size_t(5)*k*n : bytes + size_t(k/128)*n*36*sizeof(uint32_t);
    bool ok = reserved >= required;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> w(k*m, 1.0f), x(k*n, 1.0f), guard_values(64, 9876.0f);
    std::vector<uint8_t> packed(ggml_row_size(type, k)*m);
    ggml_quantize_chunk(type, w.data(), packed.data(), 0, m, k, nullptr);
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(guard, guard_values.data(), 0, guard_values.size()*sizeof(float));
    const std::vector<std::string> profiles = type == GGML_TYPE_PTQ1_0 ? std::vector<std::string>{
        "", "GGML_METAL_PTQ1_MULTICOL=1 GGML_METAL_PTQ1_MULTICOL_MAX=8 GGML_METAL_PTQ1_STAGE=1",
        "GGML_METAL_PTQ1_TENSOR=1 GGML_METAL_PTQ1_TENSOR_MIN=2", "" } : std::vector<std::string>{
        "", "GGML_METAL_Q1_0_POPCNT=0", "" };
    std::vector<float> reference;
    for (const auto & profile : profiles) {
        apply_env(profile);
        ok &= ggml_backend_buft_get_alloc_size(buft, output) == reserved;
        ggml_backend_t backend = ggml_backend_init_by_name("MTL0", nullptr);
        if (!backend) { ok = false; break; }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_free(backend); ok = false; break;
        }
        ggml_backend_synchronize(backend);
        std::vector<float> got(m*n), guards(guard_values.size());
        ggml_backend_tensor_get(output, got.data(), 0, got.size()*sizeof(float));
        ggml_backend_tensor_get(guard, guards.data(), 0, guards.size()*sizeof(float));
        if (reference.empty()) { reference = got; }
        for (size_t i = 0; i < got.size(); ++i) {
            ok &= std::isfinite(got[i]) && std::abs(got[i] - reference[i]) < 1e-3f;
        }
        ok &= guards == guard_values;
        ggml_backend_free(backend);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    apply_env("");
    std::printf("retained %s allocation: %zu output + %zu scratch bytes %s\n", ggml_type_name(type),
            bytes, reserved - bytes, ok ? "PASS" : "FAIL");
    return ok;
}

static bool check_context_lifetime() {
    apply_env("");
    ggml_backend_t first = ggml_backend_init_by_name("MTL0", nullptr);
    if (!first) { std::fprintf(stderr, "first Metal context failed\n"); return false; }
    ggml_backend_t second = ggml_backend_init_by_name("MTL0", nullptr);
    bool ok = second != nullptr;
    ggml_backend_free(first);
    if (second) {
        setenv("GGML_METAL_PTQ1_STAGE", "1", 1);
        ggml_backend_t changed = ggml_backend_init_by_name("MTL0", nullptr);
        ok &= changed == nullptr;
        if (changed) { ggml_backend_free(changed); }
        unsetenv("GGML_METAL_PTQ1_STAGE");
        setenv("GGML_METAL_Q1_0_POPCNT", "0", 1);
        changed = ggml_backend_init_by_name("MTL0", nullptr);
        ok &= changed == nullptr; // Presence-style flags distinguish unset from "0".
        if (changed) { ggml_backend_free(changed); }
        unsetenv("GGML_METAL_Q1_0_POPCNT");
        ggml_backend_free(second);
    }
    setenv("GGML_METAL_PTQ1_STAGE", "1", 1);
    ggml_backend_t changed = ggml_backend_init_by_name("MTL0", nullptr);
    ok &= changed != nullptr;
    if (changed) { ggml_backend_free(changed); }
    apply_env("");
    ggml_backend_t again = ggml_backend_init_by_name("MTL0", nullptr);
    ok &= again != nullptr;
    if (again) { ggml_backend_free(again); }
    std::printf("context lifetime: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool check_frozen_gdn(llama_model * model) {
    apply_env("");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
    llama_context * held = llama_init_from_model(model, cp);
    if (!held) { return false; }
    bool ok = !held->get_cparams().gdn_rows_plain;
    llama_token tokens[] = { 1, 2, 3 };
    ok &= llama_decode(held, llama_batch_get_one(tokens, 1)) == 0;
    setenv("GGML_GDN_ROWS_PLAIN", "1", 1);
    // Change batch width to rebuild the held context's graph after the environment changes.
    ok &= llama_decode(held, llama_batch_get_one(tokens + 1, 2)) == 0;
    ok &= !held->get_cparams().gdn_rows_plain;
    llama_free(held);
    llama_context * next = llama_init_from_model(model, cp);
    ok &= next != nullptr;
    if (next) {
        ok &= next->get_cparams().gdn_rows_plain;
        llama_free(next);
    }
    apply_env("");
    llama_context * again = llama_init_from_model(model, cp);
    ok &= again != nullptr;
    if (again) {
        ok &= !again->get_cparams().gdn_rows_plain;
        llama_free(again);
    }
    std::printf("context GDN snapshot: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s model prefix arm...\n", argv[0]); return 1; }
    llama_backend_init();
    if (!check_retained_allocation(GGML_TYPE_PTQ1_0) || !check_retained_allocation(GGML_TYPE_Q1_0) ||
        !check_context_lifetime()) { return 1; }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model || !check_frozen_gdn(model)) { return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * text = "The committee reviewed the proposal carefully, noting that the budget estimates for the second phase "
                        "depended on assumptions about supplier prices that had not been confirmed.";
    std::vector<llama_token> toks(256);
    const int n = llama_tokenize(vocab, text, std::strlen(text), toks.data(), toks.size(), true, false);
    toks.resize(n);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<std::vector<float>> previous_logits;
    for (int a = 3; a < argc; ++a) {
        apply_env(std::string(argv[a]) == "-" ? "" : argv[a]);
        std::vector<float> all;
        for (int k : {1, 2}) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
            llama_context * ctx = llama_init_from_model(model, cp);
            if (!ctx) { std::fprintf(stderr, "context creation failed\n"); return 1; }
            llama_batch b = llama_batch_init(512, 0, 1);
            for (int p = 0; p < n; p += k) {
                b.n_tokens = 0;
                for (int i = p; i < std::min(n, p + k); ++i) {
                    b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = i; b.n_seq_id[b.n_tokens] = 1;
                    b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = true; b.n_tokens++;
                }
                if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
                for (int i = 0; i < b.n_tokens; ++i) {
                    const float * l = llama_get_logits_ith(ctx, i);
                    all.insert(all.end(), l, l + n_vocab);
                }
            }
            llama_batch_free(b);
            llama_free(ctx);
        }
        for (int previous = 3; previous < a; ++previous) {
            if (std::strcmp(argv[previous], argv[a]) == 0) {
                const auto & expected = previous_logits[previous - 3];
                if (all.size() != expected.size() || std::memcmp(all.data(), expected.data(), all.size()*sizeof(float))) {
                    std::fprintf(stderr, "repeated arm %d differs from %d\n", a - 3, previous - 3);
                    return 1;
                }
            }
        }
        previous_logits.push_back(all);
        const std::string path = std::string(argv[2]) + "-" + std::to_string(a - 3) + ".bin";
        FILE * f = std::fopen(path.c_str(), "wb");
        std::fwrite(all.data(), sizeof(float), all.size(), f);
        std::fclose(f);
        std::printf("arm %d [%s]: %zu logits -> %s\n", a - 3, argv[a], all.size(), path.c_str());
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
