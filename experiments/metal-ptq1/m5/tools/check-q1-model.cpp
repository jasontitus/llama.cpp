// Full-model check for the Q1_0 prefill research kernels (GGML_METAL_Q1_MM_K32_ALIGNED,
// GGML_METAL_Q1_SWIZZLE_LOG): every float logit bitwise against the same model without them.
//
// One process loads the model once, then runs each arm in a fresh context: the first n tokens of a text file as
// one prompt with logits for every position, split into micro-batches of 512 (with 700 tokens: a 512-token
// micro-batch, where every Q1_0 projection and the output head are K32-eligible, then a 188-token one, where
// they all stay on the generic kernel). Arm 0 is the reference; every arm must match it bit for bit, and a
// repeated arm checks run-to-run determinism. The mul_mm pipelines each arm creates are read from the library's
// "loaded" log lines (logged only for a pipeline that exists). Pipelines are cached for the life of the process,
// so an arm lists the kernels it used first; the research kernel an arm's flags select must have been created
// by the end of that arm, or the arm fails (a missing kernel falls back to the generic one silently).
//
// usage: check-q1-model model.gguf text.txt n_tokens "ENV=1 ..." ["..."] ...   ("-" for no extra flags)
//   the flags of each arm are applied on top of a clean environment (every research flag unset)
//
// build (repository root, Metal build in build/):
//   clang++ -std=c++17 -O2 -Iinclude -Iggml/include experiments/metal-ptq1/m5/tools/check-q1-model.cpp \
//     -Lbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,build/bin -o check-q1-model
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static const char * k_flags[] = {
    "GGML_METAL_Q1_MM_K32_ALIGNED", "GGML_METAL_Q1_SWIZZLE_LOG", "GGML_GDN_ROWS_PLAIN", "GGML_METAL_SMALLM",
    "GGML_METAL_SMALLM_MM", "GGML_METAL_Q1_GLU", "GGML_METAL_Q1_GLU_MAX", "GGML_METAL_Q1_GLU_NR0",
    "GGML_METAL_Q1_0_POPCNT", "GGML_METAL_BATCH_INVARIANT", "GGML_METAL_N_CB", "GGML_METAL_CB_STATS", "GGML_METAL_SMALLM_MM_MAX_N", "GGML_GDN_ROWS_PLAIN_MAX_TOKENS",
};

static std::set<std::string> g_compiled;

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

// the research kernel an arm's flags select (every eligible Q1_0 projection of Bonsai-27B has M/64 % 8 == 0)
static std::string wanted_kernel(const std::string & spec) {
    const size_t p = spec.find("GGML_METAL_Q1_SWIZZLE_LOG=");
    if (p != std::string::npos) {
        const int l = std::atoi(spec.c_str() + p + std::strlen("GGML_METAL_Q1_SWIZZLE_LOG="));
        if (l >= 1 && l <= 3) { return "kernel_mul_mm_q1_0_f32_k32_swizzle" + std::to_string(l); }
    }
    return spec.find("GGML_METAL_Q1_MM_K32_ALIGNED=1") != std::string::npos ? "kernel_mul_mm_q1_0_f32_k32" : "";
}

static void apply_env(const std::string & spec) {
    for (const char * f : k_flags) { unsetenv(f); }
    std::istringstream in(spec);
    std::string kv;
    while (in >> kv) {
        const size_t eq = kv.find('=');
        if (eq != std::string::npos) { setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1); }
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s model.gguf text.txt n_tokens \"ENV=1 ...\" [\"...\"] ...\n", argv[0]);
        return 1;
    }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::ifstream file(argv[2]);
    std::stringstream ss;
    ss << file.rdbuf();
    const std::string text = ss.str().substr(0, 64*1024);
    std::vector<llama_token> toks(text.size() + 16);
    const int n_text = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false);
    const int n = std::atoi(argv[3]);
    if (n_text < n || n < 1) { std::fprintf(stderr, "the text has %d tokens, need %d\n", n_text, n); return 1; }
    toks.resize(n);

    std::vector<float> reference;
    std::set<std::string> seen;   // every mul_mm pipeline created so far
    bool ok = true;
    for (int a = 4; a < argc; ++a) {
        const std::string spec = std::strcmp(argv[a], "-") == 0 ? "" : argv[a];
        apply_env(spec);
        g_compiled.clear();

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = ((n + 255)/256)*256; cp.n_batch = n; cp.n_ubatch = 512; cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { std::fprintf(stderr, "context creation failed\n"); return 1; }
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = true;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
        std::vector<float> all((size_t) n*n_vocab);
        for (int i = 0; i < n; ++i) {
            std::memcpy(all.data() + (size_t) i*n_vocab, llama_get_logits_ith(ctx, i), n_vocab*sizeof(float));
        }
        llama_batch_free(b);
        llama_free(ctx);

        std::string kernels;
        for (const auto & k : g_compiled) { kernels += (kernels.empty() ? "" : ", ") + k; seen.insert(k); }
        const std::string want = wanted_kernel(spec);
        if (!want.empty() && !seen.count(want)) {
            std::printf("arm %d [%s]: FAIL, %s was never created\n", a - 4, spec.c_str(), want.c_str());
            ok = false;
        }
        if (reference.empty()) {
            reference = std::move(all);
            std::printf("arm %d [%s]: reference, %d tokens x %d logits; new mul_mm kernels: %s\n", a - 4, spec.c_str(), n,
                        n_vocab, kernels.empty() ? "none" : kernels.c_str());
            continue;
        }
        size_t diff = 0, first = SIZE_MAX;
        for (size_t i = 0; i < all.size(); ++i) {
            if (std::memcmp(&all[i], &reference[i], sizeof(float)) != 0) { if (!diff) { first = i; } ++diff; }
        }
        ok &= diff == 0;
        std::printf("arm %d [%s]: %s, %zu of %zu logits differ", a - 4, spec.c_str(), diff ? "FAIL" : "bitwise equal", diff,
                    all.size());
        if (diff) { std::printf(" (first at token %zu)", first / n_vocab); }
        std::printf("; new mul_mm kernels: %s\n", kernels.empty() ? "none" : kernels.c_str());
    }
    apply_env("");
    llama_model_free(model);
    llama_backend_free();
    std::printf("%s\n", ok ? "ALL BITWISE EQUAL" : "FAILURES");
    return ok ? 0 : 1;
}
