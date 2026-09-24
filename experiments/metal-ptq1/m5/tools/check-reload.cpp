// In-process flag switching check. Loads the model once, then for each arm sets the environment,
// creates a fresh context (which re-reads the research flags), evaluates a fixed token sequence in
// batches of 1 and of 2 with logits for every position, and writes all logits to <prefix>-<i>.bin.
// usage: check-reload model.gguf prefix "ENV=1 ENV2=1" ["..."] ...   (use "-" for an empty arm)
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

static const char * k_flags[] = {
    "GGML_METAL_PTQ1_MULTICOL", "GGML_METAL_PTQ1_MULTICOL_MAX", "GGML_METAL_PTQ1_GLU", "GGML_METAL_PTQ1_STAGE",
    "GGML_METAL_PTQ1_TENSOR", "GGML_METAL_PTQ1_TENSOR_MIN", "GGML_GDN_ROWS_PLAIN", "GGML_METAL_SMALLM",
    "GGML_METAL_SMALLM_MM", "GGML_METAL_PQ2_MULTICOL", "GGML_METAL_PQ2_GLU", "GGML_METAL_Q1_GLU",
    "GGML_METAL_BATCH_INVARIANT", "GGML_METAL_Q1_0_POPCNT", "GGML_METAL_PTQ1_NR0", "GGML_METAL_PTQ1_NSG",
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

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s model prefix arm...\n", argv[0]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * text = "The committee reviewed the proposal carefully, noting that the budget estimates for the second phase "
                        "depended on assumptions about supplier prices that had not been confirmed.";
    std::vector<llama_token> toks(256);
    const int n = llama_tokenize(vocab, text, std::strlen(text), toks.data(), toks.size(), true, false);
    toks.resize(n);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int a = 3; a < argc; ++a) {
        apply_env(std::string(argv[a]) == "-" ? "" : argv[a]);
        std::vector<float> all;
        for (int k : {1, 2}) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
            llama_context * ctx = llama_init_from_model(model, cp);
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
