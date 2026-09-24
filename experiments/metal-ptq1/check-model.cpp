#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// Dump full-vocabulary logits and greedy tokens. Optional teacher tokens keep
// the B run on A's exact prefix even if the argmax ever differs.
int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s model.gguf ubatch output-prefix prompt [teacher.tokens]\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_name("MTL0");
    if (!dev) { std::fprintf(stderr, "MTL0 required\n"); return 1; }
    ggml_backend_dev_t devices[] = {dev, nullptr};
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.devices = devices;
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = std::atoi(argv[2]);
    cp.n_threads = cp.n_threads_batch = 16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) { return 1; }
    auto vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    const std::string prompt = argv[4], prefix = argv[3];
    std::vector<llama_token> tokens(prompt.size()+32);
    int count = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (count < 0) { return 1; }
    tokens.resize(count);
    std::vector<llama_token> teacher;
    if (argc == 6) {
        std::ifstream f(argv[5]);
        llama_token token;
        while (f >> token) { teacher.push_back(token); }
        if (teacher.empty()) { return 1; }
    }
    std::ofstream logits_file(prefix+".logits", std::ios::binary);
    std::ofstream token_file(prefix+".tokens"), text_file(prefix+".text");
    std::ofstream meta_file(prefix+".meta");
    if (!logits_file || !token_file || !text_file || !meta_file) { return 1; }
    meta_file << nv << " " << count << " " << llama_n_ubatch(ctx) << "\n";
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) { return 1; }
    int steps = teacher.empty() ? 32 : teacher.size();
    for (int step = 0; step < steps; ++step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) { return 1; }
        for (int i = 0; i < nv; ++i) { if (!std::isfinite(logits[i])) { return 1; } }
        logits_file.write(reinterpret_cast<const char *>(logits), nv*sizeof(float));
        const llama_token next = std::max_element(logits, logits+nv)-logits;
        token_file << next << "\n";
        char piece[512];
        int len = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, true);
        if (len < 0) { return 1; }
        text_file.write(piece, len);
        llama_token feed = teacher.empty() ? next : teacher[step];
        if (llama_vocab_is_eog(vocab, feed) || step+1 == steps) { break; }
        if (llama_decode(ctx, llama_batch_get_one(&feed, 1)) != 0) { return 1; }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
