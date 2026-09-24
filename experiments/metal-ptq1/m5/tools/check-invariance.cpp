// Batch invariance probe: does a token's full-vocabulary logit vector depend on how many tokens are
// processed with it? Runs a fixed token sequence once one token per decode (n = 1), then again in
// batches of k (k = 2, 3, 4, like MTP verification), requesting logits for every position, and
// compares each position bitwise and by max |diff|. Environment variables select kernel paths.
// usage: check-invariance model.gguf prompt n_tokens k [k ...]
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static size_t g_prompt_len = 8;

static std::vector<std::vector<float>> run(llama_model * model, const std::vector<llama_token> & toks, int k) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 8192;
    cp.n_batch  = 8192;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<std::vector<float>> out;
    // the prompt prefix (first 8 tokens) always goes in one batch so every run starts from the same state
    const int pre = (int) g_prompt_len;
    llama_batch b = llama_batch_init(8192, 0, 1);
    auto submit = [&](int from, int count, bool want) {
        b.n_tokens = 0;
        for (int i = 0; i < count; ++i) {
            b.token[b.n_tokens] = toks[from + i];
            b.pos[b.n_tokens] = from + i;
            b.n_seq_id[b.n_tokens] = 1;
            b.seq_id[b.n_tokens][0] = 0;
            b.logits[b.n_tokens] = want;
            b.n_tokens++;
        }
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); std::exit(1); }
        if (want) {
            for (int i = 0; i < count; ++i) {
                const float * l = llama_get_logits_ith(ctx, i);
                out.emplace_back(l, l + n_vocab);
            }
        }
    };
    submit(0, pre, false);
    for (int p = pre; p < (int) toks.size(); p += k) {
        submit(p, std::min(k, (int) toks.size() - p), true);
    }
    llama_batch_free(b);
    llama_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s model prompt n_tokens k [k...]\n", argv[0]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(vocab, argv[2], std::strlen(argv[2]), toks.data(), toks.size(), true, false);
    toks.resize(n);
    // prompt from a file with @path; the prompt becomes the shared prefix
    g_prompt_len = std::min<size_t>(toks.size(), 8);
    if (argv[2][0] == '@') {
        FILE * f = std::fopen(argv[2] + 1, "rb");
        std::string text; char buf[4096]; size_t r;
        while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, r);
        std::fclose(f);
        toks.resize(text.size() + 16);
        n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false);
        toks.resize(n);
        g_prompt_len = toks.size();
    }
    const int want = std::atoi(argv[3]) + (argv[2][0] == '@' ? (int) g_prompt_len : 0);
    // extend the prompt with its own greedy continuation so every run sees the identical sequence
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 8192;
        cp.n_batch = 8192;
        cp.n_ubatch = 512;
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_batch b = llama_batch_get_one(toks.data(), toks.size());
        llama_decode(ctx, b);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        while ((int) toks.size() < want) {
            const float * l = llama_get_logits_ith(ctx, -1);
            llama_token t = (llama_token) (std::max_element(l, l + n_vocab) - l);
            toks.push_back(t);
            llama_batch b1 = llama_batch_get_one(&toks.back(), 1);
            llama_decode(ctx, b1);
        }
        llama_free(ctx);
    }
    const auto ref = run(model, toks, 1);
    for (int a = 4; a < argc; ++a) {
        const int k = std::atoi(argv[a]);
        const auto got = run(model, toks, k);
        int bitwise = 0; double maxabs = 0, err = 0, nrm = 0; int argmax_diff = 0;
        for (size_t p = 0; p < ref.size(); ++p) {
            bool same = true;
            for (size_t i = 0; i < ref[p].size(); ++i) {
                const double d = (double) got[p][i] - ref[p][i];
                if (got[p][i] != ref[p][i]) same = false;
                maxabs = std::max(maxabs, std::fabs(d)); err += d*d; nrm += (double) ref[p][i]*ref[p][i];
            }
            bitwise += same;
            argmax_diff += (std::max_element(ref[p].begin(), ref[p].end()) - ref[p].begin()) != (std::max_element(got[p].begin(), got[p].end()) - got[p].begin());
        }
        std::printf("k=%d positions=%zu bitwise_identical=%d max_abs=%.3g nmse=%.3g argmax_changed=%d\n",
                    k, ref.size(), bitwise, maxabs, err / nrm, argmax_diff);
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
