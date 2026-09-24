// Locate batch-size-dependent ops: process one token alone (n = 1) and as the first token of a k-token
// batch after an identical prefix, capture that token's slice of every graph node's output through the
// eval callback, and list the nodes (graph order) whose slices differ bitwise.
// usage: check-divergence model.gguf prompt k [max_report]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct capture {
    bool active = false;
    int  n_tokens = 1;
    std::vector<std::string> order;
    std::map<std::string, std::vector<float>> data;
    std::map<std::string, std::string> op;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    capture * c = (capture *) ud;
    if (ask) {
        return c->active;
    }
    if (!c->active || t->type != GGML_TYPE_F32 || t->data == nullptr || (t->buffer == nullptr && t->view_src == nullptr) ||
        ggml_nelements(t) == 0 || t->name[0] == 0) {
        return true;
    }
    const int T = c->n_tokens;
    // slice of token 0: token axis is dim 1 (2D activations) or dim 2 (per-head 3D tensors)
    int axis = -1;
    if (t->ne[1] == T && t->ne[2] == 1 && t->ne[3] == 1) axis = 1;
    else if (t->ne[2] == T && t->ne[3] == 1 && T > 1) axis = 2;
    else if (T == 1 && t->ne[2] == 1 && t->ne[3] == 1) axis = 1;
    if (axis < 0 || !ggml_is_contiguous(t)) {
        return true;
    }
    std::vector<float> all(ggml_nelements(t));
    ggml_backend_tensor_get(t, all.data(), 0, ggml_nbytes(t));
    std::vector<float> slice;
    if (axis == 1) {
        slice.assign(all.begin(), all.begin() + t->ne[0]);
    } else {
        for (int64_t j = 0; j < t->ne[1]; ++j) {
            const float * row = all.data() + j * t->ne[0];  // token 0 is the first ne0*ne1 block
            slice.insert(slice.end(), row, row + t->ne[0]);
        }
    }
    std::string key = t->name;
    // only uniquely named semantic nodes: generic and derived names repeat and would pair wrong tensors
    if (key.rfind("node_", 0) == 0 || key.find('(') != std::string::npos || key[0] == ' ') {
        return true;
    }
    if (c->data.count(key)) {
        key += "#dup";
    }
    if (!c->data.count(key)) c->order.push_back(key);
    c->data[key] = slice;
    c->op[key] = ggml_op_desc(t);
    if (getenv("SHAPES") && c->order.size() <= 12) std::fprintf(stdout, "T=%d %-36s %-10s ne=[%lld,%lld,%lld,%lld] axis=%d\n", T, t->name, ggml_op_desc(t), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], axis);
    return true;
}

static capture run(llama_model * model, const std::vector<llama_token> & toks, int pre, int k) {
    capture c;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.cb_eval = cb; cp.cb_eval_user_data = &c;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_batch b = llama_batch_init(512, 0, 1);
    auto submit = [&](int from, int count, bool cap) {
        b.n_tokens = 0;
        for (int i = 0; i < count; ++i) {
            b.token[i] = toks[from + i]; b.pos[i] = from + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = true;
            b.n_tokens++;
        }
        c.active = cap; c.n_tokens = count;
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); std::exit(1); }
        c.active = false;
    };
    submit(0, pre, false);
    submit(pre, k, true);
    llama_batch_free(b);
    llama_free(ctx);
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s model prompt k [max_report]\n", argv[0]); return 1; }
    const int k = std::atoi(argv[3]);
    const int max_report = argc > 4 ? std::atoi(argv[4]) : 20;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(vocab, argv[2], std::strlen(argv[2]), toks.data(), toks.size(), true, false);
    toks.resize(n);
    const int pre = n - k;
    capture a = run(model, toks, pre, 1);
    capture bb = run(model, toks, pre, k);
    int reported = 0, compared = 0, differing = 0;
    std::map<std::string, int> first_by_op;
    for (const auto & key : a.order) {
        if (!bb.data.count(key) || a.data[key].size() != bb.data[key].size()) continue;
        ++compared;
        const auto & x = a.data[key]; const auto & y = bb.data[key];
        double maxabs = 0; bool same = true;
        for (size_t i = 0; i < x.size(); ++i) { if (x[i] != y[i]) same = false; maxabs = std::max(maxabs, (double) std::fabs(x[i] - y[i])); }
        if (!same) {
            ++differing;
            first_by_op[a.op[key]]++;
            if (reported++ < max_report) std::printf("DIFF %-40s %-16s maxabs=%.3g\n", key.c_str(), a.op[key].c_str(), maxabs);
        }
    }
    std::printf("compared %d nodes, %d differ bitwise; differing nodes by op:", compared, differing);
    for (auto & kv : first_by_op) std::printf(" %s=%d", kv.first.c_str(), kv.second);
    std::printf("\n");
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
