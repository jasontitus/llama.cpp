// Checks BonsaiBench's MTP bridge (BonsaiBench/MTP/BonsaiMTP.cpp) on a Mac against the same model the phone
// uses: plain and MTP generation, upstream and with the PTQ1 flags, the generated tokens (must all match)
// and the MTP acceptance and speed (compare with llama-speculative-simple / llama-server, same settings:
// --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-min 0 --spec-draft-p-min 0 --temp 0).
//
// build (from the repository root, with a Metal build in build/: cmake --build build --target llama llama-common):
//   M=experiments/metal-ptq1/ios/BonsaiBench/BonsaiBench/MTP
//   clang++ -std=c++17 -O2 -DNDEBUG -I$M -Iinclude -Iggml/include -Icommon -Isrc -Ivendor \
//     experiments/metal-ptq1/ios/tools/check-mtp-bridge.cpp $M/BonsaiMTP.cpp \
//     -Lbuild/bin -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,build/bin -o check-mtp-bridge
// run:   ./check-mtp-bridge Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf [reps]   (ONLY="stack + MTP" runs one config)
// Run configurations in separate processes (ONLY=...) for speed comparisons: back-to-back GPU work drifts.
#include "BonsaiMTP.h"
#include "llama.h"
#include "log.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

static const char * PROMPT = "<|im_start|>user\nWrite a Python function that merges two sorted lists into one sorted list, with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

static void set_flags(bool stack) {
    const char * names[] = {"GGML_METAL_PTQ1_MULTICOL","GGML_METAL_PTQ1_MULTICOL_MAX","GGML_METAL_PTQ1_GLU","GGML_METAL_PTQ1_STAGE","GGML_GDN_ROWS_PLAIN","GGML_METAL_SMALLM_MM"};
    const char * vals[]  = {"1","8","1","1","1","1"};
    for (int i = 0; i < 6; i++) { if (stack) setenv(names[i], vals[i], 1); else unsetenv(names[i]); }
}

int main(int argc, char ** argv) {
    llama_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_WARN) fputs(t, stderr); }, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.load_mtp = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    printf("n_layer_nextn = %d\n", llama_model_n_layer_nextn(model));
    struct Cfg { const char * name; bool stack; int draft; };
    std::vector<Cfg> cfgs = {{"upstream plain",false,0},{"stack plain",true,0},{"upstream + MTP",false,1},{"stack + MTP",true,1}};
    std::map<std::string, std::vector<int32_t>> toks;
    int reps = argc > 2 ? atoi(argv[2]) : 2;
    for (int r = 0; r < reps; r++) {
        for (auto & c : cfgs) { if (getenv("ONLY") && std::string(getenv("ONLY")) != c.name) continue;
            set_flags(c.stack);
            std::vector<int32_t> out(128 + 4);
            bb_gen_result res;
            int n = bb_generate(model, PROMPT, 128, c.draft, getenv("WARM") ? atoi(getenv("WARM")) : 16, 1024, 8, out.data(), (int) out.size(), &res);
            if (n < 0) { printf("%-16s ERROR %s\n", c.name, res.error); continue; }
            out.resize(n);
            printf("rep %d %-16s gen %3d tok in %.3f s = %6.2f tok/s  prompt %d tok %.3f s  steps %d  drafted %d accepted %d (%.1f%%)  foot %.2f GB\n",
                   r, c.name, res.n_generated, res.generate_seconds, res.n_generated / res.generate_seconds, res.n_prompt,
                   res.prompt_seconds, res.n_steps, res.n_drafted, res.n_accepted,
                   res.n_drafted ? 100.0 * res.n_accepted / res.n_drafted : 0.0, res.footprint_bytes / 1e9); printf("      split: draft %.3f s, verify %.3f s, process %.3f s\n", res.draft_seconds, res.verify_seconds, res.process_seconds);
            if (r == 0) toks[c.name] = out; if (r == 0 && c.draft == 0 && !c.stack) { printf("bridge first 20:"); for (int i = 0; i < 20 && i < (int) out.size(); i++) printf(" %d", out[i]); printf("\n"); }
        }
    }
    auto & ref = toks["upstream plain"];
    for (auto & [k, v] : toks) {
        size_t same = 0; while (same < v.size() && same < ref.size() && v[same] == ref[same]) same++;
        printf("%-16s tokens %zu, identical to upstream plain for the first %zu%s\n", k.c_str(), v.size(), same, v == ref ? " (all)" : "");
    }
    llama_model_free(model);
    llama_backend_free();
}
