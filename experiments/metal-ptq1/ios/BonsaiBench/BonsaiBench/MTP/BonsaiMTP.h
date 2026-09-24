// Generation through llama.cpp's common speculative-decoding code (the loop of
// examples/speculative-simple, as llama-server runs it), for BonsaiBench's gen cells.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;

typedef struct bb_gen_result {
    int32_t  n_prompt;          // prompt tokens
    double   prompt_seconds;    // evaluating all but the last prompt token (the loop starts from it)
    int32_t  n_generated;       // tokens generated (at most n_predict; stops at end of generation)
    double   generate_seconds;  // from the end of the prompt pass to the last generated token
    int32_t  n_drafted;         // MTP draft tokens verified
    int32_t  n_accepted;        // of which accepted
    int32_t  n_steps;           // target decodes in the generation loop
    double   draft_seconds;     // generation loop split: MTP drafting,
    double   verify_seconds;    // target decode, its GPU time and sampling,
    double   process_seconds;   // and feeding the verified batch to the MTP context
    uint64_t footprint_bytes;   // process footprint while the contexts are alive
    uint64_t available_bytes;   // memory iOS still allows the app then (0 off device)
    char     error[256];        // set when the call returns -1
} bb_gen_result;

// llama.cpp's common code (the speculative-decoding loop) logs through its own logger, not llama_log_set:
// mirror it to a file (with W/E level prefixes) that the app reads after each run. Call once, first.
void bb_common_log_to_file(const char * path);

// Greedy generation of up to n_predict tokens after `prompt` on fresh contexts of `model`: with MTP
// self-speculation (n_draft draft tokens per step, the target model must have been loaded with
// load_mtp) or, for n_draft == 0, plain decoding through the same loop. A warmup generation of n_warmup
// tokens runs first on the same contexts. Writes the generated tokens to out_tokens (capacity) and
// returns their number, or -1 with result->error set.
int32_t bb_generate(struct llama_model * model, int32_t mtp_loaded, const char * prompt, int32_t n_predict,
                    int32_t n_draft, int32_t n_warmup, int32_t n_ctx, int32_t n_threads, int32_t * out_tokens,
                    int32_t capacity, bb_gen_result * result);

#ifdef __cplusplus
}
#endif
