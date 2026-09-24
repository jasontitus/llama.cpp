#include "BonsaiMTP.h"

#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include <mach/mach.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
#include <os/proc.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t footprint() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return info.phys_footprint;
}

uint64_t available() {
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
    return os_proc_available_memory();
#else
    return 0;
#endif
}

struct run_stats {
    std::vector<llama_token> tokens;
    double t_prompt = 0, t_gen = 0, t_draft = 0, t_verify = 0, t_process = 0;
    int drafted = 0, accepted = 0, steps = 0;
};

// One greedy generation on existing contexts. This is examples/speculative-simple's loop (including its
// checkpoint handling for partial acceptance), stopping at exactly n_predict tokens like llama-server.
// seq_rm_tgt / seq_rm_dft come from common_context_can_seq_rm, which clears the context and decodes test
// tokens: it must run once, before any prompt is evaluated (as in the example), never between runs.
bool generate(common_params & params, llama_model * model, llama_context * ctx_tgt, llama_context * ctx_dft,
              common_context_seq_rm_type seq_rm_tgt, common_context_seq_rm_type seq_rm_dft,
              const std::vector<llama_token> & inp, int n_predict, run_stats & out, std::string & err) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_seq_id seq_id = 0;

    common_speculative * spec = nullptr;
    if (ctx_dft) {
        spec = common_speculative_init(params.speculative, 1);
        if (spec == nullptr) {
            err = "failed to initialize speculative decoding";
            return false;
        }
    }
    struct spec_guard {
        common_speculative * s;
        ~spec_guard() { if (s) common_speculative_free(s); }
    } guard{spec};

    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));

    // prompt: all but the last token, which the loop evaluates with the first draft
    const double t0 = now_s();
    {
        llama_batch batch_prompt = llama_batch_init((int32_t) inp.size(), 0, 1);
        for (size_t i = 0; i + 1 < inp.size(); ++i) {
            common_batch_add(batch_prompt, inp[i], (llama_pos) i, { seq_id }, false);
        }
        int rc = batch_prompt.n_tokens > 0 ? llama_decode(ctx_tgt, batch_prompt) : 0;
        if (rc == 0 && spec && !common_speculative_process(spec, batch_prompt)) {
            err = "failed to process the prompt for speculative decoding";
            llama_batch_free(batch_prompt);
            return false;
        }
        llama_batch_free(batch_prompt);
        if (rc != 0) {
            err = "prompt decode failed (" + std::to_string(rc) + ")";
            return false;
        }
        llama_synchronize(ctx_tgt);
    }
    const double t1 = now_s();

    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
    int n_past = (int) inp.size() - 1;

    if (spec) {
        common_speculative_begin(spec, seq_id, prompt_tgt);
    }

    bool use_ckpt_tgt = false;
    bool use_ckpt_dft = seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    struct batch_guard {
        llama_batch & b;
        ~batch_guard() { llama_batch_free(b); }
    } bguard{batch_tgt};

    llama_tokens draft;
    common_prompt_checkpoint ckpt;
    int n_generated = 0;
    bool done = false;

    while (!done) {
        if (draft.empty()) {
            if (spec) {
                ckpt.update_pos(prompt_tgt.size(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));
                if (seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                    ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
                n_draft_max = std::min(n_draft_max, n_predict - n_generated - 1);
                n_draft_max = std::max(n_draft_max, 0);

                common_speculative_get_draft_params(spec, seq_id) = {
                    /* .drafting = */ true,
                    /* .n_max    = */ n_draft_max,
                    /* .n_past   = */ n_past,
                    /* .id_last  = */ id_last,
                    /* .prompt   = */ &prompt_tgt,
                    /* .result   = */ &draft,
                };
                const double td = now_s();
                common_speculative_draft(spec);
                out.t_draft += now_s() - td;
                // with n_min 0 and p_min 0 the MTP head always proposes a token; none means its context
                // failed (e.g. a Metal error on the draft context), and the run would silently turn plain
                if (n_draft_max > 0 && draft.empty()) {
                    err = "the MTP draft context produced no draft (see the library messages)";
                    return false;
                }

                if (seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);

                if (!draft.empty()) {
                    use_ckpt_tgt = seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                                   (seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));
                    const bool ckpt_dft_rs = seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft);
                    use_ckpt_dft = seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ckpt_dft_rs;
                    if (use_ckpt_tgt) {
                        ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                    if (ckpt_dft_rs) {
                        ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                } else {
                    use_ckpt_tgt = false;
                }
            }
        }

        // evaluate [id_last, draft...] on the target
        const double tv = now_s();
        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, id_last, n_past++, { seq_id }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + (llama_pos) i, { seq_id }, true);
        }
        const int rc = llama_decode(ctx_tgt, batch_tgt);
        if (rc != 0) {
            err = "llama_decode failed (" + std::to_string(rc) + ")";
            return false;
        }
        out.steps++;

        const double tp = now_s();
        if (spec && !common_speculative_process(spec, batch_tgt)) {
            err = "failed to process the batch for speculative decoding";
            return false;
        }
        out.t_process += now_s() - tp;
        const double ts = now_s();

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        const size_t n_draft = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
        if (ids.empty()) {
            err = "sampling returned no token";
            return false;
        }
        out.t_verify += (tp - tv) + (now_s() - ts);

        // partial acceptance without partial sequence removal: restore and retry with the accepted part
        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            draft = std::move(ids);
            ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        if (spec) {
            common_speculative_accept(spec, seq_id, (uint16_t) (ids.size() - 1));
        }
        n_past       += (int) ids.size() - 1;
        out.drafted  += (int) n_draft;
        out.accepted += (int) ids.size() - 1;

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            out.tokens.push_back(id_last);
            n_generated++;
            if (llama_vocab_is_eog(vocab, id_last) || n_generated >= n_predict) {
                done = true;
                break;
            }
        }

        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
        }
    }
    llama_synchronize(ctx_tgt);
    const double t2 = now_s();

    // A failed Metal command buffer only marks the backend; the next decode reports it.
    common_batch_clear(batch_tgt);
    common_batch_add(batch_tgt, id_last, n_past, { seq_id }, true);
    const int rc = llama_decode(ctx_tgt, batch_tgt);
    llama_synchronize(ctx_tgt);
    if (rc != 0) {
        err = "a GPU command failed during generation (llama_decode " + std::to_string(rc) + " afterwards)";
        return false;
    }

    out.t_prompt = t1 - t0;
    out.t_gen    = t2 - t1;
    return true;
}

} // namespace

extern "C" void bb_common_log_to_file(const char * path) {
    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), false);
    common_log_set_file(common_log_main(), path);
}

extern "C" int32_t bb_generate(struct llama_model * model, const char * prompt, int32_t n_predict, int32_t n_draft,
                               int32_t n_warmup, int32_t n_ctx, int32_t n_threads, int32_t * out_tokens,
                               int32_t capacity, bb_gen_result * result) {
    std::memset(result, 0, sizeof(*result));
    struct log_flush {
        ~log_flush() { common_log_flush(common_log_main()); }
    } flush_on_return;
    auto fail = [&](const std::string & msg) {
        std::snprintf(result->error, sizeof(result->error), "%s", msg.c_str());
        return (int32_t) -1;
    };

    // the settings of the Mac server studies: greedy, 1 draft token, no minimum draft or probability
    common_params params;
    params.n_ctx                       = n_ctx;
    params.n_batch                     = 512;
    params.n_ubatch                    = 512;
    params.n_parallel                  = 1;
    params.flash_attn_type             = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cpuparams.n_threads         = n_threads;
    params.cpuparams_batch.n_threads   = n_threads;
    params.sampling.temp               = 0.0f;
    params.sampling.seed               = 20260923;
    params.sampling.penalty_repeat     = 1.0f;
    if (n_draft > 0) {
        params.speculative.types       = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = n_draft;
        params.speculative.draft.n_min = 0;
        params.speculative.draft.p_min = 0.0f;
    }
    const auto limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
                                                             common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = limits.total;
    params.n_outputs_max_per_seq = limits.per_seq;

    if (n_draft > 0 && llama_model_n_layer_nextn(model) <= 0) {
        return fail("this model has no MTP layers loaded (use an MTP GGUF)");
    }

    // declared before the contexts, so it is destroyed after them (a context keeps a pointer to its pool)
    common_threadpools threadpools;

    llama_context * ctx_tgt = llama_init_from_model(model, common_context_params_to_llama(params));
    if (ctx_tgt == nullptr) {
        return fail("could not create the target context (out of memory?)");
    }
    llama_context_ptr ctx_tgt_owner(ctx_tgt);

    // the persistent CPU threadpool that llama-server and the examples attach through common_init, so the
    // graphs' CPU splits run as they do in the Mac server studies (its threads poll between graphs, as
    // there; gen cells of both arms pay that the same way)
    threadpools.init(ctx_tgt, params);

    common_speculative_init_result_ptr spec_init;
    llama_context * ctx_dft = nullptr;
    if (n_draft > 0) {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model, ctx_tgt);
        ctx_dft = spec_init->context();
        if (ctx_dft == nullptr) {
            return fail("could not create the MTP draft context (out of memory?)");
        }
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft;
    }

    // before anything is evaluated: this probe clears both contexts
    const common_context_seq_rm_type seq_rm_tgt = common_context_can_seq_rm(ctx_tgt);
    const common_context_seq_rm_type seq_rm_dft = ctx_dft ? common_context_can_seq_rm(ctx_dft) : COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    const std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt, true, true);
    if (inp.size() < 2 || (int) inp.size() + n_predict + n_draft + 2 > n_ctx) {
        return fail("prompt too short or context too small");
    }

    std::string err;
    if (n_warmup > 0) {
        run_stats warm;
        if (!generate(params, model, ctx_tgt, ctx_dft, seq_rm_tgt, seq_rm_dft, inp, n_warmup, warm, err)) {
            return fail("warmup: " + err);
        }
        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        if (ctx_dft) {
            llama_memory_clear(llama_get_memory(ctx_dft), true);
        }
    }

    run_stats run;
    if (!generate(params, model, ctx_tgt, ctx_dft, seq_rm_tgt, seq_rm_dft, inp, n_predict, run, err)) {
        return fail(err);
    }
    result->footprint_bytes = footprint();
    result->available_bytes = available();

    result->n_prompt         = (int32_t) inp.size();
    result->prompt_seconds   = run.t_prompt;
    result->n_generated      = (int32_t) run.tokens.size();
    result->generate_seconds = run.t_gen;
    result->n_drafted        = run.drafted;
    result->n_accepted       = run.accepted;
    result->n_steps          = run.steps;
    result->draft_seconds    = run.t_draft;
    result->verify_seconds   = run.t_verify;
    result->process_seconds  = run.t_process;
    const int32_t n = std::min<int32_t>((int32_t) run.tokens.size(), capacity);
    std::copy(run.tokens.begin(), run.tokens.begin() + n, out_tokens);
    return n;
}
