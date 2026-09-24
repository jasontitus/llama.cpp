import Foundation
import llama

/// Research flags understood by the patched Metal backend. They are environment variables that the
/// backend re-reads whenever a new context is created, so each observation sets them, then creates
/// a fresh context.
let researchFlagNames: [String] = [
    "GGML_METAL_PTQ1_MULTICOL", "GGML_METAL_PTQ1_MULTICOL_MAX", "GGML_METAL_PTQ1_GLU", "GGML_METAL_PTQ1_STAGE",
    "GGML_METAL_PTQ1_TENSOR", "GGML_METAL_PTQ1_TENSOR_MIN", "GGML_GDN_ROWS_PLAIN", "GGML_METAL_SMALLM",
    "GGML_METAL_SMALLM_MM", "GGML_METAL_PQ2_MULTICOL", "GGML_METAL_PQ2_GLU", "GGML_METAL_Q1_GLU",
    "GGML_METAL_BATCH_INVARIANT", "GGML_METAL_Q1_0_POPCNT",
]

func applyFlags(_ flags: [String: String]) {
    for name in researchFlagNames { unsetenv(name) }
    for (k, v) in flags { setenv(k, v, 1) }
}

enum EngineError: Error, LocalizedError {
    case loadFailed(String), contextFailed, decodeFailed(Int32), tokenizeFailed

    var errorDescription: String? {
        switch self {
        case .loadFailed(let p): return "could not load model at \(p)"
        case .contextFailed:     return "could not create a context (out of memory?)"
        case .decodeFailed(let c): return "llama_decode failed (\(c))"
        case .tokenizeFailed:    return "tokenization failed"
        }
    }
}

/// One loaded model. Contexts are created per observation so the research flags take effect.
final class Engine {
    let model: OpaquePointer
    let vocab: OpaquePointer
    let path: String
    let description: String
    let sizeBytes: UInt64
    let nParams: UInt64

    init(path: String) throws {
        llama_backend_init()
        var mp = llama_model_default_params()
        #if targetEnvironment(simulator)
        mp.n_gpu_layers = 0
        #else
        mp.n_gpu_layers = 99
        #endif
        guard let m = llama_model_load_from_file(path, mp) else { throw EngineError.loadFailed(path) }
        model = m
        vocab = llama_model_get_vocab(m)
        self.path = path
        var buf = [CChar](repeating: 0, count: 256)
        _ = llama_model_desc(m, &buf, 256)
        description = String(cString: buf)
        sizeBytes = llama_model_size(m)
        nParams = llama_model_n_params(m)
    }

    deinit {
        llama_model_free(model)
        llama_backend_free()
    }

    /// Weight type of the model's projections, read from llama_model_desc ("qwen35 27B PTQ1_0 - ...").
    var weightType: String {
        for t in ["PTQ1_0", "PQ2_0", "Q1_0"] where description.contains(t) { return t }
        return "other"
    }

    func tokenize(_ text: String, addSpecial: Bool) throws -> [llama_token] {
        let cap = Int32(text.utf8.count + 16)
        var out = [llama_token](repeating: 0, count: Int(cap))
        let n = llama_tokenize(vocab, text, Int32(text.utf8.count), &out, cap, addSpecial, true)
        guard n >= 0 else { throw EngineError.tokenizeFailed }
        return Array(out.prefix(Int(n)))
    }

    private func makeContext(nCtx: UInt32) throws -> OpaquePointer {
        var cp = llama_context_default_params()
        cp.n_ctx = nCtx
        cp.n_batch = 512
        cp.n_ubatch = 512
        cp.n_seq_max = 1
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
        let threads = Int32(max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 2)))
        cp.n_threads = threads
        cp.n_threads_batch = threads
        guard let ctx = llama_init_from_model(model, cp) else { throw EngineError.contextFailed }
        return ctx
    }

    private func decode(_ ctx: OpaquePointer, _ tokens: [llama_token], startPos: Int32, logitsLast: Bool) throws {
        var batch = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(batch) }
        for (i, t) in tokens.enumerated() {
            batch.token[i] = t
            batch.pos[i] = startPos + Int32(i)
            batch.n_seq_id[i] = 1
            batch.seq_id[i]![0] = 0
            batch.logits[i] = (logitsLast && i == tokens.count - 1) ? 1 : 0
        }
        batch.n_tokens = Int32(tokens.count)
        let rc = llama_decode(ctx, batch)
        guard rc == 0 else { throw EngineError.decodeFailed(rc) }
    }

    struct Generation {
        var promptTokens: Int
        var promptSeconds: Double
        var generated: [llama_token]
        var generateSeconds: Double
        var tokensPerSecond: Double { generateSeconds > 0 ? Double(max(generated.count - 1, 0)) / generateSeconds : 0 }
    }

    /// Greedy generation of n tokens after a chat prompt, on a fresh context. Generation time excludes
    /// the first token (it comes from the prompt pass), like llama-server's timings.
    func generate(prompt: String, n: Int, warmup: Int = 16) throws -> Generation {
        let ctx = try makeContext(nCtx: 2048)
        defer { llama_free(ctx) }
        let sampler = llama_sampler_init_greedy()!
        defer { llama_sampler_free(sampler) }
        let promptTokens = try tokenize(prompt, addSpecial: true)

        // warmup: a short throwaway generation compiles pipelines and fills caches
        try decode(ctx, promptTokens, startPos: 0, logitsLast: true)
        var pos = Int32(promptTokens.count)
        for _ in 0..<warmup {
            let t = llama_sampler_sample(sampler, ctx, -1)
            try decode(ctx, [t], startPos: pos, logitsLast: true)
            pos += 1
        }
        llama_synchronize(ctx)
        llama_memory_clear(llama_get_memory(ctx), true)

        let t0 = DispatchTime.now().uptimeNanoseconds
        try decode(ctx, promptTokens, startPos: 0, logitsLast: true)
        llama_synchronize(ctx)
        let t1 = DispatchTime.now().uptimeNanoseconds
        pos = Int32(promptTokens.count)
        var out: [llama_token] = []
        var tFirst: UInt64 = 0
        for i in 0..<n {
            let t = llama_sampler_sample(sampler, ctx, -1)
            out.append(t)
            if llama_vocab_is_eog(vocab, t) { break }
            if i == 0 { tFirst = DispatchTime.now().uptimeNanoseconds }
            try decode(ctx, [t], startPos: pos, logitsLast: true)
            pos += 1
        }
        llama_synchronize(ctx)
        let t2 = DispatchTime.now().uptimeNanoseconds
        return Generation(promptTokens: promptTokens.count,
                          promptSeconds: Double(t1 - t0) / 1e9,
                          generated: out,
                          generateSeconds: tFirst > 0 ? Double(t2 - tFirst) / 1e9 : 0)
    }

    /// Batch-of-k processing rate (llama-bench ppK): k tokens per decode call, repeated `reps` times on a
    /// fresh context after one warmup call; tokens/s over the timed calls.
    func batchRate(k: Int, reps: Int) throws -> Double {
        let ctx = try makeContext(nCtx: UInt32(max(2048, k + 64)))
        defer { llama_free(ctx) }
        let n = llama_vocab_n_tokens(vocab)
        var rng = SystemRandomNumberGenerator()
        let tokens = (0..<k).map { _ in llama_token(Int32.random(in: 100..<min(n, 30000), using: &rng)) }
        try decode(ctx, tokens, startPos: 0, logitsLast: true)
        llama_synchronize(ctx)
        var total: Double = 0
        for _ in 0..<reps {
            llama_memory_clear(llama_get_memory(ctx), true)
            let t0 = DispatchTime.now().uptimeNanoseconds
            try decode(ctx, tokens, startPos: 0, logitsLast: true)
            llama_synchronize(ctx)
            total += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        }
        return Double(k * reps) / total
    }
}
