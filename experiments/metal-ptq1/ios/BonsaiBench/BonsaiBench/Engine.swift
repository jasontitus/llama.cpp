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

/// GGML_* / LLAMA_* variables the app was launched with (an Xcode scheme, devicectl). Recorded with every
/// result and cleared before every observation, so they cannot leak into either arm. Read it once before the
/// first applyFlags (globals are initialized lazily).
let launchEnvironment: [String: String] = ProcessInfo.processInfo.environment.filter { isBackendVariable($0.key) }

func isBackendVariable(_ name: String) -> Bool { name.hasPrefix("GGML_") || name.hasPrefix("LLAMA_") }

func applyFlags(_ flags: [String: String]) {
    for name in Set(researchFlagNames + ProcessInfo.processInfo.environment.keys.filter(isBackendVariable)) { unsetenv(name) }
    for (k, v) in flags { setenv(k, v, 1) }
}

/// The library's warnings and errors (e.g. Metal's reason for a failed command buffer), collected per
/// observation. Everything is also written to stderr, which `devicectl device process launch --console`
/// shows.
final class LibraryLog {
    static let shared = LibraryLog()
    private let lock = NSLock()
    private var lines: [String] = []
    private var keeping = false          // the current message is a warning or error (for CONT pieces)

    func install() {
        llama_log_set({ level, text, _ in
            guard let text else { return }
            fputs(text, stderr)
            LibraryLog.shared.add(level: level, String(cString: text))
        }, nil)
    }

    private func add(level: ggml_log_level, _ text: String) {
        lock.lock(); defer { lock.unlock() }
        if level != GGML_LOG_LEVEL_CONT { keeping = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR }
        guard keeping else { return }
        if level == GGML_LOG_LEVEL_CONT, let last = lines.popLast() { lines.append(last + text) } else { lines.append(text) }
        if lines.count > 200 { lines.removeFirst(lines.count - 200) }
    }

    /// Lines collected since the last call, trimmed.
    func drain() -> [String] {
        lock.lock(); defer { lock.unlock() }
        defer { lines = [] }
        return lines.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }
}

/// Memory seen while a context is alive (the per-context buffers are gone once it is freed).
struct Probe: Codable {
    var footprintBytes: UInt64
    var availableBytes: UInt64

    static func now() -> Probe { Probe(footprintBytes: physicalFootprint(), availableBytes: UInt64(max(0, availableMemory()))) }
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
        LibraryLog.shared.install()
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

    /// ubatch: the largest graph llama.cpp builds for a batch. Metal runs most of a graph as one command
    /// buffer, so a 512-token ubatch is several seconds of GPU work on a phone; a smaller one splits it.
    private func makeContext(nCtx: UInt32, ubatch: UInt32 = 512) throws -> OpaquePointer {
        var cp = llama_context_default_params()
        cp.n_ctx = nCtx
        // Only the last token's logits are ever read. The default (n_batch rows) would reserve
        // 512 x 248k floats (~0.5 GB) of compute buffer that a phone cannot spare.
        cp.n_outputs_max = 1
        cp.n_batch = 512
        cp.n_ubatch = ubatch
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

    private func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    /// A failed Metal command buffer only marks the backend; the next decode reports it. One more decode
    /// after the timed work makes a failure there an error instead of a (fast) measurement.
    private func checkBackend(_ ctx: OpaquePointer, token: llama_token, pos: Int32) throws {
        try decode(ctx, [token], startPos: pos, logitsLast: true)
        llama_synchronize(ctx)
    }

    private func randomTokens(_ k: Int) -> [llama_token] {
        let n = llama_vocab_n_tokens(vocab)
        var rng = SystemRandomNumberGenerator()
        return (0..<k).map { _ in llama_token(Int32.random(in: 100..<min(n, 30000), using: &rng)) }
    }

    struct Generation {
        var promptTokens: Int
        var promptSeconds: Double
        var generated: [llama_token]
        var decodes: Int                  // timed single-token decodes (each followed by greedy sampling)
        var generateSeconds: Double
        var probe: Probe
        var tokensPerSecond: Double { generateSeconds > 0 ? Double(decodes) / generateSeconds : 0 }
    }

    /// "chat128": greedy generation of up to n tokens after a chat prompt, on a fresh context, returning the
    /// tokens so arms can be compared. The rate covers the decodes after the first token (which comes from
    /// the prompt pass) including greedy sampling, like llama-server's generation timing.
    func generate(prompt: String, n: Int, warmup: Int = 16) throws -> Generation {
        let ctx = try makeContext(nCtx: 1024)
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
        llama_sampler_reset(sampler)

        let t0 = now()
        try decode(ctx, promptTokens, startPos: 0, logitsLast: true)
        llama_synchronize(ctx)
        let t1 = now()
        pos = Int32(promptTokens.count)
        var t = llama_sampler_sample(sampler, ctx, -1)
        var out = [t]
        var decodes = 0
        let tStart = now()
        while out.count < n && !llama_vocab_is_eog(vocab, t) {
            try decode(ctx, [t], startPos: pos, logitsLast: true)
            pos += 1
            decodes += 1
            t = llama_sampler_sample(sampler, ctx, -1)   // waits for the logits
            out.append(t)
        }
        let t2 = now()
        let probe = Probe.now()
        try checkBackend(ctx, token: t, pos: pos)
        return Generation(promptTokens: promptTokens.count, promptSeconds: Double(t1 - t0) / 1e9, generated: out,
                          decodes: decodes, generateSeconds: Double(t2 - tStart) / 1e9, probe: probe)
    }

    /// "tgN" as llama-bench measures it: N single-token decodes from an empty context, random tokens, no
    /// sampling; tokens/s.
    func generationRate(n: Int, warmup: Int = 16) throws -> (rate: Double, probe: Probe) {
        let ctx = try makeContext(nCtx: 1024)
        defer { llama_free(ctx) }
        let tokens = randomTokens(n + 1)
        for i in 0..<warmup { try decode(ctx, [tokens[i % n]], startPos: Int32(i), logitsLast: true) }
        llama_synchronize(ctx)
        llama_memory_clear(llama_get_memory(ctx), true)
        let t0 = now()
        for i in 0..<n { try decode(ctx, [tokens[i]], startPos: Int32(i), logitsLast: true) }
        llama_synchronize(ctx)
        let secs = Double(now() - t0) / 1e9
        let probe = Probe.now()
        try checkBackend(ctx, token: tokens[n], pos: Int32(n))
        return (Double(n) / secs, probe)
    }

    /// "ppK": k tokens per decode call (llama-bench ppK; the step shape of MTP verification and concurrent
    /// requests), on a fresh context. One decode of a small batch is ~0.1 s on a phone, so both the warmup
    /// (which also brings the GPU clocks up) and the measurement run for a minimum time, not a count.
    func batchRate(k: Int, ubatch: Int = 512, warmupSeconds: Double = 1, minSeconds: Double = 2.5,
                   minReps: Int = 2) throws -> (rate: Double, probe: Probe, calls: [Double]) {
        let ctx = try makeContext(nCtx: UInt32(max(1024, k + 64)), ubatch: UInt32(ubatch))
        defer { llama_free(ctx) }
        let tokens = randomTokens(k + 1)
        let batch = Array(tokens.prefix(k))
        func once() throws -> Double {
            llama_memory_clear(llama_get_memory(ctx), true)
            let t0 = now()
            try decode(ctx, batch, startPos: 0, logitsLast: true)
            llama_synchronize(ctx)
            return Double(now() - t0) / 1e9
        }
        var warm = 0.0
        repeat { warm += try once() } while warm < warmupSeconds
        var calls: [Double] = []
        while calls.count < minReps || calls.reduce(0, +) < minSeconds {
            calls.append(try once())
        }
        let probe = Probe.now()
        try checkBackend(ctx, token: tokens[k], pos: Int32(k))
        return (Double(k * calls.count) / calls.reduce(0, +), probe, calls)
    }
}
