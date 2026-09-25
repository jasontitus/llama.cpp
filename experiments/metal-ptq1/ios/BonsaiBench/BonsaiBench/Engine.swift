import Foundation
import llama

/// Research flags understood by the patched Metal backend. They are environment variables that the
/// backend re-reads after all prior Metal contexts are freed, so each observation sets them, then creates
/// a fresh context.
let researchFlagNames: [String] = [
    "GGML_METAL_PTQ1_MULTICOL", "GGML_METAL_PTQ1_MULTICOL_MAX", "GGML_METAL_PTQ1_GLU", "GGML_METAL_PTQ1_STAGE",
    "GGML_METAL_PTQ1_TENSOR", "GGML_METAL_PTQ1_TENSOR_MIN", "GGML_GDN_ROWS_PLAIN", "GGML_METAL_SMALLM",
    "GGML_METAL_SMALLM_MM", "GGML_METAL_PQ2_MULTICOL", "GGML_METAL_PQ2_GLU", "GGML_METAL_Q1_GLU",
    "GGML_METAL_BATCH_INVARIANT", "GGML_METAL_Q1_0_POPCNT", "GGML_METAL_Q1_MM_K32_ALIGNED", "GGML_METAL_Q1_SWIZZLE_LOG",
    "GGML_METAL_N_CB", "GGML_METAL_CB_STATS", "GGML_METAL_PROFILE_OPS",
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
/// observation, plus the prefill (mul_mm) kernels it compiles, as "pipeline: kernel_mul_mm_...": a kernel is
/// compiled once per app process, so it shows in the first observation that uses it. Everything is also
/// written to stderr, which `devicectl device process launch --console` shows.
final class LibraryLog {
    static let shared = LibraryLog()
    private let lock = NSLock()
    private var lines: [String] = []
    private var keeping = false          // the current message is a warning or error (for CONT pieces)

    func install() {
        lock.lock()
        if !commonLogInstalled {
            commonLogInstalled = true
            try? FileManager.default.removeItem(at: Self.commonLogURL)
            bb_common_log_to_file(Self.commonLogURL.path)
        }
        lock.unlock()
        llama_log_set({ level, text, _ in
            guard let text else { return }
            fputs(text, stderr)
            LibraryLog.shared.add(level: level, String(cString: text))
        }, nil)
    }

    private func add(level: ggml_log_level, _ text: String) {
        lock.lock(); defer { lock.unlock() }
        if level != GGML_LOG_LEVEL_CONT { keeping = level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR }
        // "loaded <kernel>_bci=..." is logged once a pipeline exists (a failed compile logs no "loaded")
        if level == GGML_LOG_LEVEL_DEBUG, let r = text.range(of: "loaded kernel_mul_mm_") {
            let name = text[text.index(r.lowerBound, offsetBy: 7)...]
            let base = name.range(of: "_bci=").map { name[..<$0.lowerBound] } ?? name.prefix { $0 != " " }
            lines.append("pipeline: \(base)")
        }
        guard keeping else { return }
        if level == GGML_LOG_LEVEL_CONT, let last = lines.popLast() { lines.append(last + text) } else { lines.append(text) }
        if lines.count > 200 { lines.removeFirst(lines.count - 200) }
    }

    /// Where llama.cpp's common code (MTP) logs; its warnings and errors are read into the same lines.
    static let commonLogURL = FileManager.default.temporaryDirectory.appendingPathComponent("llama-common.log")
    private var commonLogOffset: UInt64 = 0
    private var commonLogInstalled = false

    /// Append common's warnings and errors written since the last call (its log prefixes lines "W "/"E ").
    func readCommonLog() {
        lock.lock(); defer { lock.unlock() }
        guard let h = try? FileHandle(forReadingFrom: Self.commonLogURL) else { return }
        defer { try? h.close() }
        try? h.seek(toOffset: commonLogOffset)
        guard let data = try? h.readToEnd() else { return }
        commonLogOffset += UInt64(data.count)
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n") where line.hasPrefix("W ") || line.hasPrefix("E ") {
            lines.append("common: " + line)
        }
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
    case loadFailed(String), contextFailed, decodeFailed(Int32), tokenizeFailed, generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .loadFailed(let p): return "could not load model at \(p)"
        case .contextFailed:     return "could not create a context (out of memory?)"
        case .decodeFailed(let c): return "llama_decode failed (\(c))"
        case .tokenizeFailed:    return "tokenization failed"
        case .generationFailed(let m): return m
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
    let hasMTP: Bool                      // loaded with its multi-token-prediction layers (an MTP GGUF)
    /// The weights were read into the app's memory instead of memory-mapped from the file: iOS cannot evict
    /// them under memory pressure (memory-mapped weights are re-read from flash after an eviction), but they
    /// count against the app's memory limit.
    let weightsInMemory: Bool

    init(path: String, weightsInMemory: Bool = false) throws {
        LibraryLog.shared.install()
        llama_backend_init()
        var mp = llama_model_default_params()
        #if targetEnvironment(simulator)
        mp.n_gpu_layers = 0
        #else
        mp.n_gpu_layers = 99
        #endif
        // A grafted MTP GGUF ("...-mtp.gguf") carries the MTP head; load it so gen cells can draft with it.
        mp.load_mtp = (path as NSString).lastPathComponent.lowercased().contains("-mtp")
        if weightsInMemory { mp.load_mode = LLAMA_LOAD_MODE_NONE }   // read into backend buffers; default: memory-mapped
        self.weightsInMemory = weightsInMemory
        guard let m = llama_model_load_from_file(path, mp) else { throw EngineError.loadFailed(path) }
        model = m
        vocab = llama_model_get_vocab(m)
        self.path = path
        var buf = [CChar](repeating: 0, count: 256)
        _ = llama_model_desc(m, &buf, 256)
        description = String(cString: buf)
        sizeBytes = llama_model_size(m)
        nParams = llama_model_n_params(m)
        hasMTP = mp.load_mtp && llama_model_n_layer_nextn(m) > 0
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

    struct Speculative {
        var promptTokens: Int
        var promptSeconds: Double
        var generated: [llama_token]
        var generateSeconds: Double
        var drafted: Int
        var accepted: Int
        var steps: Int
        var draftSeconds: Double
        var verifySeconds: Double
        var processSeconds: Double
        var probe: Probe
        /// Generated tokens per second of the generation loop (what llama-server reports as predicted/s).
        var tokensPerSecond: Double { generateSeconds > 0 ? Double(generated.count) / generateSeconds : 0 }
    }

    /// "genN": greedy generation of up to n tokens after a chat prompt through llama.cpp's common
    /// speculative-decoding loop (MTP/BonsaiMTP.cpp), as llama-server runs it: with `draft` MTP draft tokens
    /// per step, or plain decoding (draft 0) through the same loop, so both arms are timed identically.
    func speculativeGenerate(prompt: String, n: Int, draft: Int, warmup: Int = 16) throws -> Speculative {
        // MTP tensors are loaded only for "-mtp" files; drafting without them would abort in llama.cpp
        if draft > 0 && !hasMTP { throw EngineError.generationFailed("this model was not loaded with MTP layers (use an -mtp GGUF)") }
        let threads = Int32(max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 2)))
        var tokens = [Int32](repeating: 0, count: n + draft + 1)
        var r = bb_gen_result()
        let count = bb_generate(model, hasMTP ? 1 : 0, prompt, Int32(n), Int32(draft), Int32(warmup), 1024, threads, &tokens,
                                Int32(tokens.count), &r)
        guard count >= 0 else {
            let msg = withUnsafeBytes(of: r.error) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
            throw EngineError.generationFailed(msg)
        }
        return Speculative(promptTokens: Int(r.n_prompt), promptSeconds: r.prompt_seconds,
                           generated: Array(tokens.prefix(Int(count))), generateSeconds: r.generate_seconds,
                           drafted: Int(r.n_drafted), accepted: Int(r.n_accepted), steps: Int(r.n_steps),
                           draftSeconds: r.draft_seconds, verifySeconds: r.verify_seconds, processSeconds: r.process_seconds,
                           probe: Probe(footprintBytes: r.footprint_bytes, availableBytes: r.available_bytes))
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
    /// Each call's time, warmup included, kept by the caller so that it survives a failed call.
    final class CallTimes {
        var warmup: [Double] = []
        var timed: [Double] = []
        // bytes paged in system-wide during each call (-1 if unknown): the model's memory-mapped weights read
        // back from flash after iOS evicted them
        var warmupPageinBytes: [Int64] = []
        var timedPageinBytes: [Int64] = []
    }

    /// A warmup call that pages in more than this was still reading the weights back from flash.
    static let residentPageinLimit: Int64 = 64 << 20

    func batchRate(k: Int, ubatch: Int = 512, warmupSeconds: Double = 1, minSeconds: Double = 2.5,
                   minReps: Int = 2, maxWarmups: Int = 4,
                   calls log: CallTimes = CallTimes()) throws -> (rate: Double, probe: Probe) {
        let ctx = try makeContext(nCtx: UInt32(max(1024, k + 64)), ubatch: UInt32(ubatch))
        defer { llama_free(ctx) }
        let tokens = randomTokens(k + 1)
        let batch = Array(tokens.prefix(k))
        func once() throws -> (seconds: Double, pageins: Int64) {
            llama_memory_clear(llama_get_memory(ctx), true)
            let m0 = MemoryCounters.now()
            let t0 = now()
            try decode(ctx, batch, startPos: 0, logitsLast: true)
            llama_synchronize(ctx)
            let seconds = Double(now() - t0) / 1e9
            guard let m0, let m1 = MemoryCounters.now() else { return (seconds, -1) }
            return (seconds, Int64(bitPattern: (m1.systemPageins &- m0.systemPageins) &* m1.pageSize))
        }
        // Warm up for warmupSeconds, then on while a call still reads more than residentPageinLimit from flash
        // (at most maxWarmups calls): iOS evicts the memory-mapped weights while the app waits for the phone to
        // cool, and re-reading them (up to the whole model) would otherwise land in the timed calls.
        repeat {
            let c = try once()
            log.warmup.append(c.seconds)
            log.warmupPageinBytes.append(c.pageins)
        } while log.warmup.reduce(0, +) < warmupSeconds ||
                ((log.warmupPageinBytes.last ?? 0) > Self.residentPageinLimit && log.warmup.count < maxWarmups)
        while log.timed.count < minReps || log.timed.reduce(0, +) < minSeconds {
            let c = try once()
            log.timed.append(c.seconds)
            log.timedPageinBytes.append(c.pageins)
        }
        let probe = Probe.now()
        try checkBackend(ctx, token: tokens[k], pos: Int32(k))
        return (Double(k * log.timed.count) / log.timed.reduce(0, +), probe)
    }
}
