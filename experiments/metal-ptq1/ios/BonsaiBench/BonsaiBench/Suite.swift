import Foundation

/// One study of a suite: a model file, two arms by preset name, cells and protocol.
struct StudySpec: Hashable {
    var title: String
    var model: String
    var a: String
    var b: String
    var cells: [String]
    var cycles: Int
    var cooldown: Double
    var ubatch = 512
    var waitForNominal = false
    var thermalWaitLimit = 300.0     // seconds a run waits for the thermal gate before its quartet is rejected
    var gate = 1.20
    /// A diagnostic screen (Diagnose.swift) instead of an A-B-B-A study: `a`/`b`/`cells`/`cycles` are unused.
    var screen: ScreenSpec? = nil
    var attempts = 3                 // A-B-B-A: attempts per quartet
    var weightsInMemory = false      // load the model into app memory instead of memory-mapping it

    /// One line for the lists: what is compared and how.
    var detail: String {
        if let screen {
            return "\(screen.configs.map(\.name).joined(separator: "; ")) · \(screen.cells.joined(separator: ", ")) · \(screen.rounds) rounds, \(Int(cooldown)) s cooldown" +
                (weightsInMemory ? " · weights in app memory" : "")
        }
        return "B = \(b) vs A = \(a) · \(cells.joined(separator: ", ")) · \(cycles) quartets, \(Int(cooldown)) s cooldown" +
            (ubatch != 512 ? " · micro-batch \(ubatch)" : "")
    }

    /// Rough duration: cycles x 4 runs x (cooldown + run time) per cell, from tokens/s measured on an iPhone
    /// 17 Pro Max. Thermal waits and retried quartets come on top.
    var estimatedSeconds: Double {
        // a run: measuring plus the thermal gate's checks; pp512 takes ~65 s on the phone (overnight suite)
        let ppRun = { (k: Int) -> Double in k >= 256 ? 65 : 20 }
        if let screen {
            let runs = Double(screen.rounds * screen.configs.count)
            return runs * screen.cells.reduce(0.0) { $0 + cooldown + ppRun(Cell(name: $1).count) }
        }
        let genRate = model.contains("PTQ1_0") ? 6.0 : model.contains("Q1_0") ? 12.0 : 8.0
        // upstream + MTP verifies two tokens a step on upstream's slow multi-column path (phone PTQ1 pp2:
        // 2.9 tok/s), about 0.7 s a step, ~70 steps; our flags + MTP about 8 tok/s
        let genRun = { (arm: String) -> Double in
            arm == "upstream + MTP" ? 60 : arm.hasSuffix("+ MTP") ? 20 : 0
        }
        return cells.reduce(0.0) { sum, name in
            let cell = Cell(name: name)
            let run: Double
            switch cell.kind {
            case "gen" where genRun(a) + genRun(b) > 0:
                let plain = Double(cell.count + 16) / genRate + 4
                run = ((genRun(a) > 0 ? genRun(a) : plain) + (genRun(b) > 0 ? genRun(b) : plain)) / 2
            case "tg", "chat", "gen": run = Double(cell.count + 16) / genRate + 4
            default: run = ppRun(cell.count)
            }
            return sum + Double(cycles * 4) * (cooldown + run)
        }
    }
}

enum Suites {
    static let ptq1 = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    static let q1 = "Bonsai-27B-Q1_0.gguf"
    static let ptq1mtp = "Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"

    /// What "Run everything we still need" runs: the diagnostics for the two open phone issues (2026-09-25).
    /// Unattended (overnight) protocol: every run starts only at nominal, and the app waits up to an hour for it
    /// instead of rejecting the quartet after 5 minutes, so heat costs time rather than results.
    static let phone: [StudySpec] = diagnostics.map(overnight)

    /// The benchmark suite, completed overnight on 2026-09-25 (ios/results/overnight-2026-09-25); put it back in
    /// `phone` to run it again.
    static let benchmark: [StudySpec] = benchmarkStudies.map(overnight)

    private static func overnight(_ spec: StudySpec) -> StudySpec {
        var s = spec
        s.waitForNominal = true
        s.thermalWaitLimit = 3600
        return s
    }

    static let ptq1Stack: [String: String] = Presets.recommended(for: "PTQ1_0").flags

    /// Round 2 (after the 2026-09-25 diagnostics, ios/results/diagnostics-2026-09-25): the library now splits
    /// graphs into 4 command buffers on iOS (0 of 10 pp512 runs failed that way, 10 of 50 with one long one), so
    /// every run uses that unless it says otherwise. The per-op profile traced the stack's 0.938x at pp512 to
    /// SMALLM_MM (48-row BF16 projections 3.5x slower) and rows mode (delta-net +18%); both take a width cap now.
    /// 1. Do the caps recover long prompts without hurting short ones? Upstream, the stack, and the stack with
    ///    SMALLM_MM up to 64 columns and rows mode up to 8 tokens, at widths where the caps change routing
    ///    (16: rows capped; 128: SMALLM_MM capped; 512: both).
    /// 2. What 4 command buffers cost where graphs are small: 1 against 4 at 1 token (decode) and 8 tokens.
    /// 3. Where each flag stops paying: every op timed alone at 2-512 tokens, upstream and the two flags (judge
    ///    rows mode on GATED_DELTA_NET plus the GET_ROWS/CPY/SET_ROWS it removes or adds).
    private static let capped: [String: String] = ptq1Stack.merging(
        ["GGML_METAL_SMALLM_MM_MAX_N": "64", "GGML_GDN_ROWS_PLAIN_MAX_TOKENS": "8"]) { $1 }

    private static let diagnostics: [StudySpec] = [
        StudySpec(title: "PTQ1_0 prompts: the stack with batch-width caps", model: ptq1, a: "-", b: "-", cells: [],
                  cycles: 0, cooldown: 60,
                  screen: ScreenSpec(configs: [
                      ScreenConfig(name: "upstream", flags: [:]),
                      ScreenConfig(name: "stack", flags: ptq1Stack),
                      ScreenConfig(name: "stack, SMALLM_MM <= 64 columns, rows mode <= 8 tokens", flags: capped),
                  ], cells: ["pp512", "pp128", "pp16"], rounds: 6, reference: "upstream", retryFailed: true)),
        StudySpec(title: "What 4 command buffers cost on small graphs (decode, 8 tokens)", model: ptq1, a: "-", b: "-",
                  cells: [], cycles: 0, cooldown: 30,
                  screen: ScreenSpec(configs: [
                      ScreenConfig(name: "1 command buffer", flags: ["GGML_METAL_N_CB": "1"]),
                      ScreenConfig(name: "4 command buffers (iOS default)", flags: [:]),
                  ], cells: ["pp1", "pp8"], rounds: 6, reference: "1 command buffer", retryFailed: true)),
        StudySpec(title: "Per-op GPU time by prompt width: upstream and SMALLM_MM + rows mode", model: ptq1, a: "-", b: "-",
                  cells: [], cycles: 0, cooldown: 40,
                  screen: ScreenSpec(configs: [
                      ScreenConfig(name: "upstream, ops profiled", flags: [:], profileOps: true, cbStats: false),
                      ScreenConfig(name: "SMALLM_MM + rows mode, ops profiled",
                                   flags: ["GGML_METAL_SMALLM_MM": "1", "GGML_GDN_ROWS_PLAIN": "1"], profileOps: true, cbStats: false),
                  ], cells: ["pp2", "pp8", "pp32", "pp128", "pp512"], rounds: 2, reference: "upstream, ops profiled")),
    ] + memoryStudies

    /// 5. iOS evicts the memory-mapped weights (up to 6 GB re-read from flash in one run, 2026-09-25 diagnostics):
    ///    the same runs with the weights memory-mapped and read into app memory, alternating (A B A B, the model
    ///    reloaded each time), with 4 command buffers so GPU errors stay out of it. If the in-memory model does not
    ///    fit, the app is stopped while loading and the suite marks the model's remaining studies "did not fit".
    private static let memoryStudies: [StudySpec] = [false, true, false, true].enumerated().map { i, inMemory in
        StudySpec(title: "Weights " + (inMemory ? "in app memory" : "memory-mapped") + " (\(i / 2 + 1) of 2)", model: ptq1,
                  a: "-", b: "-", cells: [], cycles: 0, cooldown: 60,
                  screen: ScreenSpec(configs: [ScreenConfig(name: "upstream, 4 command buffers", flags: ["GGML_METAL_N_CB": "4"])],
                                     cells: ["pp512"], rounds: 4, reference: "upstream, 4 command buffers"),
                  weightsInMemory: inMemory)
    }

    private static let benchmarkStudies: [StudySpec] = [
        StudySpec(title: "Q1_0 prompts: the new K32 prefill kernel vs our Q1 stack", model: q1, a: "M5 stack (Q1)",
                  b: "M5 stack (Q1) + K32 prefill", cells: ["pp512", "pp128"], cycles: 3, cooldown: 40),
        StudySpec(title: "Is rows mode what slows 512-token prompts on PTQ1_0?", model: ptq1, a: "upstream",
                  b: "only GDN_ROWS_PLAIN", cells: ["pp512"], cycles: 3, cooldown: 60),
        StudySpec(title: "Bonsai 2 PTQ1_0 + MTP: upstream plain vs our flags + MTP (the headline)", model: ptq1mtp,
                  a: "upstream", b: "M5 stack (PTQ1) + MTP", cells: ["gen128"], cycles: 3, cooldown: 60),
        StudySpec(title: "Bonsai 2 PTQ1_0: generation", model: ptq1, a: "upstream", b: "M5 stack (PTQ1)",
                  cells: ["tg128", "chat128"], cycles: 3, cooldown: 60),
        StudySpec(title: "Bonsai 2 PTQ1_0: small batches", model: ptq1, a: "upstream", b: "M5 stack (PTQ1)",
                  cells: ["pp2", "pp4", "pp8"], cycles: 3, cooldown: 40),
        StudySpec(title: "Bonsai 2 PTQ1_0: MTP vs MTP (upstream + MTP vs our flags + MTP)", model: ptq1mtp,
                  a: "upstream + MTP", b: "M5 stack (PTQ1) + MTP", cells: ["gen128"], cycles: 3, cooldown: 90),
        StudySpec(title: "Q1_0: what our flags add to PrismML's popcount", model: q1,
                  a: "PrismML popcount (their option)", b: "M5 stack (Q1) + PrismML popcount",
                  cells: ["tg128", "pp2", "pp4", "pp8"], cycles: 2, cooldown: 40),
        StudySpec(title: "Bonsai 1 ternary PQ2_0 (7.2 GB, also a fit test)", model: "Ternary-Bonsai-27B-PQ2_0.gguf",
                  a: "upstream", b: "M5 stack (PQ2)", cells: ["tg128", "pp2"], cycles: 3, cooldown: 60),
        StudySpec(title: "Bonsai 2 PQ2_0 (7.2 GB, also a fit test)", model: "Ternary-Bonsai-2-27B-PQ2_0.gguf",
                  a: "upstream", b: "M5 stack (PQ2)", cells: ["tg128", "pp2"], cycles: 3, cooldown: 60),
    ]
}

extension Suites {
    /// One-tap single studies: each loads its model and sets everything itself.
    static let quick: [StudySpec] = [
        // M5 Max: pp512 1.07x, pp128 1.08x over the Q1 stack; the phone has not run it yet
        StudySpec(title: "Q1_0 prompts: the new K32 prefill kernel vs our Q1 stack", model: q1, a: "M5 stack (Q1)",
                  b: "M5 stack (Q1) + K32 prefill", cells: ["pp512", "pp128"], cycles: 2, cooldown: 40),
        // SMALLM_MM alone measured 0.975x at pp512 on the phone (2026-09-24), not the stack's ~0.70x; rows
        // mode is the only other stack flag active at 512 tokens (+4% on M5)
        StudySpec(title: "Is rows mode what slows 512-token prompts on PTQ1_0?", model: ptq1, a: "upstream",
                  b: "only GDN_ROWS_PLAIN", cells: ["pp512"], cycles: 2, cooldown: 60),
        StudySpec(title: "Is SMALLM_MM what slows 512-token prompts on PTQ1_0?", model: ptq1, a: "upstream",
                  b: "only SMALLM_MM", cells: ["pp512"], cycles: 2, cooldown: 60),
        StudySpec(title: "PTQ1_0 512-token prompts: our flags vs upstream", model: ptq1, a: "upstream",
                  b: "M5 stack (PTQ1)", cells: ["pp512"], cycles: 2, cooldown: 60),
        StudySpec(title: "PTQ1_0 plain generation (tg128): our flags vs upstream", model: ptq1, a: "upstream",
                  b: "M5 stack (PTQ1)", cells: ["tg128"], cycles: 3, cooldown: 60),
        StudySpec(title: "MTP headline: upstream plain vs our flags + MTP (needs the MTP file)", model: ptq1mtp,
                  a: "upstream", b: "M5 stack (PTQ1) + MTP", cells: ["gen128"], cycles: 3, cooldown: 60),
    ]
}

func durationText(_ seconds: Double) -> String {
    let m = Int((seconds / 60).rounded())
    return m >= 60 ? "\(m / 60) h \(m % 60) min" : "\(m) min"
}
