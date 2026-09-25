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

    /// One line for the lists: what is compared and how.
    var detail: String {
        if let screen {
            return "\(screen.configs.map(\.name).joined(separator: "; ")) · \(screen.cells.joined(separator: ", ")) · \(screen.rounds) rounds, \(Int(cooldown)) s cooldown"
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

    private static func stack(without names: [String]) -> [String: String] {
        ptq1Stack.filter { !names.contains($0.key) }
    }

    /// 1. Is PTQ1_0 pp512 slower with our flags at all? The only 0.70x came from quartets rejected for heat.
    /// 2. The pp512 GPU errors. In the earlier studies 14 of 16 failing calls ended 5.6-5.9 s after they started
    ///    (normal PTQ1 calls take 6.4-7.4 s), always in the long command buffer that holds ~90% of the graph: a
    ///    limit of about 5 s on one command buffer is the lead. Shorter command buffers (N_CB) or smaller
    ///    micro-batches should then never fail; a stats-off control checks that recording changes nothing.
    ///    Failure rates are ~11-15% per run, so only pooled zeros over many runs (10 rounds) mean anything.
    /// 3. Where the GPU time goes: every op timed on its own, upstream and stack (rates not comparable).
    /// 4. Which flag, if any, costs time at 512 tokens: the flags that act there left out one by one, the
    ///    three that only act on 2-8-token batches left out together, and upstream against itself as the null.
    private static let diagnostics: [StudySpec] = [
        StudySpec(title: "PTQ1_0 pp512 measured cool: our stack vs upstream", model: ptq1, a: "upstream",
                  b: "M5 stack (PTQ1)", cells: ["pp512"], cycles: 3, cooldown: 60, attempts: 5),
        StudySpec(title: "pp512 GPU errors: shorter command buffers and smaller micro-batches", model: ptq1, a: "-", b: "-",
                  cells: [], cycles: 0, cooldown: 40,
                  screen: ScreenSpec(configs: [
                      ScreenConfig(name: "1 long command buffer (default)", flags: [:]),
                      ScreenConfig(name: "default, stats off", flags: [:], cbStats: false),
                      ScreenConfig(name: "4 command buffers", flags: ["GGML_METAL_N_CB": "4"]),
                      ScreenConfig(name: "micro-batch 256", flags: [:], ubatch: 256),
                  ], cells: ["pp512"], rounds: 10, reference: "1 long command buffer (default)")),
        StudySpec(title: "PTQ1_0 pp512: GPU time of every op, upstream and stack", model: ptq1, a: "-", b: "-", cells: [],
                  cycles: 0, cooldown: 60,
                  screen: ScreenSpec(configs: [
                      ScreenConfig(name: "upstream, ops profiled", flags: [:], profileOps: true, cbStats: false),
                      ScreenConfig(name: "stack, ops profiled", flags: ptq1Stack, profileOps: true, cbStats: false),
                  ], cells: ["pp512"], rounds: 2, reference: "upstream, ops profiled")),
        StudySpec(title: "PTQ1_0 pp512: the stack with each flag left out", model: ptq1, a: "-", b: "-", cells: [],
                  cycles: 0, cooldown: 60,
                  screen: ScreenSpec(configs: [
                      ScreenConfig(name: "upstream", flags: [:]),
                      ScreenConfig(name: "upstream again", flags: [:]),
                      ScreenConfig(name: "stack", flags: ptq1Stack),
                      ScreenConfig(name: "stack without SMALLM_MM", flags: stack(without: ["GGML_METAL_SMALLM_MM"])),
                      ScreenConfig(name: "stack without rows mode", flags: stack(without: ["GGML_GDN_ROWS_PLAIN"])),
                      ScreenConfig(name: "stack without MULTICOL, GLU and STAGE",
                                   flags: stack(without: ["GGML_METAL_PTQ1_MULTICOL", "GGML_METAL_PTQ1_GLU", "GGML_METAL_PTQ1_STAGE"])),
                  ], cells: ["pp512"], rounds: 4, reference: "upstream", retryFailed: true)),
    ]

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
