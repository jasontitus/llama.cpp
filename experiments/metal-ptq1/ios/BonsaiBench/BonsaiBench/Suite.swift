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

    /// Rough duration: cycles x 4 runs x (cooldown + run time) per cell, from tokens/s measured on an iPhone
    /// 17 Pro Max. Thermal waits and retried quartets come on top.
    var estimatedSeconds: Double {
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
            default: run = cell.count >= 256 ? 25 : 6
            }
            return sum + Double(cycles * 4) * (cooldown + run)
        }
    }
}

enum Suites {
    static let ptq1 = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    static let q1 = "Bonsai-27B-Q1_0.gguf"
    static let ptq1mtp = "Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"

    /// The iPhone studies still missing from the results table, most valuable first (2026-09-25): the new Q1
    /// prompt kernel, the PTQ1 pp512 open issue, then completing the preliminary cells. Every comparison is
    /// upstream (or our stack) vs our flags, except the popcount one, which measures what our flags add on top
    /// of PrismML's own option (their flag is in both arms). Studies whose model is not in the app are skipped.
    /// Unattended (overnight) protocol: every run starts only at nominal, and the app waits up to an hour for it
    /// instead of rejecting the quartet after 5 minutes, so heat costs time rather than results.
    static let phone: [StudySpec] = phoneStudies.map { spec in
        var s = spec
        s.waitForNominal = true
        s.thermalWaitLimit = 3600
        return s
    }

    private static let phoneStudies: [StudySpec] = [
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
