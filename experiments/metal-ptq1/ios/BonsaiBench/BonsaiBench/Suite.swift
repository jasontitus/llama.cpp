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
    var gate = 1.20

    /// Rough duration: cycles x 4 runs x (cooldown + run time) per cell, from tokens/s measured on an iPhone
    /// 17 Pro Max. Thermal waits and retried quartets come on top.
    var estimatedSeconds: Double {
        let genRate = model.contains("PTQ1_0") ? 6.0 : model.contains("Q1_0") ? 12.0 : 8.0
        return cells.reduce(0.0) { sum, name in
            let cell = Cell(name: name)
            let run: Double
            switch cell.kind {
            case "tg", "chat": run = Double(cell.count + 16) / genRate + 4
            default: run = cell.count >= 256 ? 25 : 6
            }
            return sum + Double(cycles * 4) * (cooldown + run)
        }
    }
}

enum Suites {
    static let ptq1 = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    static let q1 = "Bonsai-27B-Q1_0.gguf"

    /// The iPhone studies still missing from the results table, most valuable first. Every comparison is
    /// upstream vs our flags, except the popcount one, which measures what our flags add on top of
    /// PrismML's own option (their flag is in both arms).
    static let phone: [StudySpec] = [
        StudySpec(title: "Bonsai 2 PTQ1_0: generation", model: ptq1, a: "upstream", b: "M5 stack (PTQ1)",
                  cells: ["tg128", "chat128"], cycles: 3, cooldown: 60),
        StudySpec(title: "Bonsai 2 PTQ1_0: small batches", model: ptq1, a: "upstream", b: "M5 stack (PTQ1)",
                  cells: ["pp2", "pp4", "pp8"], cycles: 3, cooldown: 40),
        StudySpec(title: "Bonsai 1 binary Q1_0: generation and small batches", model: q1, a: "upstream",
                  b: "M5 stack (Q1)", cells: ["chat128", "pp2", "pp4", "pp8"], cycles: 3, cooldown: 40),
        StudySpec(title: "Q1_0: what our flags add to PrismML's popcount", model: q1,
                  a: "PrismML popcount (their option)", b: "M5 stack (Q1) + PrismML popcount",
                  cells: ["tg128", "pp2", "pp4", "pp8"], cycles: 2, cooldown: 40),
        StudySpec(title: "Bonsai 1 ternary PQ2_0 (7.2 GB, also a fit test)", model: "Ternary-Bonsai-27B-PQ2_0.gguf",
                  a: "upstream", b: "M5 stack (PQ2)", cells: ["tg128", "pp2"], cycles: 3, cooldown: 60),
        StudySpec(title: "Bonsai 2 PQ2_0 (7.2 GB, also a fit test)", model: "Ternary-Bonsai-2-27B-PQ2_0.gguf",
                  a: "upstream", b: "M5 stack (PQ2)", cells: ["tg128", "pp2"], cycles: 3, cooldown: 60),
        StudySpec(title: "PTQ1_0 pp512 with a 128-token micro-batch (GPU watchdog test)", model: ptq1,
                  a: "upstream", b: "M5 stack (PTQ1)", cells: ["pp512"], cycles: 2, cooldown: 40, ubatch: 128),
    ]
}

func durationText(_ seconds: Double) -> String {
    let m = Int((seconds / 60).rounded())
    return m >= 60 ? "\(m / 60) h \(m % 60) min" : "\(m) min"
}
