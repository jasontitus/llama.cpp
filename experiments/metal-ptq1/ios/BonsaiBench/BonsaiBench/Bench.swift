import Foundation

/// A named set of research flags (one benchmark arm).
struct Arm: Codable, Hashable, Identifiable {
    var name: String
    var flags: [String: String]
    var id: String { name }
}

enum Presets {
    static let upstream = Arm(name: "upstream", flags: [:])
    static let bitExact = Arm(name: "bit-exact (rows mode)", flags: ["GGML_GDN_ROWS_PLAIN": "1"])

    /// The recommended configuration for a weight type (see experiments/metal-ptq1/m5/README.md).
    static func recommended(for weightType: String) -> Arm {
        switch weightType {
        case "PTQ1_0":
            return Arm(name: "M5 stack (PTQ1)", flags: [
                "GGML_METAL_PTQ1_MULTICOL": "1", "GGML_METAL_PTQ1_MULTICOL_MAX": "8", "GGML_METAL_PTQ1_GLU": "1",
                "GGML_METAL_PTQ1_STAGE": "1", "GGML_GDN_ROWS_PLAIN": "1", "GGML_METAL_SMALLM_MM": "1"])
        case "PQ2_0":
            return Arm(name: "M5 stack (PQ2)", flags: [
                "GGML_METAL_PQ2_MULTICOL": "1", "GGML_METAL_PQ2_GLU": "1", "GGML_GDN_ROWS_PLAIN": "1",
                "GGML_METAL_SMALLM": "1", "GGML_METAL_SMALLM_MM": "1"])
        case "Q1_0":
            return Arm(name: "M5 stack (Q1)", flags: [
                "GGML_GDN_ROWS_PLAIN": "1", "GGML_METAL_SMALLM": "1", "GGML_METAL_SMALLM_MM": "1"])
        default:
            return Arm(name: "M5 stack (generic)", flags: ["GGML_GDN_ROWS_PLAIN": "1"])
        }
    }

    static func invariant(for weightType: String) -> Arm {
        var a = recommended(for: weightType)
        a.name += " + invariant"
        a.flags["GGML_METAL_BATCH_INVARIANT"] = "1"
        return a
    }

    static func tensor(for weightType: String) -> Arm {
        var a = recommended(for: weightType)
        a.name += " + tensor"
        a.flags["GGML_METAL_PTQ1_TENSOR"] = "1"
        return a
    }

    static func all(for weightType: String) -> [Arm] {
        var arms = [upstream, bitExact, recommended(for: weightType), invariant(for: weightType)]
        if weightType == "PTQ1_0" { arms.append(tensor(for: weightType)) }
        if weightType == "Q1_0" {
            var pc = recommended(for: weightType)
            pc.name += " + popcount"
            pc.flags["GGML_METAL_Q1_0_POPCNT"] = "1"
            arms.append(pc)
        }
        return arms
    }
}

/// A measurement: "tg128" (greedy generation after a chat prompt) or "ppK" (k-token batches).
struct Cell: Codable, Hashable, Identifiable {
    var name: String
    var id: String { name }
    var isGeneration: Bool { name.hasPrefix("tg") }
    var count: Int { Int(name.dropFirst(2)) ?? 0 }
}

let defaultCells: [Cell] = ["tg128", "pp2", "pp4", "pp8", "pp512"].map { Cell(name: $0) }

struct Observation: Codable {
    var arm: String
    var position: Int                 // 0..3 in A-B-B-A
    var tokensPerSecond: Double
    var promptTokensPerSecond: Double?
    var generatedTokens: [Int32]?
    var thermalState: String
    var footprintBytes: UInt64
    var availableBytes: UInt64
    var startedAt: Date
}

struct Quartet: Codable {
    var cell: String
    var cycle: Int
    var attempt: Int
    var observations: [Observation]
    var spread: Double
    var ratio: Double                 // mean(B) / mean(A)
    var tokensIdentical: Bool?
    var accepted: Bool
}

struct CellSummary: Codable {
    var cell: String
    var aMean: Double
    var bMean: Double
    var speedupGeomean: Double
    var speedupMin: Double
    var speedupMax: Double
    var acceptedQuartets: Int
    var rejectedQuartets: Int
    var tokenMismatchQuartets: Int
}

struct RunResult: Codable {
    var tool = "BonsaiBench 1"
    var sourceCommit: String
    var device: DeviceInfo
    var model: String
    var modelDescription: String
    var modelBytes: UInt64
    var armA: Arm
    var armB: Arm
    var cycles: Int
    var cooldownSeconds: Double
    var spreadGate: Double
    var prompt: String
    var quartets: [Quartet] = []
    var summaries: [CellSummary] = []
    var started = Date()
    var finished: Date?
    var peakFootprintBytes: UInt64 = 0
    var minAvailableBytes: UInt64 = .max
    var error: String?
}

/// Source revision of the llama.cpp framework this app links; update when rebuilding the framework.
let sourceCommit = "downstream/metal-ptq1-m5-tuning"

let benchPrompt = "<|im_start|>user\nWrite a Python function that merges two sorted lists into one sorted list, " +
    "with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

/// A-B-B-A study: every cell gets `cycles` quartets in a reproducibly shuffled order, a cooldown before
/// each observation, a fresh context per observation, and the spread gate (larger/smaller of the two A
/// runs, and of the two B runs, must each be <= gate). A failed quartet is kept and repeated, at most
/// `attempts` times; never the fastest chosen.
final class Study {
    let engine: Engine
    var result: RunResult
    let cells: [Cell]
    let attempts: Int
    var cancelled = false
    let log: (String) -> Void

    init(engine: Engine, armA: Arm, armB: Arm, cells: [Cell], cycles: Int, cooldown: Double, gate: Double,
         attempts: Int, log: @escaping (String) -> Void) {
        self.engine = engine
        self.cells = cells
        self.attempts = attempts
        self.log = log
        let attrs = try? FileManager.default.attributesOfItem(atPath: engine.path)
        let size = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
        result = RunResult(sourceCommit: sourceCommit, device: DeviceInfo.capture(),
                           model: (engine.path as NSString).lastPathComponent, modelDescription: engine.description,
                           modelBytes: size, armA: armA, armB: armB, cycles: cycles, cooldownSeconds: cooldown,
                           spreadGate: gate, prompt: benchPrompt)
    }

    private func observe(cell: Cell, arm: Arm, position: Int) throws -> Observation {
        Thread.sleep(forTimeInterval: result.cooldownSeconds)
        applyFlags(arm.flags)
        let start = Date()
        var obs = Observation(arm: arm.name, position: position, tokensPerSecond: 0, thermalState: thermalStateName(),
                              footprintBytes: 0, availableBytes: 0, startedAt: start)
        if cell.isGeneration {
            let g = try engine.generate(prompt: benchPrompt, n: cell.count)
            obs.tokensPerSecond = g.tokensPerSecond
            obs.promptTokensPerSecond = g.promptSeconds > 0 ? Double(g.promptTokens) / g.promptSeconds : nil
            obs.generatedTokens = g.generated
        } else {
            obs.tokensPerSecond = try engine.batchRate(k: cell.count, reps: cell.count >= 256 ? 2 : 5)
        }
        obs.footprintBytes = physicalFootprint()
        obs.availableBytes = UInt64(max(0, availableMemory()))
        result.peakFootprintBytes = max(result.peakFootprintBytes, obs.footprintBytes)
        if obs.availableBytes > 0 { result.minAvailableBytes = min(result.minAvailableBytes, obs.availableBytes) }
        log(String(format: "%@ %@ %d%@ %.2f tok/s  [%@]", cell.name, arm.name, position + 1,
                   position == 0 || position == 3 ? "A" : "B", obs.tokensPerSecond, obs.thermalState))
        return obs
    }

    func run() {
        var order = cells
        for cycle in 0..<result.cycles {
            var g = SeededGenerator(seed: UInt64(20260923 + cycle))
            order.shuffle(using: &g)
            for cell in order {
                if cancelled { return }
                for attempt in 1...attempts {
                    do {
                        var obs: [Observation] = []
                        for (pos, arm) in [result.armA, result.armB, result.armB, result.armA].enumerated() {
                            if cancelled { return }
                            obs.append(try observe(cell: cell, arm: arm, position: pos))
                        }
                        let a = obs.filter { $0.position == 0 || $0.position == 3 }.map(\.tokensPerSecond)
                        let b = obs.filter { $0.position == 1 || $0.position == 2 }.map(\.tokensPerSecond)
                        let spread = max(a.max()! / a.min()!, b.max()! / b.min()!)
                        let ratio = (b.reduce(0, +) / 2) / (a.reduce(0, +) / 2)
                        var identical: Bool? = nil
                        if cell.isGeneration, let t0 = obs[0].generatedTokens {
                            identical = obs.allSatisfy { $0.generatedTokens == t0 }
                        }
                        let ok = spread <= result.spreadGate
                        result.quartets.append(Quartet(cell: cell.name, cycle: cycle, attempt: attempt, observations: obs,
                                                       spread: spread, ratio: ratio, tokensIdentical: identical, accepted: ok))
                        log(String(format: "%@ cycle %d: B/A %.3f spread %.3f%@%@", cell.name, cycle, ratio, spread,
                                   ok ? "" : "  REJECTED", identical == false ? "  TOKENS DIFFER" : ""))
                        if ok { break }
                    } catch {
                        result.error = error.localizedDescription
                        log("error: \(error.localizedDescription)")
                        return
                    }
                }
            }
        }
        summarize()
        result.finished = Date()
    }

    func summarize() {
        result.summaries = cells.map { cell in
            let qs = result.quartets.filter { $0.cell == cell.name }
            let acc = qs.filter(\.accepted)
            let ratios = acc.map(\.ratio)
            let aVals = acc.flatMap { $0.observations.filter { $0.position == 0 || $0.position == 3 }.map(\.tokensPerSecond) }
            let bVals = acc.flatMap { $0.observations.filter { $0.position == 1 || $0.position == 2 }.map(\.tokensPerSecond) }
            let geo = ratios.isEmpty ? 0 : Foundation.exp(ratios.map { Foundation.log($0) }.reduce(0, +) / Double(ratios.count))
            return CellSummary(cell: cell.name,
                               aMean: aVals.isEmpty ? 0 : aVals.reduce(0, +) / Double(aVals.count),
                               bMean: bVals.isEmpty ? 0 : bVals.reduce(0, +) / Double(bVals.count),
                               speedupGeomean: geo, speedupMin: ratios.min() ?? 0, speedupMax: ratios.max() ?? 0,
                               acceptedQuartets: acc.count, rejectedQuartets: qs.count - acc.count,
                               tokenMismatchQuartets: qs.filter { $0.tokensIdentical == false }.count)
        }
    }
}

/// SplitMix64, so the cell order per cycle is reproducible.
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
