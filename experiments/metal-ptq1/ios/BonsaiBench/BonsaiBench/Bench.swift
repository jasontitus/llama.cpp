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
    static let prismPopcount = Arm(name: "PrismML popcount (their option)", flags: ["GGML_METAL_Q1_0_POPCNT": "1"])

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
            // PrismML's own bit-plane option (off by default in their code). Not one of our changes: compare
            // "PrismML popcount" with "M5 stack (Q1) + PrismML popcount" for what ours add on top of it.
            arms.append(prismPopcount)
            var pc = recommended(for: weightType)
            pc.name += " + PrismML popcount"
            pc.flags["GGML_METAL_Q1_0_POPCNT"] = "1"
            arms.append(pc)
        }
        return arms + diagnostic(for: weightType)
    }

    /// One flag of the recommended stack at a time (a numeric parameter goes with the flag it tunes), to find
    /// which one is responsible for a difference.
    static func diagnostic(for weightType: String) -> [Arm] {
        let stack = recommended(for: weightType).flags
        return stack.keys.sorted().filter { $0 != "GGML_METAL_PTQ1_MULTICOL_MAX" }.map { name in
            var flags = [name: stack[name]!]
            if name == "GGML_METAL_PTQ1_MULTICOL", let m = stack["GGML_METAL_PTQ1_MULTICOL_MAX"] { flags["GGML_METAL_PTQ1_MULTICOL_MAX"] = m }
            let short = name.replacingOccurrences(of: "GGML_METAL_", with: "").replacingOccurrences(of: "GGML_", with: "")
            return Arm(name: "only \(short)", flags: flags)
        }
    }
}

/// A measurement:
/// - "tgN": N single-token decodes from an empty context with random tokens, no sampling (llama-bench tgN,
///   the Mac studies' tg128);
/// - "chatN": greedy generation of N tokens after a chat prompt, with the tokens compared between arms;
/// - "ppK": k-token batches (llama-bench ppK).
struct Cell: Codable, Hashable, Identifiable {
    var name: String
    var id: String { name }
    var kind: String { String(name.prefix { $0.isLetter }) }
    var count: Int { Int(name.drop { $0.isLetter }) ?? 0 }
}

let defaultCells: [Cell] = ["tg128", "chat128", "pp2", "pp4", "pp8", "pp512"].map { Cell(name: $0) }

struct Observation: Codable {
    var arm: String
    var position: Int                 // 0..3 in A-B-B-A
    var tokensPerSecond: Double
    var promptTokensPerSecond: Double?
    var generatedTokens: [Int32]?
    var thermalBefore: String
    var thermalAfter: String
    var thermalWaitSeconds: Double    // waited for the thermal gate before the cooldown
    var interrupted: Bool             // the app left the foreground during the observation
    var error: String?                // the observation failed (e.g. a Metal command buffer error)
    var callSeconds: [Double]?        // ppK: each timed decode call, to tell a stall from a uniform slowdown
    var libraryMessages: [String]     // library warnings/errors during the observation
    var footprintBytes: UInt64        // while the observation's context was alive
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
    var rejectReason: String?
}

struct CellSummary: Codable {
    var cell: String
    var aMean: Double?
    var bMean: Double?
    var speedupGeomean: Double?
    var speedupMin: Double?
    var speedupMax: Double?
    var acceptedQuartets: Int
    var rejectedQuartets: Int
    var tokenMismatchQuartets: Int
    var complete: Bool                // an accepted quartet for every cycle
}

struct RunResult: Codable {
    var tool = "BonsaiBench 2"
    var build: BuildInfo
    var launchEnvironment: [String: String]
    var device: DeviceInfo
    var model: String
    var modelDescription: String
    var modelBytes: UInt64
    var modelSHA256Verified: String?  // set when the file matched the published hash in this app
    var armA: Arm
    var armB: Arm
    var cycles: Int
    var cooldownSeconds: Double
    var waitForNominal: Bool          // thermal gate: wait for nominal (else only leave serious/critical)
    var spreadGate: Double
    var prompt: String
    var quartets: [Quartet] = []
    var summaries: [CellSummary] = []
    var started = Date()
    var finished: Date?               // nil in a file left by a study that was killed
    var cancelled = false
    var peakFootprintBytes: UInt64 = 0
    var minAvailableBytes: UInt64?
    var error: String?
}

/// What was built: written to BuildStamp.plist by the "Stamp build" script (project.yml).
struct BuildInfo: Codable {
    var sourceRevision: String        // git HEAD of the llama.cpp tree when the app was built
    var sourceDirtyFiles: String      // changed files under ggml/ src/ include/ at that time
    var frameworkSHA256: String       // the llama.framework binary that was embedded

    static let current: BuildInfo = {
        let stamp = Bundle.main.url(forResource: "BuildStamp", withExtension: "plist")
            .flatMap { NSDictionary(contentsOf: $0) as? [String: String] } ?? [:]
        func key(_ k: String) -> String { stamp[k] ?? "unknown" }
        return BuildInfo(sourceRevision: key("BBSourceRevision"), sourceDirtyFiles: key("BBSourceDirtyFiles"),
                         frameworkSHA256: key("BBFrameworkSHA256"))
    }()
}

let benchPrompt = "<|im_start|>user\nWrite a Python function that merges two sorted lists into one sorted list, " +
    "with a docstring.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

/// Whether the app is in the foreground, readable from the study thread. iOS does not run GPU work for
/// background apps, so the study pauses while inactive and rejects quartets that span a trip away.
final class AppActivity {
    static let shared = AppActivity()
    private let lock = NSLock()
    private var active = true
    private var changes = 0

    func set(active a: Bool) {
        lock.lock(); defer { lock.unlock() }
        if a != active { active = a; changes += 1 }
    }

    var state: (active: Bool, changes: Int) {
        lock.lock(); defer { lock.unlock() }
        return (active, changes)
    }
}

/// A-B-B-A study: every cell gets `cycles` quartets in a reproducibly shuffled order, a cooldown before
/// each observation, a fresh context per observation, and the spread gate (larger/smaller of the two A
/// runs, and of the two B runs, must each be <= gate). A quartet is also rejected if the thermal state
/// changed during it or the app left the foreground. A rejected quartet is kept and repeated, at most
/// `attempts` times, never the fastest chosen; a cell without an accepted quartet for every cycle is
/// marked incomplete. The result is saved after every quartet.
final class Study {
    let engine: Engine
    private(set) var result: RunResult
    let cells: [Cell]
    let attempts: Int
    let log: (String) -> Void
    let save: (RunResult) -> Void
    private let lock = NSLock()
    private var _cancelled = false
    var cancelled: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _cancelled }
        set { lock.lock(); _cancelled = newValue; lock.unlock() }
    }

    init(engine: Engine, armA: Arm, armB: Arm, cells: [Cell], cycles: Int, cooldown: Double, waitForNominal: Bool,
         gate: Double, attempts: Int, log: @escaping (String) -> Void, save: @escaping (RunResult) -> Void) {
        self.engine = engine
        self.cells = cells
        self.attempts = attempts
        self.log = log
        self.save = save
        let url = URL(fileURLWithPath: engine.path)
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { UInt64($0) } ?? 0
        result = RunResult(build: .current, launchEnvironment: launchEnvironment, device: DeviceInfo.capture(),
                           model: url.lastPathComponent, modelDescription: engine.description, modelBytes: size,
                           modelSHA256Verified: VerifiedMark.get(url), armA: armA, armB: armB, cycles: cycles,
                           cooldownSeconds: cooldown, waitForNominal: waitForNominal, spreadGate: gate, prompt: benchPrompt)
    }

    private struct Interrupted: Error {}

    /// Sleep in short steps so Stop stays responsive.
    private func sleep(_ seconds: Double) throws {
        let end = Date().addingTimeInterval(seconds)
        while Date() < end {
            if cancelled { throw CancellationError() }
            Thread.sleep(forTimeInterval: min(0.5, end.timeIntervalSinceNow))
        }
    }

    private func observe(cell: Cell, arm: Arm, position: Int) throws -> Observation {
        while !AppActivity.shared.state.active { try sleep(1) }
        // A warm phone does not recover within a short cooldown, and compute-bound arms lose more to its lower
        // clocks than others (a non-linear drift A-B-B-A cannot cancel): wait (bounded) for the gate state.
        let allowed = result.waitForNominal ? ["nominal"] : ["nominal", "fair"]
        let waitStart = Date()
        while !allowed.contains(thermalStateName()) && Date().timeIntervalSince(waitStart) < 300 { try sleep(5) }
        let waited = Date().timeIntervalSince(waitStart)
        try sleep(result.cooldownSeconds)
        applyFlags(arm.flags)
        let changes = AppActivity.shared.state.changes
        var obs = Observation(arm: arm.name, position: position, tokensPerSecond: 0, thermalBefore: thermalStateName(),
                              thermalAfter: "", thermalWaitSeconds: waited, interrupted: false, error: nil, callSeconds: nil,
                              libraryMessages: [], footprintBytes: 0, availableBytes: 0, startedAt: Date())
        _ = LibraryLog.shared.drain()
        do {
            let probe: Probe
            switch cell.kind {
            case "chat":
                let g = try engine.generate(prompt: benchPrompt, n: cell.count)
                obs.tokensPerSecond = g.tokensPerSecond
                obs.promptTokensPerSecond = g.promptSeconds > 0 ? Double(g.promptTokens) / g.promptSeconds : nil
                obs.generatedTokens = g.generated
                probe = g.probe
            case "tg":
                let r = try engine.generationRate(n: cell.count)
                obs.tokensPerSecond = r.rate
                probe = r.probe
            default:
                let r = try engine.batchRate(k: cell.count, minReps: cell.count >= 256 ? 2 : 5)
                obs.tokensPerSecond = r.rate
                obs.callSeconds = r.calls
                probe = r.probe
            }
            obs.footprintBytes = probe.footprintBytes
            obs.availableBytes = probe.availableBytes
        } catch let e as EngineError {
            // A failed decode or context (Metal error, out of memory, the app sent to the background) is a
            // result for this configuration: record it, reject the quartet, go on.
            obs.error = e.localizedDescription
        }
        obs.libraryMessages = LibraryLog.shared.drain()
        obs.interrupted = AppActivity.shared.state.changes != changes
        obs.thermalAfter = thermalStateName()
        result.peakFootprintBytes = max(result.peakFootprintBytes, obs.footprintBytes)
        if obs.availableBytes > 0 { result.minAvailableBytes = min(result.minAvailableBytes ?? .max, obs.availableBytes) }
        log(String(format: "%@ %@ %d%@ %.2f tok/s  [%@%@]%@", cell.name, arm.name, position + 1,
                   position == 0 || position == 3 ? "A" : "B", obs.tokensPerSecond, obs.thermalBefore,
                   obs.thermalAfter == obs.thermalBefore ? "" : "→" + obs.thermalAfter,
                   obs.interrupted ? "  INTERRUPTED" : ""))
        if let e = obs.error { log("  FAILED: \(e)") }
        for m in obs.libraryMessages.prefix(6) { log("  lib: \(m)") }
        return obs
    }

    private func quartet(cell: Cell, cycle: Int, attempt: Int) throws -> Quartet {
        var obs: [Observation] = []
        for (pos, arm) in [result.armA, result.armB, result.armB, result.armA].enumerated() {
            obs.append(try observe(cell: cell, arm: arm, position: pos))
        }
        let a = obs.filter { $0.position == 0 || $0.position == 3 }.map(\.tokensPerSecond)
        let b = obs.filter { $0.position == 1 || $0.position == 2 }.map(\.tokensPerSecond)
        let valid = (a + b).allSatisfy { $0.isFinite && $0 > 0 }
        let spread = valid ? max(a.max()! / a.min()!, b.max()! / b.min()!) : .infinity
        let ratio = valid ? (b.reduce(0, +) / 2) / (a.reduce(0, +) / 2) : .nan
        var identical: Bool? = nil
        if cell.kind == "chat", let t0 = obs[0].generatedTokens {
            identical = obs.allSatisfy { $0.generatedTokens == t0 }
        }
        let thermal = Set(obs.flatMap { [$0.thermalBefore, $0.thermalAfter] })
        let reason: String? =
            obs.contains(where: \.interrupted) ? "app left the foreground" :
            obs.contains(where: { $0.error != nil }) ? "an observation failed" :
            !valid ? "no valid rate" :
            thermal.count > 1 ? "thermal state changed (\(thermal.sorted().joined(separator: ", ")))" :
            spread > result.spreadGate ? String(format: "spread %.3f over the gate", spread) : nil
        return Quartet(cell: cell.name, cycle: cycle, attempt: attempt, observations: obs, spread: spread, ratio: ratio,
                       tokensIdentical: identical, accepted: reason == nil, rejectReason: reason)
    }

    func run() {
        defer {
            summarize()
            result.finished = Date()
            save(result)
        }
        save(result)   // a study killed in its first observation still leaves a record
        var order = cells
        do {
            for cycle in 0..<result.cycles {
                var g = SeededGenerator(seed: UInt64(20260923 + cycle))
                order.shuffle(using: &g)
                for cell in order {
                    var accepted = false
                    for attempt in 1...attempts where !accepted {
                        let q = try quartet(cell: cell, cycle: cycle, attempt: attempt)
                        result.quartets.append(q)
                        accepted = q.accepted
                        log(String(format: "%@ cycle %d: B/A %.3f spread %.3f%@%@", cell.name, cycle + 1, q.ratio, q.spread,
                                   q.rejectReason.map { "  REJECTED: " + $0 } ?? "",
                                   q.tokensIdentical == false ? "  TOKENS DIFFER" : ""))
                        save(result)
                    }
                    if !accepted { log("\(cell.name) cycle \(cycle + 1): no acceptable quartet in \(attempts) attempts") }
                }
            }
        } catch is CancellationError {
            result.cancelled = true
            log("stopped")
        } catch {
            result.error = error.localizedDescription
            log("error: \(error.localizedDescription)")
        }
    }

    func summarize() {
        result.summaries = cells.map { cell in
            let qs = result.quartets.filter { $0.cell == cell.name }
            let acc = qs.filter(\.accepted)
            let ratios = acc.map(\.ratio)
            let aVals = acc.flatMap { $0.observations.filter { $0.position == 0 || $0.position == 3 }.map(\.tokensPerSecond) }
            let bVals = acc.flatMap { $0.observations.filter { $0.position == 1 || $0.position == 2 }.map(\.tokensPerSecond) }
            let mean = { (v: [Double]) -> Double? in v.isEmpty ? nil : v.reduce(0, +) / Double(v.count) }
            let geo = ratios.isEmpty ? nil : Foundation.exp(ratios.map { Foundation.log($0) }.reduce(0, +) / Double(ratios.count))
            return CellSummary(cell: cell.name, aMean: mean(aVals), bMean: mean(bVals), speedupGeomean: geo,
                               speedupMin: ratios.min(), speedupMax: ratios.max(), acceptedQuartets: acc.count,
                               rejectedQuartets: qs.count - acc.count,
                               tokenMismatchQuartets: qs.filter { $0.tokensIdentical == false }.count,
                               complete: Set(acc.map(\.cycle)).count == result.cycles)
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
