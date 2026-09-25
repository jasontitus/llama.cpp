import Foundation

/// A named set of research flags (one benchmark arm).
struct Arm: Codable, Hashable, Identifiable {
    var name: String
    var flags: [String: String]
    var draft = 0                     // MTP draft tokens per step in gen cells (0 = plain decoding)
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

    /// Q1_0 prompt processing on the static-K32 tensor kernel with a grouped grid (swizzle 1): +7% pp512 on
    /// M5 Max over the Q1 stack, bitwise equal output there (experiments/metal-ptq1/m5/EXPERIMENTS.md).
    static var q1Prefill: Arm {
        var a = recommended(for: "Q1_0")
        a.name += " + K32 prefill"
        a.flags["GGML_METAL_Q1_SWIZZLE_LOG"] = "1"
        return a
    }

    static func tensor(for weightType: String) -> Arm {
        var a = recommended(for: weightType)
        a.name += " + tensor"
        a.flags["GGML_METAL_PTQ1_TENSOR"] = "1"
        return a
    }

    /// With an MTP model, "+ MTP" arms draft one token per step in gen cells (the Mac studies' setting).
    static func all(for weightType: String, mtp: Bool = false) -> [Arm] {
        var arms = [upstream, bitExact, recommended(for: weightType), invariant(for: weightType)]
        if mtp {
            // "+ invariant + MTP": batch-invariant verification, so MTP output can be compared bit for bit
            for base in [upstream, recommended(for: weightType), invariant(for: weightType)] {
                var a = base
                a.name += " + MTP"
                a.draft = 1
                arms.append(a)
            }
        }
        if weightType == "PTQ1_0" { arms.append(tensor(for: weightType)) }
        if weightType == "Q1_0" {
            // PrismML's own bit-plane option (off by default in their code). Not one of our changes: compare
            // "PrismML popcount" with "M5 stack (Q1) + PrismML popcount" for what ours add on top of it.
            arms.append(prismPopcount)
            var pc = recommended(for: weightType)
            pc.name += " + PrismML popcount"
            pc.flags["GGML_METAL_Q1_0_POPCNT"] = "1"
            arms.append(pc)
            arms.append(q1Prefill)
        }
        return arms + diagnostic(for: weightType)
    }

    /// One flag of the recommended stack at a time, to find which one is responsible for a difference. A flag
    /// that only acts with another carries it (PTQ1 staging needs the multi-column path; the column limit
    /// tunes both the multi-column and the fused FFN kernels), and the arm's name says so.
    static func diagnostic(for weightType: String) -> [Arm] {
        let stack = recommended(for: weightType).flags
        let short = { (n: String) in n.replacingOccurrences(of: "GGML_METAL_", with: "").replacingOccurrences(of: "GGML_", with: "") }
        let needs: [String: [String]] = [
            "GGML_METAL_PTQ1_MULTICOL": ["GGML_METAL_PTQ1_MULTICOL_MAX"],
            "GGML_METAL_PTQ1_GLU": ["GGML_METAL_PTQ1_MULTICOL_MAX"],
            "GGML_METAL_PTQ1_STAGE": ["GGML_METAL_PTQ1_MULTICOL", "GGML_METAL_PTQ1_MULTICOL_MAX"],
        ]
        return stack.keys.sorted().filter { $0 != "GGML_METAL_PTQ1_MULTICOL_MAX" }.map { name in
            var flags = [name: stack[name]!]
            for d in needs[name] ?? [] { if let v = stack[d] { flags[d] = v } }
            let with = (needs[name] ?? []).filter { $0 != "GGML_METAL_PTQ1_MULTICOL_MAX" && stack[$0] != nil }.map(short)
            return Arm(name: "only \(short(name))" + (with.isEmpty ? "" : " (with \(with.joined(separator: ", ")))"), flags: flags)
        }
    }
}

/// A measurement:
/// - "tgN": N single-token decodes from an empty context with random tokens, no sampling (llama-bench tgN,
///   the Mac studies' tg128);
/// - "chatN": greedy generation of N tokens after a chat prompt, with the tokens compared between arms;
/// - "ppK": k-token batches (llama-bench ppK);
/// - "genN": greedy generation of N tokens after the chat prompt through llama.cpp's speculative-decoding
///   loop, plain or with MTP drafts per arm (llama-server's generation speed; the tokens are compared).
struct Cell: Codable, Hashable, Identifiable {
    var name: String
    var id: String { name }
    var kind: String { String(name.prefix { $0.isLetter }) }
    var count: Int { Int(name.drop { $0.isLetter }) ?? 0 }
}

let defaultCells: [Cell] = ["tg128", "chat128", "gen128", "pp2", "pp4", "pp8", "pp512"].map { Cell(name: $0) }

/// Cells switched on for a new study: gen128 only with an MTP model (it is what MTP arms are measured on).
func defaultSelectedCells(mtp: Bool) -> Set<String> {
    mtp ? ["gen128"] : ["tg128", "chat128", "pp2", "pp4", "pp8", "pp512"]
}

struct Observation: Codable {
    var arm: String
    var position: Int                 // 0..3 in A-B-B-A
    var tokensPerSecond: Double
    var promptTokensPerSecond: Double?
    var generatedTokens: [Int32]?
    var thermalBefore: String         // when the measurement started (after the gate and cooldown)
    var thermalAfter: String
    var thermalMax: String            // the hottest state seen during the run
    var thermalWaitSeconds: Double    // waited for the thermal gate (cooldowns excluded)
    var gateReached: Bool             // the thermal gate was met within the study's thermal wait limit
    var interrupted: Bool             // the app left the foreground during the observation
    var error: String?                // the observation failed (e.g. a Metal command buffer error)
    var callSeconds: [Double]?        // ppK: each timed decode call, to tell a stall from a uniform slowdown
    var draftSeconds: Double?         // genN: generation loop split into MTP drafting,
    var verifySeconds: Double?        // target decode and sampling,
    var processSeconds: Double?       // and feeding the batch to the MTP context
    var drafted: Int?                 // genN with MTP: draft tokens verified
    var accepted: Int?                // of which accepted
    var generateSteps: Int?           // genN: target decodes
    var warmupCallSeconds: [Double]?  // ppK: the warmup calls (a failure often happens there)
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
    var tokensIdentical: Bool?        // all four runs generated the same tokens
    var tokensIdenticalWithinArms: Bool? // A with A and B with B (a difference here is nondeterminism)
    var tokensIdenticalBetweenArms: Bool? // A with B (MTP verification can legitimately differ: see README)
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
    var tokenMismatchQuartets: Int    // runs of the same arm, or (without MTP) of the two arms, generated different tokens
    var mtpTokenDifferenceQuartets: Int // the arms differ where MTP makes that expected (see README)
    var aAcceptance: Double?          // MTP draft acceptance of each arm, accepted quartets
    var bAcceptance: Double?
    var complete: Bool                // an accepted quartet for every cycle
}

struct RunResult: Codable {
    var tool = "BonsaiBench 3"
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
    var waitForNominal: Bool          // thermal gate for a quartet's first run: nominal (else nominal or fair)
    var thermalWaitLimitSeconds: Double? // how long a run waits for the gate before its quartet is rejected (300 if absent)
    var promptUbatch: Int             // ubatch for ppK: 512 as on the Mac; smaller splits long GPU submissions
    var spreadGate: Double
    var prompt: String
    var quartets: [Quartet] = []
    var unfinishedQuartet: [Observation]? // runs of a quartet the study stopped in (saved after every run)
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
    var sourceRevision: String        // git HEAD when the app was built
    var appDirtyFiles: String         // uncommitted files under experiments/metal-ptq1/ios at that time
    var frameworkRevision: String     // git HEAD when llama.xcframework was built (build-xcframework.sh)
    var frameworkDirtyFiles: String   // uncommitted files under ggml/ src/ include/ then
    var frameworkSHA256: String       // the llama.framework binary, before it is signed into the app

    static let current: BuildInfo = {
        let stamp = Bundle.main.url(forResource: "BuildStamp", withExtension: "plist")
            .flatMap { NSDictionary(contentsOf: $0) as? [String: String] } ?? [:]
        func key(_ k: String) -> String { stamp[k] ?? "unknown" }
        return BuildInfo(sourceRevision: key("BBSourceRevision"), appDirtyFiles: key("BBAppDirtyFiles"),
                         frameworkRevision: key("BBFrameworkRevision"), frameworkDirtyFiles: key("BBFrameworkDirtyFiles"),
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
/// runs, and of the two B runs, must each be <= gate). A quartet is also rejected if its runs started in
/// different thermal states, if any run reached serious/critical, or if the app left the foreground. A rejected quartet is kept and repeated, at most
/// `attempts` times, never the fastest chosen; a cell without an accepted quartet for every cycle is
/// marked incomplete. The result is saved after every quartet.
final class Study {
    let engine: Engine
    private(set) var result: RunResult
    let cells: [Cell]
    let attempts: Int
    let log: (String) -> Void
    let progress: (String) -> Void    // one line: where the study is and what it is waiting for
    private var here = ""
    let save: (RunResult) -> Void
    private let lock = NSLock()
    private var _cancelled = false
    var cancelled: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _cancelled }
        set { lock.lock(); _cancelled = newValue; lock.unlock() }
    }

    init(engine: Engine, armA: Arm, armB: Arm, cells: [Cell], cycles: Int, cooldown: Double, waitForNominal: Bool,
         thermalWaitLimit: Double = 300,
         promptUbatch: Int = 512, gate: Double, attempts: Int, log: @escaping (String) -> Void, progress: @escaping (String) -> Void = { _ in },
         save: @escaping (RunResult) -> Void) {
        self.engine = engine
        self.cells = cells
        self.attempts = attempts
        self.log = log
        self.progress = progress
        self.save = save
        let url = URL(fileURLWithPath: engine.path)
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { UInt64($0) } ?? 0
        result = RunResult(build: .current, launchEnvironment: launchEnvironment, device: DeviceInfo.capture(),
                           model: url.lastPathComponent, modelDescription: engine.description, modelBytes: size,
                           modelSHA256Verified: VerifiedMark.get(url), armA: armA, armB: armB, cycles: cycles,
                           cooldownSeconds: cooldown, waitForNominal: waitForNominal, promptUbatch: promptUbatch,
                           spreadGate: gate, prompt: benchPrompt)
        result.thermalWaitLimitSeconds = thermalWaitLimit
    }

    private struct Interrupted: Error {}

    /// Sleep in short steps so Stop stays responsive; with a label, show a countdown.
    private func sleep(_ seconds: Double, _ label: String? = nil) throws {
        let end = Date().addingTimeInterval(seconds)
        while Date() < end {
            if cancelled { throw CancellationError() }
            if let label { progress(String(format: "%@ · %@ %.0f s", here, label, end.timeIntervalSinceNow.rounded(.up))) }
            Thread.sleep(forTimeInterval: min(0.5, end.timeIntervalSinceNow))
        }
    }

    private let thermal = ThermalMonitor()

    /// Wait until the app is in front and, after the cooldown, the phone is at or below `limit`: the state a run
    /// starts in is what the quartet is judged by. Thermal waiting is bounded by the study's limit (5 minutes by
    /// default, an hour in the unattended suite), after which the quartet is rejected.
    private func gate(limit: Int) throws -> (reached: Bool, waited: Double) {
        let start = Date()
        let maxWait = result.thermalWaitLimitSeconds ?? 300
        var cooled = 0.0
        while true {
            while !AppActivity.shared.state.active {
                progress("\(here) · paused: the app is not in front")
                try sleep(1)
            }
            while thermalRank(thermalStateName()) > limit && Date().timeIntervalSince(start) - cooled < maxWait {
                let waited = Int((Date().timeIntervalSince(start) - cooled) / 60)
                progress("\(here) · phone is \(thermalStateName()), waiting for \(thermalNames[limit]) (\(waited) of up to \(Int(maxWait / 60)) min)")
                try sleep(5)
            }
            try sleep(result.cooldownSeconds, "cooldown")
            cooled += result.cooldownSeconds
            let waited = Date().timeIntervalSince(start) - cooled
            if AppActivity.shared.state.active && thermalRank(thermalStateName()) <= limit { return (true, waited) }
            if waited >= maxWait { return (false, waited) }
        }
    }

    private func observe(cell: Cell, arm: Arm, position: Int, limit: Int) throws -> Observation {
        let quartetHere = here
        here += " · run \(position + 1)/4 (\(position == 0 || position == 3 ? "A" : "B"): \(arm.name))"
        defer { here = quartetHere }
        progress(here)
        let g = try gate(limit: limit)
        progress("\(here) · measuring")
        applyFlags(arm.flags)
        let changes = AppActivity.shared.state.changes
        thermal.reset()
        var obs = Observation(arm: arm.name, position: position, tokensPerSecond: 0, thermalBefore: thermalStateName(),
                              thermalAfter: "", thermalMax: "", thermalWaitSeconds: g.waited, gateReached: g.reached,
                              interrupted: false, error: nil, callSeconds: nil, draftSeconds: nil, verifySeconds: nil,
                              processSeconds: nil, drafted: nil, accepted: nil,
                              generateSteps: nil, warmupCallSeconds: nil,
                              libraryMessages: [], footprintBytes: 0, availableBytes: 0, startedAt: Date())
        _ = LibraryLog.shared.drain()
        let calls = Engine.CallTimes()
        do {
            let probe: Probe
            // MTP arms draft only in gen cells; anywhere else they would measure plain decoding under an MTP name
            if arm.draft > 0 && cell.kind != "gen" { throw EngineError.generationFailed("an MTP arm only runs in gen cells") }
            switch cell.kind {
            case "chat":
                let r = try engine.generate(prompt: benchPrompt, n: cell.count)
                obs.tokensPerSecond = r.tokensPerSecond
                obs.promptTokensPerSecond = r.promptSeconds > 0 ? Double(r.promptTokens) / r.promptSeconds : nil
                obs.generatedTokens = r.generated
                probe = r.probe
            case "gen":
                let r = try engine.speculativeGenerate(prompt: benchPrompt, n: cell.count, draft: arm.draft)
                obs.tokensPerSecond = r.tokensPerSecond
                obs.promptTokensPerSecond = r.promptSeconds > 0 ? Double(r.promptTokens - 1) / r.promptSeconds : nil
                obs.generatedTokens = r.generated
                obs.drafted = arm.draft > 0 ? r.drafted : nil
                obs.accepted = arm.draft > 0 ? r.accepted : nil
                obs.generateSteps = r.steps
                obs.draftSeconds = r.draftSeconds
                obs.verifySeconds = r.verifySeconds
                obs.processSeconds = r.processSeconds
                probe = r.probe
            case "tg":
                let r = try engine.generationRate(n: cell.count)
                obs.tokensPerSecond = r.rate
                probe = r.probe
            default:
                let r = try engine.batchRate(k: cell.count, ubatch: result.promptUbatch, minReps: cell.count >= 256 ? 2 : 5, calls: calls)
                obs.tokensPerSecond = r.rate
                probe = r.probe
            }
            obs.footprintBytes = probe.footprintBytes
            obs.availableBytes = probe.availableBytes
        } catch let e as EngineError {
            // A failed decode or context (Metal error, out of memory, the app sent to the background) is a
            // result for this configuration: record it, reject the quartet, go on.
            obs.error = e.localizedDescription
        }
        if cell.kind == "pp" {
            obs.callSeconds = calls.timed
            obs.warmupCallSeconds = calls.warmup
        }
        if cell.kind == "gen" { LibraryLog.shared.readCommonLog() }
        obs.libraryMessages = LibraryLog.shared.drain()
        obs.interrupted = AppActivity.shared.state.changes != changes
        thermal.note()
        obs.thermalAfter = thermalStateName()
        obs.thermalMax = thermalNames[thermal.maxRank]
        result.peakFootprintBytes = max(result.peakFootprintBytes, obs.footprintBytes)
        if obs.availableBytes > 0 { result.minAvailableBytes = min(result.minAvailableBytes ?? .max, obs.availableBytes) }
        log(String(format: "%@ %@ %d%@ %.2f tok/s  [%@%@]%@%@", cell.name, arm.name, position + 1,
                   position == 0 || position == 3 ? "A" : "B", obs.tokensPerSecond, obs.thermalBefore,
                   obs.thermalMax == obs.thermalBefore ? "" : "→" + obs.thermalMax,
                   obs.gateReached ? "" : "  GATE NOT MET", obs.interrupted ? "  INTERRUPTED" : ""))
        if let e = obs.error { log("  FAILED: \(e)") }
        for m in obs.libraryMessages.prefix(6) { log("  lib: \(m)") }
        return obs
    }

    /// Why a quartet cannot be accepted because of heat, as soon as that is certain.
    private func thermalReject(_ obs: [Observation]) -> String? {
        if let o = obs.first(where: { !$0.gateReached }) {
            return "the phone did not cool to the gate within \(Int((result.thermalWaitLimitSeconds ?? 300) / 60)) minutes (it was \(o.thermalBefore))"
        }
        if obs.contains(where: { thermalRank($0.thermalMax) >= 2 }) { return "the phone reached serious/critical" }
        let starts = Set(obs.map(\.thermalBefore))
        if starts.count > 1 { return "runs started in different thermal states (\(starts.sorted().joined(separator: ", ")))" }
        return nil
    }

    private func quartet(cell: Cell, cycle: Int, attempt: Int) throws -> Quartet {
        var obs: [Observation] = []
        var anchor = 1
        for (pos, arm) in [result.armA, result.armB, result.armB, result.armA].enumerated() {
            // The first run starts at nominal or fair (nominal with waitForNominal); later runs no hotter than
            // it and never above fair.
            let limit = pos == 0 ? (result.waitForNominal ? 0 : 1) : min(anchor, 1)
            let o = try observe(cell: cell, arm: arm, position: pos, limit: limit)
            obs.append(o)
            if pos == 0 { anchor = thermalRank(o.thermalBefore) }
            result.unfinishedQuartet = obs
            save(result)
            // Once heat has settled the verdict, more runs would only heat the phone further.
            if thermalReject(obs) != nil && !obs.contains(where: \.interrupted) { break }
        }
        result.unfinishedQuartet = nil
        let complete = obs.count == 4
        let a = obs.filter { $0.position == 0 || $0.position == 3 }.map(\.tokensPerSecond)
        let b = obs.filter { $0.position == 1 || $0.position == 2 }.map(\.tokensPerSecond)
        let valid = complete && (a + b).allSatisfy { $0.isFinite && $0 > 0 }
        let spread = valid ? max(a.max()! / a.min()!, b.max()! / b.min()!) : .infinity
        let ratio = valid ? (b.reduce(0, +) / 2) / (a.reduce(0, +) / 2) : .nan
        // Tokens are compared only between runs that all produced them (a failed run is not a mismatch).
        var identical: Bool? = nil
        var withinArms: Bool? = nil
        var betweenArms: Bool? = nil
        if ["chat", "gen"].contains(cell.kind), complete, obs.allSatisfy({ $0.error == nil && $0.generatedTokens != nil }) {
            identical = obs.allSatisfy { $0.generatedTokens == obs[0].generatedTokens }
            withinArms = obs[0].generatedTokens == obs[3].generatedTokens && obs[1].generatedTokens == obs[2].generatedTokens
            betweenArms = obs[0].generatedTokens == obs[1].generatedTokens
        }
        let reason: String? =
            obs.contains(where: \.interrupted) ? "app left the foreground" :
            thermalReject(obs) ??
            (obs.contains(where: { $0.error != nil }) ? "an observation failed" :
            !valid ? "no valid rate" :
            spread > result.spreadGate ? String(format: "spread %.3f over the gate", spread) : nil)
        return Quartet(cell: cell.name, cycle: cycle, attempt: attempt, observations: obs, spread: spread, ratio: ratio,
                       tokensIdentical: identical, tokensIdenticalWithinArms: withinArms, tokensIdenticalBetweenArms: betweenArms,
                       accepted: reason == nil, rejectReason: reason)
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
                        here = "\(cell.name) · cycle \(cycle + 1)/\(result.cycles)" + (attempt > 1 ? " · retry \(attempt - 1)" : "")
                        let q = try quartet(cell: cell, cycle: cycle, attempt: attempt)
                        result.quartets.append(q)
                        accepted = q.accepted
                        log(String(format: "%@ cycle %d: B/A %.3f spread %.3f%@%@", cell.name, cycle + 1, q.ratio, q.spread,
                                   q.rejectReason.map { "  REJECTED: " + $0 } ?? "",
                                   q.tokensIdenticalWithinArms == false ? "  TOKENS DIFFER WITHIN AN ARM" :
                                   q.tokensIdenticalBetweenArms == false ? (mtpTokensMayDiffer ? "  tokens differ between arms (MTP, expected)" : "  TOKENS DIFFER") : ""))
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

    /// Token equality between a plain and an MTP arm is not guaranteed: two-token verification is not
    /// batch-invariant unless both arms use GGML_METAL_BATCH_INVARIANT.
    private var mtpTokensMayDiffer: Bool {
        let inv = { (a: Arm) in a.flags["GGML_METAL_BATCH_INVARIANT"] == "1" }
        return (result.armA.draft > 0 || result.armB.draft > 0) && !(inv(result.armA) && inv(result.armB))
    }

    private func acceptance(_ qs: [Quartet], positions: [Int]) -> Double? {
        let obs = qs.flatMap(\.observations).filter { positions.contains($0.position) }
        let drafted = obs.compactMap(\.drafted).reduce(0, +)
        return drafted > 0 ? Double(obs.compactMap(\.accepted).reduce(0, +)) / Double(drafted) : nil
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
                               tokenMismatchQuartets: qs.filter { $0.tokensIdenticalWithinArms == false ||
                                   ($0.tokensIdenticalBetweenArms == false && !mtpTokensMayDiffer) }.count,
                               mtpTokenDifferenceQuartets: mtpTokensMayDiffer ? qs.filter { $0.tokensIdenticalWithinArms == true &&
                                   $0.tokensIdenticalBetweenArms == false }.count : 0,
                               aAcceptance: acceptance(acc, positions: [0, 3]), bAcceptance: acceptance(acc, positions: [1, 2]),
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

let thermalNames = ["nominal", "fair", "serious", "critical"]

func thermalRank(_ s: String) -> Int { thermalNames.firstIndex(of: s) ?? 3 }

/// The hottest thermal state since `reset()`, from the system's change notifications, so a run that touches
/// serious and comes back before it ends is still seen.
final class ThermalMonitor {
    private let lock = NSLock()
    private var hottest = 0
    private var token: NSObjectProtocol?

    init() {
        token = NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification,
                                                       object: nil, queue: nil) { [weak self] _ in self?.note() }
    }

    deinit { if let token { NotificationCenter.default.removeObserver(token) } }

    func reset() { lock.lock(); hottest = thermalRank(thermalStateName()); lock.unlock() }

    func note() { lock.lock(); hottest = max(hottest, thermalRank(thermalStateName())); lock.unlock() }

    var maxRank: Int { lock.lock(); defer { lock.unlock() }; return min(hottest, 3) }
}
