import Foundation
import QuartzCore
import UIKit

// Diagnostic screens: several configurations of one model measured in mirrored rounds (a random order, then
// the same order reversed), each run starting at the thermal gate, with what the library and the phone can
// report about it. A screen finds which configuration differs; an A-B-B-A study (Bench.swift) then measures that
// difference properly.

/// One configuration of a screen: research flags plus protocol options.
struct ScreenConfig: Codable, Hashable {
    var name: String
    var flags: [String: String]
    var ubatch = 512
    /// GGML_METAL_PROFILE_OPS: every op in its own command buffer, with its GPU time. Serialized, so the rate
    /// of such a run is not comparable with a normal run (no ratios); its op table is the result.
    var profileOps = false
    /// GGML_METAL_CB_STATS: status, error and timing of every command buffer of every graph.
    var cbStats = true
}

struct ScreenSpec: Hashable {
    var configs: [ScreenConfig]
    var cells: [String]                // ppK cells
    var rounds: Int                    // an even number: each random order is followed by its reverse
    var reference: String              // the configuration the others are compared with, round by round
    var retryFailed = false            // repeat a failed run once (performance screens; never a failure screen)
}

/// One command buffer of one graph, as GGML_METAL_CB_STATS writes it (ms from the graph's t0).
struct CBStat: Codable {
    var i: Int
    var main: Bool
    var nodes: Int
    var status: Int                    // MTLCommandBufferStatus: 4 completed, 5 error
    var ks: Double                     // scheduled by the host
    var gs: Double                     // GPU start (-1: not recorded)
    var ge: Double                     // GPU end
    var code: Int?
    var domain: String?
    var err: String?
    var enc: [EncoderStat]?
}

struct EncoderStat: Codable {
    var state: Int                     // MTLCommandEncoderErrorState: 0 unknown, 1 completed, 2 affected, 3 pending, 4 faulted
    var label: String?
    var signposts: Int?                // debug signposts Metal recorded for the encoder
    var last: [String]?                // the last few of them
}

struct CBGraph: Codable {
    var graph: Int
    var ctx: Int?                      // the backend context's number in the process
    var t0: Double?                    // host seconds (mach_absolute_time clock) of the graph's first scheduling
    var now: Double?                   // host seconds and Unix time when the line was written
    var unix: Double?
    var n_nodes: Int
    var n_cb: Int
    var cbs: [CBStat]
}

/// GPU time per op kind of one graph (GGML_METAL_PROFILE_OPS), largest first.
struct OpProfile: Codable {
    var nodes: Int
    var totalMicroseconds: Double
    var ops: [OpTime]
}

struct OpTime: Codable {
    var key: String                    // op, fused count, types and, for MUL_MAT, K x M and columns
    var count: Int
    var microseconds: Double
}

/// System-wide VM counters (host_statistics64), in pages, and this process's page-ins and faults.
struct MemoryCounters: Codable {
    var pageSize: UInt64
    var free: UInt64
    var active: UInt64
    var inactive: UInt64
    var wired: UInt64
    var compressed: UInt64             // pages held by the compressor
    var systemPageins: UInt64
    var systemPageouts: UInt64
    var appPageins: Int64
    var appFaults: Int64

    static func now() -> MemoryCounters? {
        var vm = vm_statistics64()
        var n = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &vm) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(n)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &n) }
        }
        guard kr == KERN_SUCCESS else { return nil }
        var ev = task_events_info()
        var m = mach_msg_type_number_t(MemoryLayout<task_events_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kt = withUnsafeMutablePointer(to: &ev) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(m)) { task_info(mach_task_self_, task_flavor_t(TASK_EVENTS_INFO), $0, &m) }
        }
        var page: vm_size_t = 0
        host_page_size(mach_host_self(), &page)
        return MemoryCounters(pageSize: UInt64(page), free: UInt64(vm.free_count), active: UInt64(vm.active_count),
                              inactive: UInt64(vm.inactive_count), wired: UInt64(vm.wire_count),
                              compressed: UInt64(vm.compressor_page_count), systemPageins: vm.pageins, systemPageouts: vm.pageouts,
                              appPageins: kt == KERN_SUCCESS ? Int64(ev.pageins) : -1,
                              appFaults: kt == KERN_SUCCESS ? Int64(ev.faults) : -1)
    }
}

struct PressureEvent: Codable {
    var host: Double                   // host seconds (CACurrentMediaTime)
    var level: String                  // normal, warning, critical
}

/// What failed in a failed run: the call that ran on the GPU when it failed (a failed command buffer is reported
/// by the next decode, so it is the last recorded call) and the failed command buffer.
struct FailureInfo: Codable {
    var phase: String                  // warmup or timed
    var callSeconds: Double?           // how long that call took until it returned
    var commandBuffer: Int?            // index of the failed command buffer (the main one is n_cb)
    var mainCommandBuffer: Bool?
    var code: Int?
    var domain: String?
    var error: String?
    var gpuStartMs: Double?            // of the failed command buffer, from the graph's t0
    var gpuEndMs: Double?
    var encoderStates: [Int]?          // its encoders: 2 affected (a victim), 4 faulted (the cause)
    var mainStartDelayMs: Double?      // the graph's main command buffer: GPU start minus host scheduling
}

struct ScreenRun: Codable {
    var round: Int
    var position: Int                  // order within the round
    var attempt: Int                   // 2 for a retried run
    var config: String
    var cell: String
    var ubatch: Int
    var startedAt: Date
    var hostStart: Double              // host seconds (same clock as the command-buffer stats)
    var hostEnd: Double
    var tokensPerSecond: Double        // 0 if the run failed
    var callSeconds: [Double]
    var warmupCallSeconds: [Double]
    var callPageinBytes: [Int64]       // paged in system-wide during each timed / warmup call (-1 unknown)
    var warmupPageinBytes: [Int64]
    var error: String?
    var failure: FailureInfo?
    var thermalBefore: String
    var thermalAfter: String
    var thermalMax: String
    var thermalWaitSeconds: Double
    var gateReached: Bool
    var interrupted: Bool              // the app left the foreground during the run
    var availableBytesBefore: UInt64
    var availableBytesAfter: UInt64
    var footprintBytes: UInt64
    var memoryBefore: MemoryCounters?
    var memoryAfter: MemoryCounters?
    var pressureEvents: [PressureEvent] // memory-pressure level changes during the run
    var batteryState: String
    var batteryLevel: Float
    var lowPowerMode: Bool
    var libraryMessages: [String]
    var commandBuffers: [CBGraph]?
    var droppedStatsLines: Int?        // command-buffer lines that could not be read
    var opProfile: OpProfile?
}

struct ScreenSummary: Codable {
    var config: String
    var cell: String
    var runs: Int
    var failed: Int                    // GPU/decode failures, runs that left the foreground not counted
    var failuresByError: [String: Int]
    var excluded: Int                  // left the foreground, gate not met, or reached serious: not in rates
    var medianTokensPerSecond: Double?
    /// This configuration over the reference, from the rounds in which both have a valid run: median, min and
    /// max of the per-round ratios. None for profiled configurations (serialized; rates not comparable).
    var ratioMedian: Double?
    var ratioMin: Double?
    var ratioMax: Double?
    var pairedRounds: Int
    var longestCommandBufferMs: Double?  // longest GPU time of one completed command buffer, over all its runs
}

struct ScreenResult: Codable {
    var tool = "BonsaiBench 3 diagnostics"
    var title: String
    var build: BuildInfo
    var launchEnvironment: [String: String]
    var device: DeviceInfo
    var model: String
    var modelDescription: String
    var modelBytes: UInt64
    var modelSHA256Verified: String?
    var weightsInMemory: Bool          // loaded into app memory (true) or memory-mapped
    var configs: [ScreenConfig]
    var cells: [String]
    var rounds: Int
    var reference: String
    var cooldownSeconds: Double
    var waitForNominal: Bool
    var thermalWaitLimitSeconds: Double
    var seed: String                   // the random orders' seed (a string: JSON numbers lose 64-bit precision)
    var runs: [ScreenRun] = []
    var summaries: [ScreenSummary] = []
    var started = Date()
    var finished: Date?
    var cancelled = false
    var error: String?
}

/// Memory-pressure level changes (normal, warning, critical) with their host time.
final class MemoryPressureLog {
    private let source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical],
                                                                 queue: .global(qos: .utility))
    private let lock = NSLock()
    private var events: [PressureEvent] = []
    init() {
        source.setEventHandler { [weak self] in
            guard let self else { return }
            let d = self.source.data
            let level = d.contains(.critical) ? "critical" : d.contains(.warning) ? "warning" : "normal"
            self.lock.lock(); self.events.append(PressureEvent(host: CACurrentMediaTime(), level: level)); self.lock.unlock()
        }
        source.activate()
    }
    deinit { source.cancel() }
    func since(_ host: Double) -> [PressureEvent] {
        lock.lock(); defer { lock.unlock() }
        return events.filter { $0.host >= host }
    }
}

final class Screen {
    let engine: Engine
    let spec: ScreenSpec
    private(set) var result: ScreenResult
    private let log: (String) -> Void
    private let progress: (String) -> Void
    private let save: (ScreenResult) -> Void
    private let lock = NSLock()
    private var _cancelled = false
    var cancelled: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _cancelled }
        set { lock.lock(); _cancelled = newValue; lock.unlock() }
    }
    private var here = ""
    private let thermal = ThermalMonitor()
    private let pressure = MemoryPressureLog()
    private let seed = UInt64.random(in: 1...UInt64.max)
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("bonsaidiag", isDirectory: true)

    init(engine: Engine, title: String, spec: ScreenSpec, cooldown: Double, waitForNominal: Bool, thermalWaitLimit: Double,
         log: @escaping (String) -> Void, progress: @escaping (String) -> Void, save: @escaping (ScreenResult) -> Void) {
        self.engine = engine
        self.spec = spec
        self.log = log
        self.progress = progress
        self.save = save
        let url = URL(fileURLWithPath: engine.path)
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { UInt64($0) } ?? 0
        result = ScreenResult(title: title, build: .current, launchEnvironment: launchEnvironment, device: DeviceInfo.capture(),
                              model: url.lastPathComponent, modelDescription: engine.description, modelBytes: size,
                              modelSHA256Verified: VerifiedMark.get(url), weightsInMemory: engine.weightsInMemory,
                              configs: spec.configs, cells: spec.cells,
                              rounds: spec.rounds, reference: spec.reference, cooldownSeconds: cooldown,
                              waitForNominal: waitForNominal, thermalWaitLimitSeconds: thermalWaitLimit, seed: String(seed))
    }

    private func sleep(_ seconds: Double, _ label: String? = nil) throws {
        let end = Date().addingTimeInterval(seconds)
        while Date() < end {
            if cancelled { throw CancellationError() }
            if let label { progress(String(format: "%@ · %@ %.0f s", here, label, end.timeIntervalSinceNow.rounded(.up))) }
            Thread.sleep(forTimeInterval: max(0, min(0.5, end.timeIntervalSinceNow)))
        }
    }

    /// As Study.gate: wait until the app is in front and, after the cooldown, the phone is at or below `limit`
    /// (nominal with waitForNominal, else fair), for up to the thermal wait limit.
    private func gate(limit: Int) throws -> (reached: Bool, waited: Double) {
        let start = Date()
        let maxWait = result.thermalWaitLimitSeconds
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

    private static func battery() -> (state: String, level: Float, lowPower: Bool) {
        DispatchQueue.main.sync {
            UIDevice.current.isBatteryMonitoringEnabled = true
            let names: [UIDevice.BatteryState: String] = [.unplugged: "unplugged", .charging: "charging", .full: "full"]
            return (names[UIDevice.current.batteryState] ?? "unknown", UIDevice.current.batteryLevel,
                    ProcessInfo.processInfo.isLowPowerModeEnabled)
        }
    }

    private func measure(cell: Cell, config: ScreenConfig, round: Int, position: Int, attempt: Int) throws -> ScreenRun {
        let outer = here
        here += " · \(config.name)" + (attempt > 1 ? " (retry)" : "")
        defer { here = outer }
        progress(here)
        let g = try gate(limit: result.waitForNominal ? 0 : 1)
        progress("\(here) · measuring")
        try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        let stamp = UUID().uuidString
        let statsURL = scratch.appendingPathComponent("cb-\(stamp).jsonl")
        let profileURL = scratch.appendingPathComponent("ops-\(stamp).jsonl")
        var flags = config.flags
        if config.cbStats { flags["GGML_METAL_CB_STATS"] = statsURL.path }
        if config.profileOps { flags["GGML_METAL_PROFILE_OPS"] = profileURL.path }
        applyFlags(flags)
        defer {
            applyFlags([:])
            try? FileManager.default.removeItem(at: statsURL)
            try? FileManager.default.removeItem(at: profileURL)
        }
        let battery = Self.battery()
        let before = Probe.now()
        let changes = AppActivity.shared.state.changes
        let memoryBefore = MemoryCounters.now()
        thermal.reset()
        let hostStart = CACurrentMediaTime()
        var run = ScreenRun(round: round, position: position, attempt: attempt, config: config.name, cell: cell.name,
                            ubatch: config.ubatch, startedAt: Date(), hostStart: hostStart, hostEnd: 0, tokensPerSecond: 0,
                            callSeconds: [], warmupCallSeconds: [], callPageinBytes: [], warmupPageinBytes: [],
                            error: nil, failure: nil,
                            thermalBefore: thermalStateName(), thermalAfter: "", thermalMax: "", thermalWaitSeconds: g.waited,
                            gateReached: g.reached, interrupted: false, availableBytesBefore: before.availableBytes,
                            availableBytesAfter: 0, footprintBytes: 0, memoryBefore: memoryBefore, memoryAfter: nil,
                            pressureEvents: [], batteryState: battery.state, batteryLevel: battery.level,
                            lowPowerMode: battery.lowPower, libraryMessages: [])
        _ = LibraryLog.shared.drain()
        let calls = Engine.CallTimes()
        do {
            // a profiled run serializes every op: one warmup and one timed call are enough
            let r = config.profileOps
                ? try engine.batchRate(k: cell.count, ubatch: config.ubatch, warmupSeconds: 0, minSeconds: 0, minReps: 1, calls: calls)
                : try engine.batchRate(k: cell.count, ubatch: config.ubatch, minReps: cell.count >= 256 ? 2 : 5, calls: calls)
            run.tokensPerSecond = r.rate
            run.footprintBytes = r.probe.footprintBytes
        } catch let e as EngineError {
            run.error = e.localizedDescription
        }
        run.hostEnd = CACurrentMediaTime()
        run.callSeconds = calls.timed
        run.warmupCallSeconds = calls.warmup
        run.callPageinBytes = calls.timedPageinBytes
        run.warmupPageinBytes = calls.warmupPageinBytes
        run.availableBytesAfter = Probe.now().availableBytes
        run.memoryAfter = MemoryCounters.now()
        run.pressureEvents = pressure.since(hostStart)
        run.libraryMessages = LibraryLog.shared.drain()
        run.interrupted = AppActivity.shared.state.changes != changes
        thermal.note()
        run.thermalAfter = thermalStateName()
        run.thermalMax = thermalNames[thermal.maxRank]
        if config.cbStats {
            let (graphs, dropped) = Self.readCommandBuffers(statsURL)
            run.commandBuffers = graphs
            run.droppedStatsLines = dropped
        }
        if config.profileOps { run.opProfile = Self.readOpProfile(profileURL) }
        if run.error != nil { run.failure = Self.failure(run) }
        logRun(run, cell: cell, config: config)
        return run
    }

    private static func failure(_ run: ScreenRun) -> FailureInfo {
        var f = FailureInfo(phase: run.callSeconds.isEmpty ? "warmup" : "timed",
                            callSeconds: run.callSeconds.last ?? run.warmupCallSeconds.last)
        if let g = run.commandBuffers?.first(where: { $0.cbs.contains { $0.status == 5 } }),
           let cb = g.cbs.first(where: { $0.status == 5 }) {
            f.commandBuffer = cb.i
            f.mainCommandBuffer = cb.main
            f.code = cb.code
            f.domain = cb.domain
            f.error = cb.err
            f.gpuStartMs = cb.gs
            f.gpuEndMs = cb.ge
            f.encoderStates = cb.enc?.map(\.state)
            if let m = g.cbs.first(where: \.main), m.gs >= 0, m.ks >= 0 { f.mainStartDelayMs = m.gs - m.ks }
        }
        return f
    }

    private func logRun(_ run: ScreenRun, cell: Cell, config: ScreenConfig) {
        let longest = run.commandBuffers?.flatMap(\.cbs).filter { $0.status == 4 && $0.gs >= 0 && $0.ge >= 0 }.map { $0.ge - $0.gs }.max()
        var line = "\(cell.name) \(config.name) r\(run.round + 1)" + (run.attempt > 1 ? " retry" : "") + "  "
        if let e = run.error {
            line += "FAILED: \(e)"
            if let f = run.failure, let s = f.callSeconds { line += String(format: " (the %@ call ended after %.2f s)", f.phase, s) }
        } else {
            line += String(format: "%.2f tok/s", run.tokensPerSecond)
        }
        if let longest { line += String(format: "  longest command buffer %.0f ms", longest) }
        if run.pressureEvents.contains(where: { $0.level != "normal" }) { line += "  memory pressure" }
        let reread = run.warmupPageinBytes.filter { $0 > 0 }.reduce(0, +)
        if reread > Engine.residentPageinLimit {
            line += String(format: "  %.1f GB re-read in %d warmup calls", Double(reread) / 1e9, run.warmupCallSeconds.count)
        }
        if run.callPageinBytes.contains(where: { $0 > Screen.timedPageinLimit }) { line += "  STILL READING IN TIMED CALLS" }
        log(line)
        // the backend warns about n_cb > 2 for every context; not news here
        for m in run.libraryMessages.prefix(6) where !m.hasPrefix("pipeline:") && !m.contains("n_cb =") { log("  lib: \(m)") }
    }

    static func readCommandBuffers(_ url: URL) -> ([CBGraph]?, Int) {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return (nil, 0) }
        let d = JSONDecoder()
        var graphs: [CBGraph] = []
        var dropped = 0
        for line in text.split(separator: "\n") {
            if let g = try? d.decode(CBGraph.self, from: Data(line.utf8)) { graphs.append(g) } else { dropped += 1 }
        }
        return (graphs, dropped)
    }

    /// The timed pp graph of the run: the last profiled graph with at least half the GPU time of the largest
    /// (before it: the warmup call; after it: the one-token backend check).
    static func readOpProfile(_ url: URL) -> OpProfile? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var graphs: [OpProfile] = []
        for line in text.split(separator: "\n") {
            guard let o = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any],
                  let ops = o["ops"] as? [[Any]] else { continue }
            var agg: [String: (Int, Double)] = [:]
            var total = 0.0
            for op in ops where op.count >= 9 {
                let name = op[0] as? String ?? "?"
                let fused = (op[1] as? Int) ?? 1
                let t0 = op[2] as? String ?? ""
                let us = (op[8] as? Double) ?? Double(op[8] as? Int ?? 0)
                var key = name + (fused > 1 ? "(+\(fused - 1))" : "") + " " + t0
                if name == "MUL_MAT" || name == "MUL_MAT_ID" {
                    key += " \(op[4])x\(op[5]) n=\(op[7])"
                }
                agg[key, default: (0, 0)].0 += 1
                agg[key, default: (0, 0)].1 += us
                total += us
            }
            let times = agg.map { OpTime(key: $0.key, count: $0.value.0, microseconds: $0.value.1) }
                .sorted { $0.microseconds > $1.microseconds }
            graphs.append(OpProfile(nodes: o["n_nodes"] as? Int ?? 0, totalMicroseconds: total, ops: Array(times.prefix(60))))
        }
        let largest = graphs.map(\.totalMicroseconds).max() ?? 0
        return graphs.last { $0.totalMicroseconds >= largest / 2 }
    }

    /// Did the configuration take effect? A framework built without the diagnostics, or a flag that did not
    /// reach the backend, would otherwise turn a screen into copies of the default.
    private func manipulationProblem(_ run: ScreenRun, _ config: ScreenConfig) -> String? {
        guard run.error == nil else { return nil }   // a failed run may have stopped before any graph
        if config.cbStats && (run.commandBuffers ?? []).isEmpty {
            return "\(config.name): no command-buffer stats were written (is the app's llama.xcframework current?)"
        }
        if config.cbStats, let want = Int(config.flags["GGML_METAL_N_CB"] ?? ""),
           let g = run.commandBuffers?.first, g.n_cb != want {
            return "\(config.name): the backend used \(g.n_cb) command buffers, not \(want)"
        }
        if config.profileOps && run.opProfile == nil {
            return "\(config.name): no op profile was written"
        }
        return nil
    }

    /// A timed call that paged in more than this was slowed by reading the weights from flash.
    static let timedPageinLimit: Int64 = 256 << 20

    /// A run that says something about a rate: finished, in front, started at the gate, never serious, and no
    /// timed call reading the weights back from flash.
    private func valid(_ r: ScreenRun) -> Bool {
        r.error == nil && !r.interrupted && r.tokensPerSecond > 0 && r.gateReached && thermalRank(r.thermalMax) < 2 &&
            !r.callPageinBytes.contains { $0 > Self.timedPageinLimit }
    }

    private func summarize() {
        var out: [ScreenSummary] = []
        let profiled = Set(spec.configs.filter(\.profileOps).map(\.name))
        for cell in spec.cells {
            let runs = result.runs.filter { $0.cell == cell }
            for config in spec.configs.map(\.name) {
                let mine = runs.filter { $0.config == config }
                let ok = mine.filter(valid)
                var ratios: [Double] = []
                if !profiled.contains(config) && config != spec.reference {
                    for r in ok {
                        if let ref = runs.last(where: { $0.round == r.round && $0.config == spec.reference && valid($0) }) {
                            ratios.append(r.tokensPerSecond / ref.tokensPerSecond)
                        }
                    }
                }
                let failures = mine.filter { $0.error != nil && !$0.interrupted }
                var byError: [String: Int] = [:]
                for f in failures {
                    let key = f.failure?.code.map { "\(f.failure?.domain ?? "") \($0)" } ?? (f.error ?? "?")
                    byError[key, default: 0] += 1
                }
                let longest = mine.compactMap(\.commandBuffers).flatMap { $0 }.flatMap(\.cbs)
                    .filter { $0.status == 4 && $0.gs >= 0 && $0.ge >= 0 }.map { $0.ge - $0.gs }.max()
                let excluded = mine.filter { $0.error == nil && !valid($0) }.count + mine.filter { $0.error != nil && $0.interrupted }.count
                out.append(ScreenSummary(config: config, cell: cell, runs: mine.count, failed: failures.count,
                                         failuresByError: byError, excluded: excluded,
                                         medianTokensPerSecond: median(ok.map(\.tokensPerSecond)), ratioMedian: median(ratios),
                                         ratioMin: ratios.min(), ratioMax: ratios.max(), pairedRounds: ratios.count,
                                         longestCommandBufferMs: longest))
            }
        }
        result.summaries = out
    }

    private func median(_ v: [Double]) -> Double? {
        guard !v.isEmpty else { return nil }
        let s = v.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    func run() {
        defer {
            summarize()
            result.finished = Date()
            save(result)
        }
        save(result)
        var rng = SplitMix64(seed: seed)
        var order = spec.configs
        do {
            for round in 0..<spec.rounds {
                // mirrored rounds: a random order, then the same order reversed
                if round % 2 == 0 { order = spec.configs.shuffled(using: &rng) } else { order.reverse() }
                for name in spec.cells {
                    let cell = Cell(name: name)
                    for (pos, config) in order.enumerated() {
                        for attempt in 1...(spec.retryFailed ? 2 : 1) {
                            here = "\(name) · round \(round + 1) of \(spec.rounds) · setup \(pos + 1) of \(order.count)"
                            let r = try measure(cell: cell, config: config, round: round, position: pos, attempt: attempt)
                            result.runs.append(r)
                            summarize()
                            save(result)
                            if let why = manipulationProblem(r, config) {
                                result.error = "check failed: " + why
                                log("STOPPED: \(why)")
                                return
                            }
                            if r.error == nil || r.interrupted { break }
                        }
                    }
                }
            }
        } catch is CancellationError {
            result.cancelled = true
        } catch {
            result.error = error.localizedDescription
        }
        applyFlags([:])
        summarize()
        for s in result.summaries {
            var line = "\(s.cell) \(s.config): "
            line += s.ratioMedian.map { String(format: "%.3fx vs %@ (%d rounds)", $0, spec.reference, s.pairedRounds) } ??
                (s.medianTokensPerSecond.map { String(format: "%.2f tok/s", $0) } ?? "no valid run")
            line += ", \(s.failed) of \(s.runs) runs failed"
            if let l = s.longestCommandBufferMs { line += String(format: ", longest command buffer %.0f ms", l) }
            log(line)
        }
        // profiled configurations: the op kinds whose GPU time differs most between the first two
        let profiles = spec.configs.filter(\.profileOps).compactMap { c in result.runs.last { $0.config == c.name && $0.opProfile != nil } }
        if profiles.count >= 2, let a = profiles[0].opProfile, let b = profiles[1].opProfile {
            var diff: [String: Double] = [:]
            for o in a.ops { diff[o.key, default: 0] -= o.microseconds }
            for o in b.ops { diff[o.key, default: 0] += o.microseconds }
            log(String(format: "op profile: %@ %.1f ms, %@ %.1f ms", profiles[0].config, a.totalMicroseconds / 1000,
                       profiles[1].config, b.totalMicroseconds / 1000))
            for (k, v) in diff.sorted(by: { abs($0.value) > abs($1.value) }).prefix(6) {
                log(String(format: "  %+.2f ms  %@", v / 1000, k))
            }
        }
    }
}

extension ScreenResult {
    /// One line per configuration, for the suite's notes.
    var oneLine: String {
        summaries.map { s -> String in
            var p = s.config
            if let r = s.ratioMedian { p += String(format: " %.3fx", r) }
            if s.failed > 0 { p += " (\(s.failed)/\(s.runs) failed)" }
            return p
        }.joined(separator: "; ")
    }
}

/// A small seeded generator, so a screen's order can be reproduced from its recorded seed.
struct SplitMix64: RandomNumberGenerator {
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
