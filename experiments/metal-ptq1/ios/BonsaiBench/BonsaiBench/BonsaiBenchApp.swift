import SwiftUI
import UniformTypeIdentifiers

@main
struct BonsaiBenchApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

@MainActor
final class BenchState: ObservableObject {
    @Published var models: [URL] = []
    @Published var selected: URL?
    @Published var engine: Engine?
    @Published var status = "Download a model below, or copy a .gguf into the app's Documents (Finder or Files)."
    @Published var log: [String] = []

    /// A line of the on-screen log, also appended to Documents/bonsaibench-log.txt so that a Mac can read it
    /// (devicectl device copy from) while a suite runs, without touching the app.
    func addLog(_ line: String) {
        log.append(line)
        LogFile.append(line)
    }
    @Published var progress = ""
    @Published var running = false
    @Published var loading = false
    @Published var armA = Presets.upstream
    @Published var armB = Presets.upstream
    @Published var cells: Set<String> = Set(defaultCells.map(\.name))
    @Published var cycles = 3
    @Published var cooldown = 60.0
    @Published var waitForNominal = false
    @Published var promptUbatch = 512
    @Published var gate = 1.20
    @Published var result: RunResult?
    @Published var resultURL: URL?
    @Published var device = DeviceInfo.capture()
    @Published var thermal = thermalStateName()      // live, so it is visible before starting a study
    private var thermalObserver: NSObjectProtocol?
    private var study: Study?
    private var screen: Screen?
    private var lastLoadHadMTP: Bool?

    // Suite: the studies run in order; `suiteStep` is the one running (nil when no suite runs). Its state is
    // persisted (SuiteRecord) so that a suite the app died in can be resumed with what it had done.
    @Published var suite = Suites.phone              // the phone suite, or one quick test while it runs
    @Published var quickTitle: String?               // the quick test running (it borrows the suite runner)
    private var savedSuite: (included: Set<Int>, outcome: [Int: String], notes: [String], resume: Int?)?
    @Published var suiteIncluded: Set<Int> = Set(Suites.phone.indices)
    @Published var suiteStep: Int?
    @Published var suiteOutcome: [Int: String] = [:]   // done, incomplete, failed, skipped, did not fit
    @Published var suiteResumeAt: Int?                 // offered after the app died during a suite
    @Published var suiteNotes: [String] = []           // what went wrong, per study (each study clears the log)
    private var suiteStopped = false
    private var suiteKills: [Int: Int] = [:]           // study -> times the app died during it
    private var manualSettings: (cells: Set<String>, cycles: Int, cooldown: Double, nominal: Bool, gate: Double, ubatch: Int)?

    private struct SuiteRecord: Codable {
        var titles: [String]?                          // the suite this record belongs to (studies by index)
        var included: [Int]
        var step: Int?                                 // the study running; set while it runs
        var deathCounted = false                       // this launch already counted the app dying in `step`
        var kills: [Int: Int]
        var outcome: [Int: String]
        var notes: [String]
    }

    private func saveSuite(step: Int?, deathCounted: Bool = false) {
        guard quickTitle == nil else { return }    // a quick test never touches the phone suite's saved state
        let r = SuiteRecord(titles: suite.map(\.title), included: Array(suiteIncluded), step: step, deathCounted: deathCounted, kills: suiteKills,
                            outcome: suiteOutcome, notes: suiteNotes)
        UserDefaults.standard.set(try? JSONEncoder().encode(r), forKey: "suite")
    }

    init() {
        _ = launchEnvironment   // capture before any arm changes the environment
        Downloader.shared.onFinished = { [weak self] in self?.refresh() }
        Downloader.shared.restore(autostart: true)
        // iOS kills an app that exceeds its memory limit without a crash report. The marker written before a
        // load says which model was loading when the app died; keep a record of it.
        let killedLoading = UserDefaults.standard.string(forKey: "loadingModel")
        if let m = killedLoading {
            writeDidNotFit(model: m)
            UserDefaults.standard.removeObject(forKey: "loadingModel")
        }
        // A suite the app died in: count the death once, decide what to skip, and offer to resume.
        if let data = UserDefaults.standard.data(forKey: "suite"),
           let rec = try? JSONDecoder().decode(SuiteRecord.self, from: data), let step = rec.step,
           rec.titles == suite.map(\.title) {   // a record from a different suite (older app) is not resumed
            suiteIncluded = Set(rec.included)
            suiteKills = rec.kills
            suiteOutcome = rec.outcome
            suiteNotes = rec.notes
            if !rec.deathCounted {
                suiteKills[step, default: 0] += 1
                if let m = killedLoading {
                    // it does not fit: skip every remaining study of that model
                    for j in suite.indices where j >= step && suite[j].model == m { suiteOutcome[j] = "did not fit" }
                    suiteNotes.append("\(modelTitle(m)): the app was stopped while loading it, most likely out of memory; its studies are skipped.")
                } else if suiteKills[step, default: 0] >= 2 {
                    suiteOutcome[step] = "failed"
                    suiteNotes.append("\"\(suite[step].title)\": the app was stopped during it twice (out of memory?); skipped.")
                } else {
                    suiteNotes.append("\"\(suite[step].title)\": the app was stopped during it; resuming runs it again.")
                }
                saveSuite(step: step, deathCounted: true)
            }
            suiteResumeAt = suiteRemaining(from: step).first
        }
        if let m = killedLoading {
            status = "The app was stopped while loading \(m), most likely out of memory: it does not fit on this phone."
        } else if let unfinished = Self.unfinishedStudy(in: documents) {
            status = "The last study did not finish (\(unfinished)); its quartets up to then are saved in that file."
        }
        thermalObserver = NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification,
                                                                 object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.thermal = thermalStateName() }
        }
        autorun()
    }

    /// Unattended study, for running from a Mac:
    ///   xcrun devicectl device process launch --device <id> --terminate-existing --environment-variables \
    ///     '{"BONSAIBENCH_AUTORUN":"{\"model\":\"Bonsai-27B-Q1_0.gguf\",\"a\":\"upstream\",\"b\":\"only SMALLM_MM\",\"cells\":[\"pp512\"],\"cycles\":1}"}' \
    ///     dev.bonsaibench.app
    /// Arms are preset names as shown in the app; cells may include any tgN/chatN/ppK; cycles and cooldown are
    /// optional. The phone must stay unlocked with the app in front. The result is written to Documents.
    ///
    /// The whole suite: '{"BONSAIBENCH_AUTORUN":"{\"suite\":true}"}' (optionally \"from\": <study index>).
    private func autorun() {
        guard let spec = ProcessInfo.processInfo.environment["BONSAIBENCH_AUTORUN"] else { return }
        guard let o = (try? JSONSerialization.jsonObject(with: Data(spec.utf8))) as? [String: Any] else {
            status = "BONSAIBENCH_AUTORUN is not valid JSON"
            return
        }
        if o["suite"] as? Bool == true {
            // resume a suite the app died in unless told where to start
            if let from = o["from"] as? Int { startSuite(from: from) } else if let r = suiteResumeAt { startSuite(from: r, resuming: true) } else { startSuite() }
            return
        }
        guard let model = o["model"] as? String else {
            status = "BONSAIBENCH_AUTORUN needs a \"model\" (or \"suite\": true)"
            return
        }
        load(documents.appendingPathComponent(model)) { [weak self] engine in
            guard let self, let engine else { return }
            let arms = Presets.all(for: engine.weightType, mtp: engine.hasMTP)
            let a = arms.first { $0.name == (o["a"] as? String ?? "upstream") }
            let defaultB = engine.hasMTP ? Presets.recommended(for: engine.weightType).name + " + MTP" : Presets.recommended(for: engine.weightType).name
            let b = arms.first { $0.name == (o["b"] as? String ?? defaultB) }
            guard let a, let b else {
                self.status = "Autorun: unknown arm; presets are " + arms.map(\.name).joined(separator: ", ")
                return
            }
            self.armA = a
            self.armB = b
            if let cells = o["cells"] as? [String] { self.cells = Set(cells) }
            if let n = o["cycles"] as? Int { self.cycles = n }
            if let c = o["cooldown"] as? Double { self.cooldown = c }
            if let w = o["waitForNominal"] as? Bool { self.waitForNominal = w }
            if let u = o["ubatch"] as? Int, [128, 256, 512].contains(u) { self.promptUbatch = u }
            self.start()
        }
    }

    private static func unfinishedStudy(in dir: URL) -> String? {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        guard let last = files.filter({ $0.hasPrefix("bonsaibench-") && $0.hasSuffix(".json") }).max(),
              let data = try? Data(contentsOf: dir.appendingPathComponent(last)),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj["finished"] == nil ? last : nil
    }

    var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    func refresh() {
        let files = (try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        models = files.filter { $0.pathExtension.lowercased() == "gguf" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        // Models copied in with Finder or devicectl would otherwise go into iCloud / Finder backups.
        for var url in models {
            var rv = URLResourceValues()
            rv.isExcludedFromBackup = true
            try? url.setResourceValues(rv)
        }
        device.appAvailableMemoryBytes = UInt64(max(0, availableMemory()))
        Downloader.shared.updateFreeSpace()
    }

    func size(_ url: URL) -> UInt64 {
        UInt64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }

    func importModel(_ url: URL) {
        let dest = documents.appendingPathComponent(url.lastPathComponent)
        // The picker also shows this app's own Documents; importing a file onto itself is a no-op.
        if url.resolvingSymlinksInPath().standardizedFileURL == dest.resolvingSymlinksInPath().standardizedFileURL {
            status = "\(dest.lastPathComponent) is already in the app."
            refresh()
            return
        }
        // Copy under a name the model list ignores, then rename over any existing file in one step, so a
        // partial copy never shows up as a model.
        let part = documents.appendingPathComponent(url.lastPathComponent + ".import.part")
        status = "Copying \(url.lastPathComponent)…"
        loading = true
        Task.detached {
            let ok = url.startAccessingSecurityScopedResource()
            defer { if ok { url.stopAccessingSecurityScopedResource() } }
            do {
                try? FileManager.default.removeItem(at: part)
                try FileManager.default.copyItem(at: url, to: part)
                guard rename(part.path, dest.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
                await MainActor.run { self.loading = false; self.status = "Imported \(dest.lastPathComponent)"; self.refresh() }
            } catch {
                try? FileManager.default.removeItem(at: part)
                await MainActor.run { self.loading = false; self.status = "Import failed: \(error.localizedDescription)" }
            }
        }
    }

    func load(_ url: URL, then: ((Engine?) -> Void)? = nil) {
        guard !loading, !running else { then?(nil); return }
        study = nil          // it holds the engine; the old model must be freed before the next one loads
        let old = WeakEngine(engine)
        engine = nil
        selected = url
        loading = true
        status = "Loading \(url.lastPathComponent)…"
        UserDefaults.standard.set(url.lastPathComponent, forKey: "loadingModel")
        UserDefaults.standard.synchronize()
        Task.detached {
            // Wait (bounded) until the previous model is really gone, then run one GPU command: Metal releases
            // residency-set memory lazily, and two 6-7 GB models must not be held at once.
            let t0 = Date()
            while old.engine != nil && Date().timeIntervalSince(t0) < 5 { try? await Task.sleep(nanoseconds: 50_000_000) }
            flushGPU()
            do {
                let e = try Engine(path: url.path)
                await MainActor.run {
                    UserDefaults.standard.removeObject(forKey: "loadingModel")
                    self.loading = false
                    self.engine = e
                    self.armA = Presets.upstream
                    self.armB = Presets.recommended(for: e.weightType)
                    if e.hasMTP, let b = Presets.all(for: e.weightType, mtp: true).first(where: { $0.draft > 0 && $0.name != "upstream + MTP" }) {
                        self.armB = b      // an MTP model is loaded to measure MTP: upstream plain vs our flags + MTP
                    }
                    // switch the default cells only when moving between MTP and plain models
                    if self.suiteStep == nil && self.lastLoadHadMTP != e.hasMTP { self.cells = defaultSelectedCells(mtp: e.hasMTP) }
                    self.lastLoadHadMTP = e.hasMTP
                    self.status = "Loaded \(e.description) (\(e.weightType)); footprint \(gb(physicalFootprint())), available \(gb(UInt64(max(0, availableMemory()))))"
                    then?(e)
                }
            } catch {
                await MainActor.run {
                    UserDefaults.standard.removeObject(forKey: "loadingModel")
                    self.loading = false
                    self.status = "Load failed: \(error.localizedDescription)"
                    then?(nil)
                }
            }
        }
    }

    /// Start a study with the current settings; `then` runs on the main queue when it has finished, with
    /// whether its results were saved. Returns false (and says why) if it cannot start.
    @discardableResult
    func start(thermalWaitLimit: Double = 300, attempts: Int = 3, then: ((RunResult, Bool) -> Void)? = nil) -> Bool {
        guard let engine else { status = "Load a model first."; return false }
        guard !running else { return false }
        guard !Downloader.shared.busy else { status = "Wait for downloads and hash checks to finish."; return false }
        // the default cells in their usual order, then any others (autorun) by name
        let known = Set(defaultCells.map(\.name))
        let chosen = defaultCells.filter { cells.contains($0.name) } +
            cells.subtracting(known).sorted().map { Cell(name: $0) }.filter { ["tg", "chat", "gen", "pp"].contains($0.kind) && $0.count > 0 }
        if (armA.draft > 0 || armB.draft > 0) && chosen.contains(where: { $0.kind != "gen" }) {
            status = "An MTP arm only measures gen cells: switch off the other cells, or pick arms without MTP."
            return false
        }
        guard !chosen.isEmpty else { status = "Choose at least one cell."; return false }
        running = true
        log = []
        result = nil
        resultURL = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let url = documents.appendingPathComponent("bonsaibench-\(stamp).json")
        var saveError: String?
        let s = Study(engine: engine, armA: armA, armB: armB, cells: chosen, cycles: cycles, cooldown: cooldown,
                      waitForNominal: waitForNominal, thermalWaitLimit: thermalWaitLimit, promptUbatch: promptUbatch,
                      gate: gate, attempts: attempts,
                      log: { line in Task { @MainActor in self.addLog(line) } },
                      progress: { p in Task { @MainActor in self.progress = p } },
                      save: { r in
                          // Atomic, after every quartet: a study killed by iOS keeps what it measured.
                          do { try JSONEncoder.pretty.encode(r).write(to: url, options: .atomic); saveError = nil }
                          catch { saveError = error.localizedDescription }
                      })
        study = s
        addLog("\(modelTitle(engine.path.split(separator: "/").last.map(String.init) ?? "")) · A = \(armA.name) · B = \(armB.name)")
        // A dedicated thread at user-initiated priority: CPU-side graph encoding stays on performance cores.
        let t = Thread {
            s.run()
            DispatchQueue.main.async {
                self.result = s.result
                self.resultURL = saveError == nil ? url : nil
                self.running = false
                self.progress = ""
                self.study = nil
                UIApplication.shared.isIdleTimerDisabled = false
                let what = s.result.error.map { "Stopped by an error: \($0)." } ?? (s.result.cancelled ? "Stopped." : "Done.")
                self.status = what + (saveError.map { " Could not save the results: \($0)" } ?? " Results saved to Documents/\(url.lastPathComponent).")
                then?(s.result, saveError == nil)
            }
        }
        t.qualityOfService = .userInitiated
        t.stackSize = 8 << 20
        t.start()
        return true
    }

    func stop() {
        suiteStopped = true
        study?.cancelled = true
        screen?.cancelled = true
        if loading && suiteStep != nil { status = "Stopping the suite once the model has finished loading…" }
    }

    // MARK: quick tests

    /// Run one study with the suite runner (it loads the model and sets arms and protocol), then put the
    /// phone suite back as it was.
    func runQuick(_ spec: StudySpec) {
        guard !running, !loading, suiteStep == nil else {
            status = "Wait for the running study or model load to finish."
            return
        }
        savedSuite = (suiteIncluded, suiteOutcome, suiteNotes, suiteResumeAt)
        quickTitle = spec.title
        suite = [spec]
        suiteIncluded = [0]
        startSuite(from: 0)
    }

    // MARK: suite

    /// Included studies from `from` on that have not been ruled out (did not fit, died twice).
    func suiteRemaining(from: Int) -> [Int] {
        suite.indices.filter { $0 >= from && suiteIncluded.contains($0) && !["did not fit", "failed"].contains(suiteOutcome[$0] ?? "") }
    }

    func startSuite(from first: Int = 0, resuming: Bool = false) {
        guard !running, !loading, suiteStep == nil else {
            status = "The suite cannot start while a study or a model load is running."
            return
        }
        suiteStopped = false
        if !resuming {
            suiteOutcome = [:]
            suiteNotes = []
            suiteKills = [:]
        }
        suiteResumeAt = nil
        manualSettings = (cells, cycles, cooldown, waitForNominal, gate, promptUbatch)
        suiteNext(first)
    }

    /// Run the next remaining study from `from` on, then the rest. Each study loads its model if another one is
    /// loaded, sets the arms and protocol from its spec, and saves its own result file.
    private func suiteNext(_ from: Int) {
        guard !suiteStopped, let i = suiteRemaining(from: from).first else {
            suiteFinished()
            return
        }
        suiteStep = i
        saveSuite(step: i)
        UIApplication.shared.isIdleTimerDisabled = true
        let spec = suite[i]
        let skip = { (why: String) in
            self.suiteOutcome[i] = "skipped"
            self.suiteNotes.append("\"\(spec.title)\" skipped: \(why)")
            // next runloop turn, so nothing from this step still holds the engine
            DispatchQueue.main.async { self.suiteNext(i + 1) }
        }
        let url = documents.appendingPathComponent(spec.model)
        guard FileManager.default.fileExists(atPath: url.path) else { return skip("\(spec.model) is not in the app") }
        let go = { (engine: Engine) in
            if let screenSpec = spec.screen {
                self.whenDownloadsIdle {
                    let started = self.startScreen(spec, screenSpec, engine) { r, saved in
                        self.recordScreenOutcome(i, spec, r, saved)
                        DispatchQueue.main.async { self.suiteNext(i + 1) }
                    }
                    if !started { skip("the diagnostic could not start (\(self.status))") }
                }
                return
            }
            let arms = Presets.all(for: engine.weightType, mtp: engine.hasMTP)
            guard let a = arms.first(where: { $0.name == spec.a }), let b = arms.first(where: { $0.name == spec.b }) else {
                return skip("no arm \"\(spec.a)\" or \"\(spec.b)\" for \(engine.weightType)")
            }
            self.whenDownloadsIdle {
                self.armA = a
                self.armB = b
                self.cells = Set(spec.cells)
                self.cycles = spec.cycles
                self.cooldown = spec.cooldown
                self.waitForNominal = spec.waitForNominal
                self.gate = spec.gate
                self.promptUbatch = spec.ubatch
                let started = self.start(thermalWaitLimit: spec.thermalWaitLimit, attempts: spec.attempts) { r, saved in
                    self.recordOutcome(i, spec, r, saved)
                    DispatchQueue.main.async { self.suiteNext(i + 1) }
                }
                if !started { skip("the study could not start (\(self.status))") }
            }
        }
        if let e = engine, selected?.lastPathComponent == spec.model {
            go(e)
        } else {
            load(url) { e in
                if self.suiteStopped { return self.suiteFinished() }
                if let e { go(e) } else { skip("the model did not load (\(self.status))") }
            }
        }
    }

    /// Downloads and hash checks skew timings and block starting: wait for them (the user may cancel them).
    private func whenDownloadsIdle(_ go: @escaping () -> Void) {
        if suiteStopped { return suiteFinished() }
        guard Downloader.shared.busy else { return go() }
        progress = "Waiting for downloads and hash checks to finish (cancel them below to continue)"
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { self.whenDownloadsIdle(go) }
    }

    /// Start a diagnostic screen on the loaded model; its result goes to Documents/bonsaidiag-<time>.json.
    private func startScreen(_ spec: StudySpec, _ screenSpec: ScreenSpec, _ engine: Engine,
                             then: @escaping (ScreenResult, Bool) -> Void) -> Bool {
        guard !running else { return false }
        running = true
        log = []
        result = nil
        resultURL = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let url = documents.appendingPathComponent("bonsaidiag-\(stamp).json")
        var saveError: String?
        let s = Screen(engine: engine, title: spec.title, spec: screenSpec, cooldown: spec.cooldown,
                       waitForNominal: spec.waitForNominal, thermalWaitLimit: spec.thermalWaitLimit,
                       log: { line in Task { @MainActor in self.addLog(line) } },
                       progress: { p in Task { @MainActor in self.progress = p } },
                       save: { r in
                           // atomic, after every run: a screen killed by iOS keeps what it measured
                           do { try JSONEncoder.pretty.encode(r).write(to: url, options: .atomic); saveError = nil }
                           catch { saveError = error.localizedDescription }
                       })
        screen = s
        addLog("\(spec.title) · \(modelTitle(spec.model)) · \(screenSpec.configs.count) configurations × \(screenSpec.rounds) rounds")
        let t = Thread {
            s.run()
            DispatchQueue.main.async {
                self.running = false
                self.progress = ""
                self.screen = nil
                UIApplication.shared.isIdleTimerDisabled = false
                let what = s.result.error.map { "Stopped by an error: \($0)." } ?? (s.result.cancelled ? "Stopped." : "Done.")
                self.status = what + (saveError.map { " Could not save the results: \($0)" } ?? " Results saved to Documents/\(url.lastPathComponent).")
                then(s.result, saveError == nil)
            }
        }
        t.qualityOfService = .userInitiated
        t.stackSize = 8 << 20
        t.start()
        return true
    }

    private func recordScreenOutcome(_ i: Int, _ spec: StudySpec, _ r: ScreenResult, _ saved: Bool) {
        let failed = r.runs.filter { $0.error != nil }.count
        if r.cancelled {
            suiteNotes.append("\"\(spec.title)\" stopped before it finished; its runs so far are saved.")
        } else if !saved {
            suiteOutcome[i] = "failed"
            suiteNotes.append("\"\(spec.title)\": the results could not be saved.")
        } else if let e = r.error {
            suiteOutcome[i] = "failed"
            suiteNotes.append("\"\(spec.title)\" failed: \(e)")
        } else {
            suiteOutcome[i] = "done"
            suiteNotes.append("\"\(spec.title)\": \(r.oneLine)" + (failed > 0 ? " (\(failed) of \(r.runs.count) runs failed, recorded)." : "."))
        }
        saveSuite(step: i)
    }

    /// Done only if every cell has an accepted quartet for every cycle; otherwise say what is missing.
    private func recordOutcome(_ i: Int, _ spec: StudySpec, _ r: RunResult, _ saved: Bool) {
        let incomplete = r.summaries.filter { !$0.complete }.map(\.cell)
        let failedRuns = (r.quartets.flatMap(\.observations) + (r.unfinishedQuartet ?? [])).filter { $0.error != nil }.count
        if r.cancelled {
            suiteNotes.append("\"\(spec.title)\" stopped before it finished; its runs so far are saved.")
        } else if !saved {
            suiteOutcome[i] = "failed"
            suiteNotes.append("\"\(spec.title)\": the results could not be saved.")
        } else if let e = r.error {
            suiteOutcome[i] = "failed"
            suiteNotes.append("\"\(spec.title)\" failed: \(e)")
        } else if !incomplete.isEmpty {
            suiteOutcome[i] = "incomplete"
            suiteNotes.append("\"\(spec.title)\": no accepted quartet for every cycle in \(incomplete.joined(separator: ", "))" +
                              (failedRuns > 0 ? "; \(failedRuns) runs failed" : "") + ".")
        } else {
            suiteOutcome[i] = "done"
        }
        saveSuite(step: i)
    }

    private func suiteFinished() {
        guard suiteStep != nil || !suiteStopped || manualSettings != nil else { return }
        let stopped = suiteStopped
        suiteStep = nil
        saveSuite(step: nil)
        progress = ""
        UIApplication.shared.isIdleTimerDisabled = false
        if let m = manualSettings {
            (cells, cycles, cooldown, waitForNominal, gate, promptUbatch) = (m.cells, m.cycles, m.cooldown, m.nominal, m.gate, m.ubatch)
            manualSettings = nil
        }
        if let title = quickTitle {
            let outcome = suiteOutcome[0]
            let notes = suiteNotes
            quickTitle = nil
            suite = Suites.phone
            if let s = savedSuite {
                (suiteIncluded, suiteOutcome, suiteNotes, suiteResumeAt) = (s.included, s.outcome, s.notes, s.resume)
            }
            savedSuite = nil
            status = (stopped ? "Stopped: " : "Finished: ") + title + (outcome.map { " (\($0))" } ?? "") +
                (notes.isEmpty ? ". Results below; the file is in Documents." : ". " + notes.joined(separator: " "))
            return
        }
        let done = suiteOutcome.values.filter { $0 == "done" }.count
        status = (stopped ? "Suite stopped" : "Suite finished") +
            ": \(done) of \(suiteIncluded.count) studies complete; each study's results are in Documents."
    }

    /// A small result file for a model that iOS killed the app while loading: the "does it fit" answer.
    private func writeDidNotFit(model: String) {
        let record: [String: Any] = [
            "tool": "BonsaiBench 2", "event": "the app was stopped while loading this model (most likely out of memory)",
            "model": model, "time": ISO8601DateFormatter().string(from: Date()),
            "device": (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(DeviceInfo.capture()))) ?? [:],
            "build": ["sourceRevision": BuildInfo.current.sourceRevision, "frameworkSHA256": BuildInfo.current.frameworkSHA256],
        ]
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        if let data = try? JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: documents.appendingPathComponent("bonsaibench-\(stamp)-did-not-fit.json"), options: .atomic)
        }
    }

}

extension JSONEncoder {
    static var pretty: JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        e.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return e
    }
}

func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }

/// Lets a load wait until the previous engine has actually been freed.
final class WeakEngine: @unchecked Sendable {
    weak var engine: Engine?
    init(_ e: Engine?) { engine = e }
}

/// Documents/bonsaibench-log.txt: every on-screen log line with a timestamp, written off the main thread and
/// restarted when it passes 5 MB.
enum LogFile {
    static let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("bonsaibench-log.txt")
    private static let queue = DispatchQueue(label: "bonsaibench.logfile", qos: .utility)

    static func append(_ line: String) {
        let data = Data((ISO8601DateFormatter().string(from: Date()) + " " + line + "\n").utf8)
        queue.async {
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
            if size > 5_000_000 { try? FileManager.default.removeItem(at: url) }
            if let h = try? FileHandle(forWritingTo: url) {
                defer { try? h.close() }
                _ = try? h.seekToEnd()
                try? h.write(contentsOf: data)
            } else {
                try? data.write(to: url)
            }
        }
    }
}
