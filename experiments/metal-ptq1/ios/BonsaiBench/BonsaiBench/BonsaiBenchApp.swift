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
    @Published var running = false
    @Published var loading = false
    @Published var armA = Presets.upstream
    @Published var armB = Presets.upstream
    @Published var cells: Set<String> = Set(defaultCells.map(\.name))
    @Published var cycles = 3
    @Published var cooldown = 8.0
    @Published var waitForNominal = false
    @Published var promptUbatch = 512
    @Published var gate = 1.20
    @Published var result: RunResult?
    @Published var resultURL: URL?
    @Published var device = DeviceInfo.capture()
    private var study: Study?

    init() {
        _ = launchEnvironment   // capture before any arm changes the environment
        Downloader.shared.onFinished = { [weak self] in self?.refresh() }
        Downloader.shared.restore(autostart: true)
        // iOS kills an app that exceeds its memory limit without a crash report; say so on the next launch.
        if let m = UserDefaults.standard.string(forKey: "loadingModel") {
            status = "The app was stopped while loading \(m), most likely out of memory: it does not fit on this phone."
            UserDefaults.standard.removeObject(forKey: "loadingModel")
        } else if let unfinished = Self.unfinishedStudy(in: documents) {
            status = "The last study did not finish (\(unfinished)); its quartets up to then are saved in that file."
        }
        autorun()
    }

    /// Unattended study, for running from a Mac:
    ///   xcrun devicectl device process launch --device <id> --terminate-existing --environment-variables \
    ///     '{"BONSAIBENCH_AUTORUN":"{\"model\":\"Bonsai-27B-Q1_0.gguf\",\"a\":\"upstream\",\"b\":\"only SMALLM_MM\",\"cells\":[\"pp512\"],\"cycles\":1}"}' \
    ///     dev.bonsaibench.app
    /// Arms are preset names as shown in the app; cells may include any tgN/chatN/ppK; cycles and cooldown are
    /// optional. The phone must stay unlocked with the app in front. The result is written to Documents.
    private func autorun() {
        guard let spec = ProcessInfo.processInfo.environment["BONSAIBENCH_AUTORUN"] else { return }
        guard let o = (try? JSONSerialization.jsonObject(with: Data(spec.utf8))) as? [String: Any],
              let model = o["model"] as? String else {
            status = "BONSAIBENCH_AUTORUN is not valid JSON with a \"model\""
            return
        }
        load(documents.appendingPathComponent(model)) { [weak self] engine in
            guard let self, let engine else { return }
            let arms = Presets.all(for: engine.weightType)
            let a = arms.first { $0.name == (o["a"] as? String ?? "upstream") }
            let b = arms.first { $0.name == (o["b"] as? String ?? Presets.recommended(for: engine.weightType).name) }
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
            if let u = o["ubatch"] as? Int { self.promptUbatch = u }
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
        guard !loading, !running else { return }
        study = nil          // it holds the engine; the old model must be freed before the next one loads
        engine = nil
        selected = url
        loading = true
        status = "Loading \(url.lastPathComponent)…"
        UserDefaults.standard.set(url.lastPathComponent, forKey: "loadingModel")
        UserDefaults.standard.synchronize()
        Task.detached {
            do {
                let e = try Engine(path: url.path)
                await MainActor.run {
                    UserDefaults.standard.removeObject(forKey: "loadingModel")
                    self.loading = false
                    self.engine = e
                    self.armA = Presets.upstream
                    self.armB = Presets.recommended(for: e.weightType)
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

    func start() {
        guard let engine, !running, !Downloader.shared.busy else { return }
        running = true
        log = []
        result = nil
        resultURL = nil
        UIApplication.shared.isIdleTimerDisabled = true
        // the default cells in their usual order, then any others (autorun) by name
        let known = Set(defaultCells.map(\.name))
        let chosen = defaultCells.filter { cells.contains($0.name) } +
            cells.subtracting(known).sorted().map { Cell(name: $0) }.filter { ["tg", "chat", "pp"].contains($0.kind) && $0.count > 0 }
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let url = documents.appendingPathComponent("bonsaibench-\(stamp).json")
        var saveError: String?
        let s = Study(engine: engine, armA: armA, armB: armB, cells: chosen, cycles: cycles, cooldown: cooldown,
                      waitForNominal: waitForNominal, promptUbatch: promptUbatch,
                      gate: gate, attempts: 3,
                      log: { line in Task { @MainActor in self.log.append(line) } },
                      save: { r in
                          // Atomic, after every quartet: a study killed by iOS keeps what it measured.
                          do { try JSONEncoder.pretty.encode(r).write(to: url, options: .atomic); saveError = nil }
                          catch { saveError = error.localizedDescription }
                      })
        study = s
        log.append("\(modelTitle(engine.path.split(separator: "/").last.map(String.init) ?? "")) · A = \(armA.name) · B = \(armB.name)")
        // A dedicated thread at user-initiated priority: CPU-side graph encoding stays on performance cores.
        let t = Thread {
            s.run()
            DispatchQueue.main.async {
                self.result = s.result
                self.resultURL = saveError == nil ? url : nil
                self.running = false
                self.study = nil
                UIApplication.shared.isIdleTimerDisabled = false
                let what = s.result.error.map { "Stopped by an error: \($0)." } ?? (s.result.cancelled ? "Stopped." : "Done.")
                self.status = what + (saveError.map { " Could not save the results: \($0)" } ?? " Results saved to Documents/\(url.lastPathComponent).")
            }
        }
        t.qualityOfService = .userInitiated
        t.stackSize = 8 << 20
        t.start()
    }

    func stop() { study?.cancelled = true }
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
