import SwiftUI
import UniformTypeIdentifiers

@main
struct BonsaiBenchApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

@MainActor
final class BenchState: ObservableObject {
    @Published var models: [URL] = []
    @Published var selected: URL?
    @Published var engine: Engine?
    @Published var status = "Copy a Bonsai .gguf into the app's Documents (Finder file sharing or Files)."
    @Published var log: [String] = []
    @Published var running = false
    @Published var armA = Presets.upstream
    @Published var armB = Presets.upstream
    @Published var cells: Set<String> = Set(defaultCells.map(\.name))
    @Published var cycles = 3
    @Published var cooldown = 8.0
    @Published var gate = 1.20
    @Published var result: RunResult?
    @Published var resultURL: URL?
    let device = DeviceInfo.capture()
    private var study: Study?

    var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    func refresh() {
        let files = (try? FileManager.default.contentsOfDirectory(at: documents, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        models = files.filter { $0.pathExtension.lowercased() == "gguf" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    func size(_ url: URL) -> UInt64 {
        UInt64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }

    func importModel(_ url: URL) {
        let dest = documents.appendingPathComponent(url.lastPathComponent)
        status = "Copying \(url.lastPathComponent)…"
        Task.detached {
            let ok = url.startAccessingSecurityScopedResource()
            defer { if ok { url.stopAccessingSecurityScopedResource() } }
            do {
                if FileManager.default.fileExists(atPath: dest.path) { try FileManager.default.removeItem(at: dest) }
                try FileManager.default.copyItem(at: url, to: dest)
                await MainActor.run { self.status = "Imported \(dest.lastPathComponent)"; self.refresh() }
            } catch {
                await MainActor.run { self.status = "Import failed: \(error.localizedDescription)" }
            }
        }
    }

    func load(_ url: URL) {
        engine = nil
        selected = url
        status = "Loading \(url.lastPathComponent)…"
        Task.detached {
            do {
                let e = try Engine(path: url.path)
                await MainActor.run {
                    self.engine = e
                    self.armA = Presets.upstream
                    self.armB = Presets.recommended(for: e.weightType)
                    self.status = "Loaded \(e.description) (\(e.weightType)); footprint \(gb(physicalFootprint())), available \(gb(UInt64(max(0, availableMemory()))))"
                }
            } catch {
                await MainActor.run { self.status = "Load failed: \(error.localizedDescription)" }
            }
        }
    }

    func start() {
        guard let engine else { return }
        running = true
        log = []
        result = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let chosen = defaultCells.filter { cells.contains($0.name) }
        let s = Study(engine: engine, armA: armA, armB: armB, cells: chosen, cycles: cycles, cooldown: cooldown,
                      gate: gate, attempts: 3) { line in
            Task { @MainActor in self.log.append(line) }
        }
        study = s
        Task.detached {
            s.run()
            let data = try? JSONEncoder.pretty.encode(s.result)
            let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let url = await self.documents.appendingPathComponent("bonsaibench-\(stamp).json")
            if let data { try? data.write(to: url) }
            await MainActor.run {
                self.result = s.result
                self.resultURL = url
                self.running = false
                UIApplication.shared.isIdleTimerDisabled = false
                self.status = s.result.error.map { "Stopped: \($0)" } ?? "Done. Results saved to Documents."
            }
        }
    }

    func stop() { study?.cancelled = true }
}

extension JSONEncoder {
    static var pretty: JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }
}

func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
