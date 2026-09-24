import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var state = BenchState()
    @ObservedObject private var downloads = Downloader.shared
    @Environment(\.scenePhase) private var scenePhase
    @State private var importing = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Device") {
                    LabeledContent("Hardware", value: state.device.machine)
                    LabeledContent("GPU", value: "\(state.device.gpuName) (Apple\(state.device.highestAppleFamily))")
                    LabeledContent("Memory", value: gb(state.device.physicalMemoryBytes))
                    LabeledContent("App can still use", value: state.device.appAvailableMemoryBytes > 0 ? gb(state.device.appAvailableMemoryBytes) : "n/a")
                }

                Section {
                    ForEach(state.models, id: \.self) { url in
                        let need = estimatedNeedBytes(modelFileBytes: state.size(url))
                        let avail = state.device.appAvailableMemoryBytes
                        Button {
                            state.load(url)
                        } label: {
                            VStack(alignment: .leading) {
                                Text(url.lastPathComponent).font(.body)
                                Text("\(gb(state.size(url))) file, ~\(gb(need)) needed" + (avail > 0 ? (need < avail ? "  ✓ likely fits" : "  ✗ likely too large") : ""))
                                    .font(.caption).foregroundStyle(avail > 0 && need >= avail ? .red : .secondary)
                            }
                        }
                        .disabled(state.running || state.loading)
                    }
                    Button("Import a .gguf…") { importing = true }.disabled(state.running || state.loading)
                } header: {
                    Text("Models")
                } footer: {
                    Text(state.status)
                }

                Section {
                    ForEach(Catalog.models) { m in
                        let url = state.documents.appendingPathComponent(m.file)
                        // compare names: the listing and `documents` can differ in form (/var vs /private/var)
                        let installed = state.models.contains { $0.lastPathComponent == m.file }
                        DownloadRow(model: m, installed: installed, verified: installed && VerifiedMark.get(url) == m.sha256,
                                    available: state.device.appAvailableMemoryBytes, busy: state.running, downloads: downloads)
                    }
                    Toggle("Use cellular data", isOn: $downloads.allowCellular)
                } header: {
                    Text("Download from Hugging Face")
                } footer: {
                    VStack(alignment: .leading, spacing: 4) {
                        if !downloads.message.isEmpty { Text(downloads.message).foregroundStyle(.green) }
                        Text("PrismML's files at the revisions the Mac studies used. A download is installed only if its size and SHA-256 match; \"Verify\" checks a model you copied in. Downloads continue while the phone is locked. Free space: \(gb(downloads.freeSpace)).")
                    }
                }

                if let engine = state.engine {
                    Section("Arms (\(engine.weightType))") {
                        Picker("A", selection: $state.armA) {
                            ForEach(Presets.all(for: engine.weightType)) { Text($0.name).tag($0) }
                        }
                        Picker("B", selection: $state.armB) {
                            ForEach(Presets.all(for: engine.weightType)) { Text($0.name).tag($0) }
                        }
                    }
                    .disabled(state.running)
                    Section("Protocol") {
                        ForEach(defaultCells) { cell in
                            Toggle(cell.name, isOn: Binding(
                                get: { state.cells.contains(cell.name) },
                                set: { on in if on { state.cells.insert(cell.name) } else { state.cells.remove(cell.name) } }))
                        }
                        Stepper("Quartets per cell: \(state.cycles)", value: $state.cycles, in: 1...6)
                        Stepper(String(format: "Cooldown: %.0f s", state.cooldown), value: $state.cooldown, in: 0...60, step: 2)
                        Stepper(String(format: "Spread gate: %.2f", state.gate), value: $state.gate, in: 1.05...2.0, step: 0.05)
                    }
                    .disabled(state.running)
                    Section {
                        if state.running {
                            Button("Stop after this observation", role: .destructive) { state.stop() }
                        } else {
                            Button("Run A-B-B-A") { state.start() }.disabled(state.cells.isEmpty || downloads.busy || state.loading)
                        }
                    } footer: {
                        Text(downloads.busy ? "Wait for downloads and checks to finish (or cancel them): they would skew the timings."
                             : "Keep the app in the foreground and the phone on a stable surface; thermal state is recorded with every observation.")
                    }
                }

                if let r = state.result {
                    Section("Results: \(r.armB.name) vs \(r.armA.name)") {
                        ForEach(r.summaries, id: \.cell) { s in
                            VStack(alignment: .leading) {
                                if let geo = s.speedupGeomean, let a = s.aMean, let b = s.bMean {
                                    Text(String(format: "%@  %.2f → %.2f tok/s   ×%.3f", s.cell, a, b, geo)).font(.body.monospaced())
                                } else {
                                    Text("\(s.cell)  no accepted quartet").font(.body.monospaced())
                                }
                                Text(String(format: "range %.3f–%.3f · %d accepted, %d rejected%@%@", s.speedupMin ?? 0, s.speedupMax ?? 0,
                                            s.acceptedQuartets, s.rejectedQuartets, s.complete ? "" : " · incomplete",
                                            s.tokenMismatchQuartets > 0 ? " · tokens differ" : ""))
                                    .font(.caption).foregroundStyle(s.complete ? Color.secondary : Color.orange)
                            }
                        }
                        LabeledContent("Peak footprint", value: gb(r.peakFootprintBytes))
                        if let url = state.resultURL { ShareLink("Share JSON", item: url) }
                    }
                }

                if !state.log.isEmpty {
                    Section("Log") {
                        ForEach(Array(state.log.suffix(200).enumerated()), id: \.offset) { _, line in
                            Text(line).font(.caption.monospaced())
                        }
                    }
                }
            }
            .navigationTitle("BonsaiBench")
            .onChange(of: scenePhase) { _, phase in AppActivity.shared.set(active: phase == .active) }
            .onAppear { state.refresh() }
            .refreshable { state.refresh() }
            .fileImporter(isPresented: $importing, allowedContentTypes: [.data]) { res in
                if case .success(let url) = res { state.importModel(url) }
            }
        }
    }
}

struct DownloadRow: View {
    let model: CatalogModel
    let installed: Bool
    let verified: Bool
    let available: UInt64
    let busy: Bool
    @ObservedObject var downloads: Downloader

    var body: some View {
        let need = estimatedNeedBytes(modelFileBytes: model.bytes)
        let fit = available > 0 ? (need < available ? " · likely fits" : " · likely too large") : ""
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.title)
                    Text("\(gb(model.bytes)) · ~\(gb(need)) needed\(fit)")
                        .font(.caption).foregroundStyle(available > 0 && need >= available ? .red : .secondary)
                }
                Spacer()
                switch downloads.phase[model.file] {
                case .downloading?:
                    Button("Cancel", role: .destructive) { downloads.cancel(model.file) }.buttonStyle(.borderless)
                case .verifying?:
                    ProgressView()
                case .failed?:
                    Button("Retry") { downloads.start(model) }.buttonStyle(.borderless).disabled(busy)
                case nil:
                    if verified {
                        Label("Verified", systemImage: "checkmark.seal.fill").foregroundStyle(.green).font(.callout)
                    } else if installed {
                        Button("Verify") { downloads.verifyInstalled(model) }.buttonStyle(.borderless).disabled(busy)
                    } else {
                        Button("Get") { downloads.start(model) }.buttonStyle(.borderless).disabled(busy)
                    }
                }
            }
            switch downloads.phase[model.file] {
            case .downloading(let f)?:
                ProgressView(value: f)
                Text(String(format: "%.1f%% of %@", f * 100, gb(model.bytes)))
                    .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
            case .verifying?:
                Text("Checking size and SHA-256…").font(.caption2).foregroundStyle(.secondary)
            case .failed(let msg)?:
                Text(msg).font(.caption2).foregroundStyle(.red)
            case nil:
                if installed && !verified {
                    Text("In the app but not checked against the published SHA-256.").font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }
}
