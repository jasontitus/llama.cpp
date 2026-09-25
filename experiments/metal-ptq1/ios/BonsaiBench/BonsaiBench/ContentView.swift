import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var state = BenchState()
    @ObservedObject private var downloads = Downloader.shared
    @Environment(\.scenePhase) private var scenePhase
    @State private var importing = false
    @AppStorage("showCustom") private var showCustom = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Device") {
                    LabeledContent("Hardware", value: state.device.machine)
                    LabeledContent("GPU", value: "\(state.device.gpuName) (Apple\(state.device.highestAppleFamily))")
                    LabeledContent("Memory", value: gb(state.device.physicalMemoryBytes))
                    LabeledContent("App can still use", value: state.device.appAvailableMemoryBytes > 0 ? gb(state.device.appAvailableMemoryBytes) : "n/a")
                    LabeledContent("Thermal state") {
                        Text(state.thermal)
                            .foregroundStyle(state.thermal == "nominal" ? Color.green : state.thermal == "fair" ? Color.orange : Color.red)
                    }
                }

                // The main action: every study still needed, unattended. The list with a switch per study is below
                // the quick tests.
                if state.quickTitle == nil {
                    Section {
                        if let step = state.suiteStep {
                            Text("Study \(step + 1) of \(state.suite.count): \(state.suite[step].title)")
                            if !state.progress.isEmpty { Text(state.progress).font(.callout.monospacedDigit()) }
                            Button("Stop", role: .destructive) { state.stop() }
                        } else {
                            let total = state.suite.indices.filter { state.suiteIncluded.contains($0) }.reduce(0.0) { $0 + state.suite[$1].estimatedSeconds }
                            Button("Run everything we still need (~\(durationText(total)) plus cooling)") { state.startSuite() }
                                .buttonStyle(.borderedProminent)
                                .disabled(state.running || state.loading || downloads.busy || state.suiteIncluded.isEmpty)
                            if let r = state.suiteResumeAt {
                                Button("Resume at study \(r + 1)") { state.startSuite(from: r, resuming: true) }
                                    .disabled(state.running || state.loading || downloads.busy)
                            }
                        }
                    } header: {
                        Text("Unattended run")
                    } footer: {
                        Text("Runs the \(state.suiteIncluded.count) studies listed below the quick tests, in order, and saves one result file per study. Every run starts only when the phone is nominal (cool); the app waits up to an hour for it, so it can run overnight. Keep the phone plugged in, flat, with the app open; the screen stays on. If iOS stops the app, reopen it and resume.")
                    }
                }

                Section {
                    ForEach(Suites.quick, id: \.title) { spec in
                        let have = state.models.contains { $0.lastPathComponent == spec.model }
                        VStack(alignment: .leading, spacing: 6) {
                            Text(spec.title).font(.body)
                            Text("\(modelTitle(spec.model)) · B = \(spec.b) vs A = \(spec.a) · \(spec.cells.joined(separator: ", ")) · \(spec.cycles) quartets · ~\(durationText(spec.estimatedSeconds))")
                                .font(.caption).foregroundStyle(.secondary)
                            if state.quickTitle == spec.title {
                                if !state.progress.isEmpty { Text(state.progress).font(.callout.monospacedDigit()) }
                                Button("Stop", role: .destructive) { state.stop() }
                            } else if !have {
                                Text("Needs \(spec.model) in the app").font(.caption).foregroundStyle(.orange)
                            } else {
                                Button("Run") { state.runQuick(spec) }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(state.running || state.loading || state.suiteStep != nil || downloads.busy)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                } header: {
                    Text("Quick tests")
                } footer: {
                    Text("One tap each: the test loads its model and sets everything. Keep the app open and the phone flat; results appear below and are saved to Documents.")
                }

                Section {
                    if state.quickTitle != nil {
                        Text("A quick test is running.").foregroundStyle(.secondary)
                    } else {
                    ForEach(Array(state.suite.enumerated()), id: \.offset) { i, spec in
                        let have = state.models.contains { $0.lastPathComponent == spec.model }
                        HStack(alignment: .top, spacing: 10) {
                            // static icons only: an animation keeps the GPU compositing during measurements
                            Group {
                                if state.suiteStep == i {
                                    Image(systemName: "play.circle.fill").foregroundStyle(Color.accentColor)
                                } else {
                                    switch state.suiteOutcome[i] {
                                    case "done": Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                                    case "incomplete": Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.orange)
                                    case "failed", "did not fit": Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
                                    case "skipped": Image(systemName: "slash.circle").foregroundStyle(.orange)
                                    default:
                                        if !have { Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange) }
                                        else { Image(systemName: state.suiteIncluded.contains(i) ? "circle" : "minus.circle").foregroundStyle(.secondary) }
                                    }
                                }
                            }
                            .frame(width: 22)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("\(i + 1). \(spec.title)" + (state.suiteOutcome[i].map { " (\($0))" } ?? ""))
                                Text("B = \(spec.b) vs A = \(spec.a) · \(spec.cells.joined(separator: ", ")) · \(spec.cycles) quartets, \(Int(spec.cooldown)) s cooldown\(spec.ubatch != 512 ? " · micro-batch \(spec.ubatch)" : "") · ~\(durationText(spec.estimatedSeconds))" + (have ? "" : " · model not in the app"))
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Toggle("", isOn: Binding(
                                get: { state.suiteIncluded.contains(i) },
                                set: { on in if on { state.suiteIncluded.insert(i) } else { state.suiteIncluded.remove(i) } }))
                                .labelsHidden()
                                .disabled(state.suiteStep != nil)
                        }
                    }
                    ForEach(state.suiteNotes, id: \.self) { Text($0).font(.caption).foregroundStyle(.orange) }
                    }
                } header: {
                    Text("Everything we still need: the studies")
                } footer: {
                    Text("Started with the button at the top. Switch a study off to skip it; a study whose model is not in the app is skipped.")
                }

                Section {
                    if let url = state.selected {
                        HStack(alignment: .top, spacing: 12) {
                            if state.loading {
                                ProgressView()
                            } else if state.engine != nil {
                                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.title2)
                            } else {
                                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange).font(.title2)
                            }
                            VStack(alignment: .leading, spacing: 3) {
                                Text(modelTitle(url.lastPathComponent)).font(.headline)
                                Text(url.lastPathComponent).font(.caption.monospaced()).foregroundStyle(.secondary)
                                if state.loading {
                                    Text("Loading…").font(.caption)
                                } else if let e = state.engine {
                                    Text("\(e.weightType) · \(gb(state.size(url))) · " +
                                         (VerifiedMark.get(url) != nil ? "SHA-256 verified" : "not checked against the published hash"))
                                        .font(.caption).foregroundStyle(.secondary)
                                } else {
                                    Text("Not loaded (see the message below)").font(.caption).foregroundStyle(.orange)
                                }
                            }
                        }
                    } else {
                        Text("No model loaded. Pick one below.").foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Model under test")
                } footer: {
                    Text(state.status)
                }

                Section {
                    Toggle("Custom study (choose model, arms and cells yourself)", isOn: $showCustom)
                }

                if showCustom {
                Section {
                    ForEach(state.models, id: \.self) { url in
                        let need = estimatedNeedBytes(modelFileBytes: state.size(url), mtp: url.lastPathComponent.lowercased().contains("-mtp"))
                        let avail = state.device.appAvailableMemoryBytes
                        let isSelected = state.selected?.lastPathComponent == url.lastPathComponent
                        Button {
                            state.load(url)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(modelTitle(url.lastPathComponent)).foregroundStyle(.primary)
                                    Text(url.lastPathComponent).font(.caption2.monospaced()).foregroundStyle(.secondary)
                                    Text("\(gb(state.size(url))) file, read from flash · ~\(gb(need)) app memory" + (avail > 0 ? (need < avail ? " · likely fits" : " · likely too large") : ""))
                                        .font(.caption).foregroundStyle(avail > 0 && need >= avail ? .red : .secondary)
                                }
                                Spacer()
                                if isSelected && state.loading {
                                    ProgressView()
                                } else if isSelected && state.engine != nil {
                                    Label("Loaded", systemImage: "checkmark.circle.fill").font(.callout).foregroundStyle(.green)
                                } else {
                                    Text("Load").font(.callout).foregroundStyle(Color.accentColor)
                                }
                            }
                        }
                        .disabled(state.running || state.loading || state.suiteStep != nil || (isSelected && state.engine != nil))
                    }
                    Button("Import a .gguf…") { importing = true }.disabled(state.running || state.loading || state.suiteStep != nil)
                } header: {
                    Text("Choose a model")
                } footer: {
                    Text(state.models.isEmpty ? "No models yet: download one at the bottom of this page."
                         : "Tap a model to load it; the loaded one is marked. Loading replaces the previous model.")
                }

                if let engine = state.engine {
                    Section("Arms for \(modelTitle(state.selected?.lastPathComponent ?? ""))") {
                        Picker("A", selection: $state.armA) {
                            ForEach(Presets.all(for: engine.weightType, mtp: engine.hasMTP)) { Text($0.name).tag($0) }
                        }
                        Picker("B", selection: $state.armB) {
                            ForEach(Presets.all(for: engine.weightType, mtp: engine.hasMTP)) { Text($0.name).tag($0) }
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
                        Stepper(String(format: "Cooldown: %.0f s", state.cooldown), value: $state.cooldown, in: 0...180, step: 10)
                        Toggle("Start quartets only when nominal", isOn: $state.waitForNominal)
                        Picker("Prompt micro-batch (pp cells)", selection: $state.promptUbatch) {
                            Text("512 (as on the Mac)").tag(512)
                            Text("256").tag(256)
                            Text("128").tag(128)
                        }
                        Stepper(String(format: "Spread gate: %.2f", state.gate), value: $state.gate, in: 1.05...2.0, step: 0.05)
                    }
                    .disabled(state.running)
                    Section {
                        if state.suiteStep != nil {
                            Text("The phone suite is running (see above).").foregroundStyle(.secondary)
                        } else if state.running {
                            if !state.progress.isEmpty {
                                Text(state.progress).font(.callout.monospacedDigit())
                            }
                            Button("Stop", role: .destructive) { state.stop() }
                        } else {
                            Button("Run A-B-B-A on \(modelTitle(state.selected?.lastPathComponent ?? ""))") { state.start() }
                                .disabled(state.cells.isEmpty || downloads.busy || state.loading || state.suiteStep != nil)
                        }
                    } footer: {
                        Text(downloads.busy ? "Wait for downloads and checks to finish (or cancel them): they would skew the timings."
                             : "Keep the app in the foreground and the phone on a stable surface; thermal state is recorded with every observation.")
                    }
                }

                }
                if let r = state.result {
                    Section {
                        ForEach(r.summaries, id: \.cell) { s in
                            VStack(alignment: .leading) {
                                if let geo = s.speedupGeomean, let a = s.aMean, let b = s.bMean {
                                    Text(String(format: "%@  %.2f → %.2f tok/s   ×%.3f", s.cell, a, b, geo)).font(.body.monospaced())
                                } else {
                                    Text("\(s.cell)  no accepted quartet").font(.body.monospaced())
                                }
                                Text(String(format: "range %.3f–%.3f · %d accepted, %d rejected%@%@%@", s.speedupMin ?? 0, s.speedupMax ?? 0,
                                            s.acceptedQuartets, s.rejectedQuartets, s.complete ? "" : " · incomplete",
                                            s.tokenMismatchQuartets > 0 ? " · tokens differ" : "",
                                            s.mtpTokenDifferenceQuartets > 0 ? " · plain and MTP tokens differ (expected without invariant mode)" : ""))
                                    .font(.caption).foregroundStyle(s.complete ? Color.secondary : Color.orange)
                                if s.aAcceptance != nil || s.bAcceptance != nil {
                                    Text("MTP draft acceptance: " + [("A", s.aAcceptance), ("B", s.bAcceptance)]
                                        .compactMap { n, v in v.map { String(format: "%@ %.1f%%", n, $0 * 100) } }.joined(separator: ", "))
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                        LabeledContent("Peak footprint", value: gb(r.peakFootprintBytes))
                        if let url = state.resultURL { ShareLink("Share JSON", item: url) }
                    } header: {
                        Text("Results · \(modelTitle(r.model))")
                    } footer: {
                        Text("B = \(r.armB.name), A = \(r.armA.name); ×speedup is B/A.")
                    }
                }

                if !state.log.isEmpty {
                    Section("Log") {
                        ForEach(Array(state.log.suffix(200).enumerated()), id: \.offset) { _, line in
                            Text(line).font(.caption.monospaced())
                        }
                    }
                }

                Section {
                    ForEach(Catalog.models) { m in
                        let url = state.documents.appendingPathComponent(m.file)
                        // compare names: the listing and `documents` can differ in form (/var vs /private/var)
                        let installed = state.models.contains { $0.lastPathComponent == m.file }
                        DownloadRow(model: m, installed: installed, verified: installed && VerifiedMark.get(url) == m.sha256,
                                    available: state.device.appAvailableMemoryBytes,
                                    busy: state.running || state.loading || state.suiteStep != nil, downloads: downloads)
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
                    Text("\(gb(model.bytes)) file · ~\(gb(need)) app memory\(fit)")
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

/// "Bonsai 2 ternary (PTQ1_0)" for a catalog file ("... + MTP head" for its grafted MTP GGUF), else the
/// file name without its extension.
func modelTitle(_ file: String) -> String {
    if let m = Catalog.models.first(where: { $0.file == file }) { return m.title }
    if file.hasSuffix("-mtp.gguf"), let m = Catalog.models.first(where: { $0.file == file.replacingOccurrences(of: "-mtp.gguf", with: ".gguf") }) {
        return m.title + " + MTP head"
    }
    return (file as NSString).deletingPathExtension
}
