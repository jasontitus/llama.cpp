import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var state = BenchState()
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
                        .disabled(state.running)
                    }
                    Button("Import a .gguf…") { importing = true }.disabled(state.running)
                } header: {
                    Text("Models")
                } footer: {
                    Text(state.status)
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
                    Section {
                        if state.running {
                            Button("Stop after this observation", role: .destructive) { state.stop() }
                        } else {
                            Button("Run A-B-B-A") { state.start() }.disabled(state.cells.isEmpty)
                        }
                    } footer: {
                        Text("Keep the app in the foreground and the phone on a stable surface; thermal state is recorded with every observation.")
                    }
                }

                if let r = state.result {
                    Section("Results (B vs A)") {
                        ForEach(r.summaries, id: \.cell) { s in
                            VStack(alignment: .leading) {
                                Text(String(format: "%@  %.2f → %.2f tok/s   ×%.3f", s.cell, s.aMean, s.bMean, s.speedupGeomean)).font(.body.monospaced())
                                Text(String(format: "range %.3f–%.3f · %d accepted, %d rejected%@", s.speedupMin, s.speedupMax,
                                            s.acceptedQuartets, s.rejectedQuartets, s.tokenMismatchQuartets > 0 ? " · tokens differ" : ""))
                                    .font(.caption).foregroundStyle(.secondary)
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
            .onAppear { state.refresh() }
            .refreshable { state.refresh() }
            .fileImporter(isPresented: $importing, allowedContentTypes: [.data]) { res in
                if case .success(let url) = res { state.importModel(url) }
            }
        }
    }
}
