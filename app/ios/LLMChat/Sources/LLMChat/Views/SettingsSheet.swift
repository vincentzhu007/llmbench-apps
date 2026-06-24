import SwiftUI

struct SettingsSheet: View {
    @EnvironmentObject var vm: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Generation") {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Temperature")
                            Spacer()
                            Text(String(format: "%.2f", vm.temperature))
                                .foregroundStyle(.secondary)
                                .font(.caption.monospaced())
                        }
                        Slider(value: $vm.temperature, in: 0.0...2.0, step: 0.05)
                    }

                    Stepper("Max tokens: \(vm.maxTokens)", value: $vm.maxTokens, in: 64...4096, step: 64)
                }

                Section("Model") {
                    LabeledContent("Name", value: vm.model.displayName)
                    LabeledContent("Quant", value: vm.model.quant)
                    LabeledContent("Params", value: vm.model.params)
                    LabeledContent("Vocab", value: vm.model.vocab)
                    if let rom = vm.model.romSize {
                        LabeledContent("ROM size", value: rom)
                    }
                    LabeledContent("Est. decode", value: "~\(vm.model.estimatedDecodeTokPerSec) tok/s")
                }

                Section("About") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("LLMChat Gallery")
                            .font(.headline)
                        Text("Powered by Apple Foundation Models & Core AI.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("Models run entirely on-device. Prefill & decode speed are measured live.")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
