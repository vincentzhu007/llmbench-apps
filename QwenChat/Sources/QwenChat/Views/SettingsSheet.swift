import SwiftUI

struct SettingsSheet: View {
    @EnvironmentObject var vm: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                // Generation Settings
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

                    Stepper("Max Tokens: \(vm.maxTokens)", value: $vm.maxTokens, in: 64...4096, step: 64)
                }

                // Model Info
                Section("Model") {
                    LabeledContent("Name", value: vm.modelName)
                    LabeledContent("Size", value: vm.modelSize)
                    LabeledContent("Format", value: vm.modelFormat)
                    LabeledContent("Vocab", value: vm.modelVocab)
                    LabeledContent("Speed", value: vm.modelSpeed)
                }

                // About
                Section("About") {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Qwen Chat")
                            .font(.headline)
                        Text("Powered by Apple Foundation Models & Core AI")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text("Model runs entirely on-device. No network required.")
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
