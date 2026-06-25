import SwiftUI

/// Per-model chat screen. Pushed from the Gallery. Owns its own `ChatViewModel`
/// bound to one `ModelDescriptor`.
struct ChatScreen: View {
    @StateObject private var vm: ChatViewModel
    @State private var showSettings = false

    init(model: ModelDescriptor) {
        _vm = StateObject(wrappedValue: ChatViewModel(model: model))
    }

    var body: some View {
        VStack(spacing: 0) {
            ModelCard()
                .padding(.horizontal)
                .padding(.top, 8)

            if let avg = vm.avgThroughput {
                SessionStatsBar(avgThroughput: avg)
                    .padding(.horizontal)
                    .padding(.top, 6)
                    .transition(.opacity)
            }

            Divider().padding(.top, 8)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 0) {
                        if vm.messages.isEmpty {
                            EmptyStateView(modelName: vm.model.displayName)
                                .padding(.top, 60)
                        }
                        ForEach(vm.messages) { msg in
                            ChatBubble(message: msg)
                                .id(msg.id)
                                .padding(.horizontal)
                                .padding(.vertical, 4)
                        }
                        if vm.isGenerating {
                            ChatBubble(
                                message: ChatMessage(role: .assistant, content: vm.streamingText),
                                isStreaming: true,
                                livePhase: vm.streamingText.isEmpty ? .warmingUp : .generating,
                                liveTokPerSec: vm.liveTokPerSec
                            )
                            .padding(.horizontal)
                            .padding(.vertical, 4)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                }
                .onChange(of: vm.messages.count) { _, _ in
                    withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
                }
                .onChange(of: vm.streamingText) { _, _ in
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }

            InputBar()
        }
        .environmentObject(vm)
        .navigationTitle(vm.model.displayName)
        .toolbar {
            ToolbarItem(placement: .automatic) {
                HStack(spacing: 6) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 8, height: 8)
                    Text(statusText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer().frame(width: 12)
                    Button { vm.clearChat() } label: {
                        Image(systemName: "trash").font(.caption)
                    }
                    Button { showSettings = true } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                }
            }
        }
        .sheet(isPresented: $showSettings) { SettingsSheet() }
        .task {
            if case .idle = vm.modelLoadState {
                await vm.loadModel()
            }
        }
    }

    private var statusColor: Color {
        if !vm.isAvailable { return .red }
        return vm.modelLoadState == .ready ? .green : .orange
    }

    private var statusText: String {
        if !vm.isAvailable { return "Model not found" }
        return vm.modelLoadState == .ready ? "Connected" : vm.modelLoadState.rawValue
    }
}

// MARK: - Session stats

private struct SessionStatsBar: View {
    let avgThroughput: Double

    var body: some View {
        HStack(spacing: 8) {
            Label("Avg \(Int(avgThroughput)) tok/s", systemImage: "bolt.fill")
                .foregroundStyle(.orange)
            Text("(S=1: prefill≈decode)")
                .foregroundStyle(.tertiary)
                .font(.system(size: 10, design: .monospaced))
            Spacer()
        }
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Empty state

private struct EmptyStateView: View {
    let modelName: String
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 48))
                .foregroundStyle(.purple.opacity(0.6))
            Text("\(modelName) ready")
                .font(.title2.bold())
            Text("Ask anything — running locally.\nPrefill & decode speed are measured per reply.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }
}

// MARK: - Input bar

private struct InputBar: View {
    @EnvironmentObject var vm: ChatViewModel
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(alignment: .bottom, spacing: 12) {
                // Keep the TextField itself free of material/clipShape modifiers:
                // on iOS 27 beta those broke hit-testing so tapping never focused
                // it. The rounded background is a plain Color fill instead.
                TextField("Message…", text: $vm.inputText)
                    .textFieldStyle(.plain)
                    .focused($inputFocused)
                    .submitLabel(.send)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.secondary.opacity(0.18))
                    )
                    .onSubmit { Task { await vm.send() } }

                Button {
                    Task { await vm.send() }
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 32))
                        .foregroundStyle(canSend ? .blue : Color.secondary)
                }
                .disabled(!canSend)
            }
            .padding(.horizontal)
            .padding(.vertical, 10)
        }
        .background(.regularMaterial)
    }

    private var canSend: Bool {
        vm.modelLoadState == .ready
            && !vm.isGenerating
            && !vm.inputText.trimmingCharacters(in: .whitespaces).isEmpty
    }
}
