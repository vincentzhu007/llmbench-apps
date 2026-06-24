import SwiftUI

struct ContentView: View {
    @EnvironmentObject var vm: ChatViewModel
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Model Card
                ModelCard()
                    .padding(.horizontal)
                    .padding(.top, 8)

                // Divider
                Divider()
                    .padding(.top, 8)

                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            if vm.messages.isEmpty {
                                EmptyStateView()
                                    .padding(.top, 60)
                            }
                            ForEach(vm.messages) { msg in
                                ChatBubble(message: msg)
                                    .id(msg.id)
                                    .padding(.horizontal)
                                    .padding(.vertical, 4)
                            }
                            if vm.isGenerating && !vm.streamingText.isEmpty {
                                ChatBubble(message: ChatMessage(role: .assistant, content: vm.streamingText), isStreaming: true)
                                    .padding(.horizontal)
                                    .padding(.vertical, 4)
                            }
                        }
                    }
                    .onChange(of: vm.messages.count) { _, _ in
                        if let last = vm.messages.last {
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                    }
                }

                // Input Bar
                InputBar()
            }
            .navigationTitle("Qwen Chat")
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(vm.modelLoadState == .ready ? Color.green : Color.orange)
                            .frame(width: 8, height: 8)
                        Text(vm.modelLoadState == .ready ? "Connected" : vm.modelLoadState.rawValue)
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
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheet()
        }
        .task {
            if case .idle = vm.modelLoadState {
                await vm.loadModel()
            }
        }
    }
}

// MARK: - Empty State

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 48))
                .foregroundStyle(.purple.opacity(0.6))
            Text("Qwen3-0.6B ready")
                .font(.title2.bold())
            Text("Ask anything — I'm running locally\non Apple Neural Engine")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }
}

// MARK: - Input Bar

struct InputBar: View {
    @EnvironmentObject var vm: ChatViewModel

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(alignment: .bottom, spacing: 12) {
                TextField("Message...", text: $vm.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...5)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 20))

                Button {
                    Task { await vm.send() }
                } label: {
                    Image(systemName: vm.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.system(size: 32))
                        .foregroundStyle(
                            vm.inputText.trimmingCharacters(in: .whitespaces).isEmpty
                                ? Color.secondary : Color.blue
                        )
                }
                .disabled(vm.inputText.trimmingCharacters(in: .whitespaces).isEmpty || vm.isGenerating)
            }
            .padding(.horizontal)
            .padding(.vertical, 10)
        }
        .background(Material.bar)
    }
}
